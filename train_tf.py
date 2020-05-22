import os
import re
import math
import glob
import time
import argparse
import pickle

import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops

from transformers import *

import wandb
from wandb.keras import WandbCallback

from optimizers_tf import *
from detokenizer import wikitext_detokenizer

MODEL_CLASSES = {
    'gpt2': (TFGPT2LMHeadModel, GPT2TokenizerFast)
}


class TextDataset(object):
    def __init__(self, path, tokenizer, args):

        start = time.time()

        self.n_original_tokens = 0
        self.n_tokens = 0

        if os.path.isdir(path):
            self.batches = []
            self.labels = []
            for i, f in enumerate(glob.glob(os.path.join(path, '*.txt'))):
                batches, labels = self._tokenize(f, tokenizer, args, i)
                self.batches += batches
                self.labels += labels
        else:
            self.batches, self.labels = self._tokenize(
                path, tokenizer, args, 0)

        end = time.time()

        print(f'Dataset created in {int(end - start)} seconds')
        print(f'Dataset length: {len(self.batches)}')
        print(
            f'Num tokens: {self.n_tokens} | Num original tokens: {self.n_original_tokens}')

    def _tokenize(self, path, tokenizer, args, i):
        batches = []
        labels = []

        text = []
        with open(path, encoding="utf-8") as handle:
            # efficient uses less memory by going line-by-line. Drawbacks: if len(line) isn't a multiple of seq_len, the remainder will be left
            if args.efficient:
                for line in handle:
                    self.n_original_tokens += len(line.split(" "))
                    if len(line) > 0 and not line.isspace():
                        if args.detokenizer:
                            text.append(wikitext_detokenizer(line))
                        else:
                            text.append(line)
            # Default way reads in entire file into memory
            else:
                temp = handle.read()
                self.n_original_tokens += len(temp.strip().split(" "))

                if args.detokenizer:
                    text.append(wikitext_detokenizer(temp))
                else:
                    text.append(temp)

        text = tokenizer.batch_encode_plus(text)["input_ids"]

        for line in tqdm(text):
            if args.n_tokens > -1:
                line = line[:args.n_tokens]

            if len(line) < args.seq_len - 1:
                if not args.min_seq_len:
                    example = tokenizer.build_inputs_with_special_tokens(line)

                    batches.append(example[:-1])
                    labels.append(example[1:])
            else:
                for i in range(len(line) // (args.seq_len - 1)):
                    example = tokenizer.build_inputs_with_special_tokens(
                        line[i * (args.seq_len - 1): (i + 1) * (args.seq_len - 1)])

                    batches.append(example[:-1])
                    labels.append(example[1:])

                    if args.n_batches > -1 and len(batches) >= args.n_batches:
                        break

        self.n_tokens += sum([len(batch) for batch in batches])

        return batches, labels


def get_dataset(args, tokenizer):

    if args.use_serialized:
        print('loading dataset from disk')
        start = time.time()

        train_dataset = pickle.load(open(args.train_path, 'rb'))
        val_dataset = pickle.load(open(args.val_path, 'rb'))

        end = time.time()
        print(f'Dataset loaded in {int(end - start)} seconds')

    else:
        train_dataset = TextDataset(args.train_path, tokenizer, args)
        val_dataset = TextDataset(args.val_path, tokenizer, args)

        pickle.dump(train_dataset, open(f'{args.train_path}.pkl', 'wb'))
        pickle.dump(val_dataset, open(f'{args.val_path}.pkl', 'wb'))

    train_n_original_tokens = train_dataset.n_original_tokens
    train_n_tokens = train_dataset.n_tokens

    val_n_original_tokens = val_dataset.n_original_tokens
    val_n_tokens = val_dataset.n_tokens

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_dataset.batches, train_dataset.labels))
    train_dataset = train_dataset.shuffle(100).batch(
        args.batch_size, drop_remainder=True)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (val_dataset.batches, val_dataset.labels))
    val_dataset = val_dataset.shuffle(100).batch(
        args.batch_size, drop_remainder=True)

    return tokenizer, train_dataset, val_dataset, train_n_original_tokens, train_n_tokens, val_n_original_tokens, val_n_tokens


class AdjLoss(object):
    def __init__(self, n_tokens, n_original_tokens, name):
        self.n_tokens = n_tokens
        self.n_original_tokens = n_original_tokens

        self.__name__ = name

    def __call__(self, y_true, y_pred):
        cross_entropy = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True)

        cross_entropy *= ((self.n_tokens - 1) / (self.n_original_tokens - 1))

        return cross_entropy


class PPL(object):
    def __init__(self, name):
        self.__name__ = name

    def __call__(self, y_true, y_pred):
        cross_entropy = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True)

        ppl = math.e ** K.mean(cross_entropy)

        return ppl


class AdjPPL(object):
    def __init__(self, n_tokens, n_original_tokens, name):
        self.n_tokens = n_tokens
        self.n_original_tokens = n_original_tokens

        self.__name__ = name

    def __call__(self, y_true, y_pred):
        cross_entropy = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True)

        cross_entropy *= ((self.n_tokens - 1) / (self.n_original_tokens - 1))

        ppl = math.e ** K.mean(cross_entropy)

        return ppl


class Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, dir):
        super(Checkpoint, self).__init__()

        self.dir = dir

    def on_epoch_end(self, epoch, logs=None):
        checkpoint_dir = os.path.join(self.dir, f'checkpoint-{epoch}')

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.model.save_pretrained(checkpoint_dir)

    def on_train_end(self, logs=None):
        checkpoint_dir = os.path.join(self.dir, 'final_epoch')

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.model.save_pretrained(checkpoint_dir)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', default='./data/wikitext-2/wiki.train.tokens',
                        type=str, required=False)
    parser.add_argument('--val_path', default='./data/wikitext-2/wiki.test.tokens',
                        type=str, required=False)

    parser.add_argument('--use_serialized', default=False, action='store_true')
    parser.add_argument('--use_control_codes', default=False,
                        action="store_true", required=False)
    parser.add_argument('--control_codes', nargs='+',
                        default=['<|endoftext|>'])

    parser.add_argument('--seq_len', default=256, type=int, required=False)
    parser.add_argument('--n_tokens', default=-1, type=int, required=False)
    parser.add_argument('--n_batches', default=-1, type=int, required=False)
    parser.add_argument('--efficient', default=False,
                        action="store_true", required=False)
    parser.add_argument('--min_seq_len', default=False, action='store_true')
    parser.add_argument('--detokenizer', default=False,
                        action="store_true", required=False)

    parser.add_argument('--model_type', default='gpt2', type=str)
    parser.add_argument('--model_name', default='distilgpt2', type=str)
    parser.add_argument('--checkpoint', default=None, type=str)

    parser.add_argument('--optimizer', default='Adafactor', type=str)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--momentum', default=0.0, type=float)
    parser.add_argument('--relative_update_scale',
                        default=False, action='store_true')
    parser.add_argument('--disable_lr_schedule',
                        default=False, action='store_true')

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--grad_steps', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)

    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--tags', nargs='+')

    args = parser.parse_args()

    tf.random.set_seed(args.seed)

    if args.debug:
        import ptvsd
        ptvsd.enable_attach(address=('localhost', 5678),
                            redirect_output=True)
        ptvsd.wait_for_attach()
        breakpoint()

    wandb.login()
    wandb.init(project='lm-finetuning', tags=args.tags)

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)

    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    with strategy.scope():
        model, tokenizer = MODEL_CLASSES[args.model_type]
        model = model.from_pretrained(args.model_name)

        tokenizer = tokenizer.from_pretrained(args.model_name)

        # Can't use since TF models don't have resize_token_embeddings implemented
        # tokenizer.add_special_tokens(
        #     {'additional_special_tokens': args.control_codes})
        # model.resize_token_embeddings(len(tokenizer))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if args.optimizer == "SGD":
        optimizer = keras.optimizers.SGD(lr=args.lr, momentum=args.momentum)
    if args.optimizer == "AdamW":
        optimizer = AdamWeightDecay(learning_rate=args.lr)
    elif args.optimizer == "Adafactor":
        if args.relative_update_scale:
            optimizer = AdafactorOptimizer(
                beta1=args.momentum, multiply_by_parameter_scale=True)
        else:
            optimizer = AdafactorOptimizer(
                learning_rate=args.lr, beta1=args.momentum)

    tokenizer, train_dataset, val_dataset, train_n_original_tokens, train_n_tokens, val_n_original_tokens, val_n_tokens = get_dataset(
        args, tokenizer)

    n_train_steps = int(len(list(train_dataset))) * args.epochs

    wandb_callback = WandbCallback(log_weights=True)
    checkpoint_callback = Checkpoint(wandb.run.dir)

    val_adj_loss = AdjLoss(val_n_tokens, val_n_original_tokens, 'val_adj_loss')
    val_ppl = PPL('val_ppl')
    val_adj_ppl = AdjPPL(val_n_tokens, val_n_original_tokens, 'val_adj_ppl')

    model.compile(optimizer=optimizer, loss=[
                  loss, *[None] * model.config.n_layer], metrics=[val_adj_loss, val_ppl, val_adj_ppl])

    if args.disable_lr_schedule:
        model.fit(train_dataset, validation_data=val_dataset, epochs=args.epochs, callbacks=[
                  wandb_callback, checkpoint_callback])
    else:
        lr_callback = WarmUpLinearDecayScheduler(
            learning_rate_base=args.lr, total_steps=n_train_steps, warmup_steps=int(0.1 * n_train_steps))

        model.fit(train_dataset, validation_data=val_dataset, epochs=args.epochs, callbacks=[
                  wandb_callback, checkpoint_callback, lr_callback])

    # outputs = model.generate(bos_token_id=tokenizer.bos_token_id, max_length=256, temperature=1.0,
    #                          top_k=None, top_p=None, repetition_penalty=None, num_return_sequences=1)
    # outputs = tokenizer.decode(
    #     outputs[0].cpu().numpy(), skip_special_tokens=True)
    # print(outputs)


if __name__ == "__main__":
    main()
