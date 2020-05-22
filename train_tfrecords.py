import os
import re
import math
import glob
import time
import argparse
import pickle
import shutil

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
    'gpt2': TFGPT2LMHeadModel,
    'algpt2': TFALGPT2LMHeadModel
}


def get_dataset(args):
    feature_description = {
        'inputs': tf.io.FixedLenFeature((args.seq_len - 1), tf.int64),
        'labels': tf.io.FixedLenFeature((args.seq_len - 1), tf.int64),
    }

    def _parse_function(example_proto):
        x = tf.io.parse_single_example(example_proto, feature_description)
        return (x['inputs'], x['labels'])

    train_dataset = tf.data.TFRecordDataset(args.train_path)
    train_dataset = train_dataset.map(_parse_function).shuffle(
        1024).batch(args.batch_size, drop_remainder=True)

    val_dataset = tf.data.TFRecordDataset(args.val_path)
    val_dataset = val_dataset.map(_parse_function).shuffle(
        1024).batch(args.batch_size, drop_remainder=True)

    return train_dataset, val_dataset


class Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, dir, args, n_batch=0):
        super(Checkpoint, self).__init__()

        self.dir = dir
        self.args = args

        self.n_batch = n_batch

    def on_batch_end(self, batch, logs=None):
        if (self.n_batch + 1) % self.args.log_batches == 0:
            wandb.log({'train_batch_loss': logs.get('loss')},
                      step=self.n_batch + 1)
        if (self.n_batch + 1) % self.args.save_batches == 0:
            checkpoint_dir = os.path.join(
                self.dir, f'checkpoint-batch-{self.n_batch}')

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            self.model.save_pretrained(checkpoint_dir)
            print(f"saving model at iteration {self.n_batch}")

        self.n_batch += 1

    def on_epoch_end(self, epoch, logs=None):
        checkpoint_dir = os.path.join(
            self.dir, f'checkpoint-epoch-{self.n_batch}')

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.model.save_pretrained(checkpoint_dir)
        print(f"saving model at end of epoch {epoch}")

    def on_train_end(self, logs=None):
        checkpoint_dir = os.path.join(self.dir, 'final_epoch')

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.model.save_pretrained(checkpoint_dir)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--tpu', default='grpc://' +
                        os.environ['COLAB_TPU_ADDR'], required=False)

    parser.add_argument('--train_path', nargs='*',
                        default=['gs://lm-finetuning/wikitext-2/wiki.train.tokens.tfrecord'], required=False)
    parser.add_argument('--val_path', nargs='*',
                        default=['gs://lm-finetuning/wikitext-2/wiki.valid.tokens.tfrecord'], required=False)
    parser.add_argument('--train_len', default=100, type=int, required=False)
    parser.add_argument('--seq_len', default=256, type=int, required=False)

    parser.add_argument('--warmup_steps', default=10000,
                        type=int, required=False)

    parser.add_argument('--config_path', default='./', type=str)
    parser.add_argument('--model_type', default='gpt2', type=str)

    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--initial_epoch', default=None, type=int)

    parser.add_argument('--optimizer', default='Adafactor', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.0, type=float)
    parser.add_argument('--relative_update_scale',
                        default=False, action='store_true')
    parser.add_argument('--disable_lr_schedule',
                        default=False, action='store_true')

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=1, type=int)

    parser.add_argument('--save_batches', default=1000, type=int)
    parser.add_argument('--log_batches', default=10, type=int)

    parser.add_argument('--eval_only', default=False, action="store_true")

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
    wandb.init(project='lm-finetuning', config=args, tags=args.tags)

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)

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

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    config = AutoConfig.from_pretrained(args.config_path)
    model = AutoModelWithLMHead.from_config(config=config)
    os.mkdir('./temp')
    model.save_pretrained('./temp')

    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    with strategy.scope():
        model = MODEL_CLASSES[args.model_type]

        global_step = 0
        if args.checkpoint:
            global_step = int(args.checkpoint.split("-")[-1].split('/')[0])
            print(f'Starting from global step {global_step}')
            model = model.from_pretrained(args.checkpoint)
        else:
            model = model.from_pretrained('./temp', from_pt=True)

    model.compile(optimizer=optimizer, loss=[
        loss, *[None] * model.config.n_layer])

    model.summary()

    shutil.rmtree('./temp')

    if args.eval_only:
        feature_description = {
            'inputs': tf.io.FixedLenFeature((args.seq_len - 1), tf.int64),
            'labels': tf.io.FixedLenFeature((args.seq_len - 1), tf.int64),
        }

        def _parse_function(example_proto):
            x = tf.io.parse_single_example(example_proto, feature_description)
            return (x['inputs'], x['labels'])

        val_dataset = tf.data.TFRecordDataset(args.val_path)
        val_dataset = val_dataset.map(_parse_function).shuffle(
            1024).batch(args.batch_size, drop_remainder=True)

        out = model.evaluate(val_dataset)
        print(out)
    else:
        train_dataset, val_dataset = get_dataset(args)

        n_train_steps = (args.train_len // args.batch_size) * args.epochs

        wandb_callback = WandbCallback(save_model=False)
        checkpoint_callback = Checkpoint(wandb.run.dir, args, global_step)

        initial_epoch = 0
        if args.initial_epoch:
            initial_epoch = args.initial_epoch

        if args.disable_lr_schedule:
            model.fit(train_dataset, validation_data=val_dataset, epochs=args.epochs, callbacks=[
                wandb_callback, checkpoint_callback], initial_epoch=initial_epoch)
        else:
            lr_callback = WarmUpLinearDecayScheduler(
                learning_rate_base=args.lr, total_steps=n_train_steps, warmup_steps=args.warmup_steps, global_step_init=global_step)

            # model.fit(train_dataset, validation_data=val_dataset, epochs=args.epochs, callbacks=[
            #           wandb_callback, checkpoint_callback, lr_callback], initial_epoch=initial_epoch)
            model.fit(train_dataset, epochs=args.epochs, callbacks=[
                wandb_callback, checkpoint_callback, lr_callback], initial_epoch=initial_epoch)


if __name__ == "__main__":
    main()
