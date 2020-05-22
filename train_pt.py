# Some code taken from Huggingface/transformers
import os
import pickle
import argparse
import time
import glob
import math

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torch.optim import SGD

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

import wandb

from transformers import GPT2LMHeadModel, CTRLLMHeadModel, GPT2TokenizerFast, CTRLTokenizer, AdamW, get_linear_schedule_with_warmup, AutoConfig, AutoModelWithLMHead, AutoTokenizer

from optimizers import Adafactor
from detokenizer import wikitext_detokenizer
from utils import n_params


class TextDataset(Dataset):
    def __init__(self, path, tokenizer, args):

        start = time.time()

        self.n_original_tokens = 0
        self.n_tokens = 0

        if os.path.isdir(path):
            self.batches = []
            for i, f in enumerate(glob.glob(os.path.join(path, '*.txt'))):
                self.batches += self._tokenize(f, tokenizer, args, i)
        else:
            self.batches = self._tokenize(path, tokenizer, args, 0)

        end = time.time()

        print(f'Dataset created in {int(end - start)} seconds')
        print(f'Dataset length: {len(self.batches)}')
        print(
            f'Num tokens: {self.n_tokens} | Num original tokens: {self.n_original_tokens}')

    def _tokenize(self, path, tokenizer, args, i):
        if args.use_control_codes:
            tokenized_control_code = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(args.control_codes[i]))
        else:
            tokenized_control_code = []

        batches = []

        text = []
        with open(path, encoding="utf-8") as handle:
            if args.efficient or args.fast:
                for line in handle:
                    self.n_original_tokens += len(line.split(" "))
                    if len(line) > 0 and not line.isspace():
                        if args.detokenizer:
                            text.append(wikitext_detokenizer(line))
                        else:
                            text.append(line)
            else:
                temp = handle.read()
                self.n_original_tokens += len(temp.strip().split(" "))

                if args.detokenizer:
                    text.append(wikitext_detokenizer(temp))
                else:
                    text.append(temp)

        if args.fast:
            batches = tokenizer.batch_encode_plus(
                text, add_special_tokens=True, max_length=args.seq_len-1)["input_ids"]
            batches = [tokenized_control_code + batch for batch in batches]
        else:
            text = tokenizer.batch_encode_plus(text)["input_ids"]
            for line in tqdm(text):
                if args.n_tokens > -1:
                    line = line[:args.n_tokens]

                if len(line) < args.seq_len - 1:
                    if not args.min_seq_len:
                        batches.append(
                            tokenizer.build_inputs_with_special_tokens(tokenized_control_code + line))
                else:
                    for i in range(math.ceil(len(line) / (args.seq_len - 1))):
                        batches.append(tokenizer.build_inputs_with_special_tokens(
                            tokenized_control_code + line[i * (args.seq_len - 1): (i + 1) * (args.seq_len - 1)]))

                        if args.n_batches > -1 and len(batches) >= args.n_batches:
                            break

        self.n_tokens += sum([len(batch) for batch in batches])

        return batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return torch.tensor(self.batches[index])


def sample(model, tokenizer, args):
    prompt = torch.tensor(tokenizer.encode(
        "<|endoftext|>")).unsqueeze(0).to(args.device)

    outputs = model.generate(input_ids=prompt, max_length=args.max_length, do_sample=args.do_sample, temperature=args.temperature,
                             top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty, num_return_sequences=args.n_samples)

    if args.use_sliding_windows:
        for i in range(args.n_sliding_windows):
            prompt = outputs[:, -args.sliding_window_size:]

            # num_return_sequences = 1 not a bug
            outputs_t = model.generate(input_ids=prompt, max_length=args.max_length, do_sample=args.do_sample, temperature=args.temperature,
                                       top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty, num_return_sequences=1)

            outputs = torch.cat(
                [outputs[:, :-args.sliding_window_size], outputs_t], dim=-1)

    print("Sampling:")
    for i in range(args.n_samples):
        print(f'Sample #{i}\n')
        output = tokenizer.decode(
            outputs[i].cpu().numpy(), skip_special_tokens=True)
        print(output)
        print("\n\n")


def run_sample(args):
    config = AutoConfig.from_pretrained(args.config)
    model = AutoModelWithLMHead.from_pretrained(
        args.checkpoint, from_tf=args.from_tf, config=config).to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    tokenizer.add_special_tokens(
        {'additional_special_tokens': args.control_codes})
    model.resize_token_embeddings(len(tokenizer))

    if args.fp16:
        model = model.half()
    model.eval()

    with torch.no_grad():
        sample(model, tokenizer, args)


def run_eval(args):
    config = AutoConfig.from_pretrained(args.config)
    model = AutoModelWithLMHead.from_pretrained(
        args.checkpoint, from_tf=args.from_tf, config=config).to(args.device)

    if args.fp16:
        model = model.half()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    tokenizer.add_special_tokens(
        {'additional_special_tokens': args.control_codes})
    model.resize_token_embeddings(len(tokenizer))

    val_dataset = TextDataset(args.val_path, tokenizer, args)

    def collate(examples):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate)

    val_loss = 0

    model.eval()
    with torch.no_grad():
        for j, batch in tqdm(enumerate(val_dataloader), total=int(len(val_dataset) / args.batch_size)):
            inputs, labels = batch.to(args.device), batch.to(args.device)

            out = model(inputs, labels=labels)
            loss = out[0]

            # check
            val_loss += loss.item()

    val_loss /= (j + 1)

    val_perplexity = torch.exp(torch.tensor(val_loss))
    adjusted_val_perplexity = torch.exp(torch.tensor(
        val_loss) * ((val_dataset.n_tokens - 1) / (val_dataset.n_original_tokens - 1)))

    # sample(model, tokenizer, args)

    message = f'Loss: {round(val_loss, 4)} | Perplexity: {round(val_perplexity.item(), 4)} | Adjusted Perplexity: {round(adjusted_val_perplexity.item(), 4)}'
    print(message)


def train(args):
    wandb.init(project="lm-finetuning", config=args, tags=args.tags)

    if args.save_dir == None:
        args.save_dir = wandb.run.dir

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    if args.from_scratch:
        config = AutoConfig.from_pretrained(args.config)
        model = AutoModelWithLMHead.from_config(config=config).to(args.device)
    else:
        model = AutoModelWithLMHead.from_pretrained(
            args.checkpoint, from_tf=args.from_tf).to(args.device)

    tokenizer.add_special_tokens(
        {'additional_special_tokens': args.control_codes})
    model.resize_token_embeddings(len(tokenizer))

    print(f"Params: {n_params(model)}")

    train_dataset = TextDataset(args.train_path, tokenizer, args)
    val_dataset = TextDataset(args.val_path, tokenizer, args)

    def collate(examples):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate)

    train_steps = int(len(train_dataloader) /
                      args.grad_steps * args.epochs)

    if args.optimizer == 'AdamW':
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Adafactor':
        optimizer = Adafactor(
            model.parameters(), lr=args.lr, beta1=0)

    # check
    if args.lr_schedule:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
            0.1 * train_steps), num_training_steps=train_steps)

    if os.path.exists(args.checkpoint):
        print('Loading optimizer and scheduler')

        optimizer.load_state_dict(torch.load(
            os.path.join(args.checkpoint, 'optimizer.pt')))

        if args.lr_schedule:
            scheduler.load_state_dict(torch.load(
                os.path.join(args.checkpoint, 'scheduler.pt')))

    if args.accelerator == 'GPU' and args.fp16 == True:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.apex_mode, loss_scale="dynamic")
    elif args.accelerator == 'TPU':
        import torch_xla.core.xla_model as xm

    wandb.watch(model, log='parameters')

    gradients = {}

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    if os.path.exists(args.checkpoint):
        global_step = int(args.checkpoint.split('-')[-1].split('/')[0])

        # check
        epochs_trained = global_step // (len(train_dataloader) //
                                         args.grad_steps)
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // args.grad_steps) * args.grad_steps

    for epoch in range(epochs_trained, args.epochs):
        train_loss = 0
        val_loss = 0

        print(f"Epoch: {epoch}")

        model.train()
        for i, batch in tqdm(enumerate(train_dataloader), total=int(len(train_dataset) / args.batch_size)):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, labels = batch.to(args.device), batch.to(args.device)

            out = model(inputs, labels=labels)
            loss = out[0]

            loss = loss / args.grad_steps

            # check
            train_loss += loss.item()

            if args.accelerator == 'GPU' and args.fp16 == True:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (i + 1) % args.grad_steps == 0:
                # check
                if args.accelerator == 'GPU' and args.fp16 == True:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), 1)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                # check
                if args.accelerator == 'TPU':
                    xm.optimizer_step(optimizer, barrier=True)
                else:
                    optimizer.step()

                if args.lr_schedule:
                    scheduler.step()

                if global_step % args.logging_steps == 0:

                    if args.lr_schedule:
                        lr = optimizer.param_groups[0]['lr']
                    else:
                        lr = scheduler.get_last_lr()[0]

                    wandb.log({"train_loss": loss.item() * args.grad_steps,
                               "learning_rate": lr}, step=global_step)

                    if global_step % args.hist_steps == 0:
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                try:
                                    gradients[f"gradients/{name}"] = wandb.Histogram(
                                        param.grad.detach().cpu().numpy())
                                except:
                                    pass

                    wandb.log(gradients, step=global_step)

                optimizer.zero_grad()

                global_step += 1

                # Must be in grad_accum block b/c if it is > 0, the model will get saved multiple times
                if global_step % args.save_steps == 0:
                    print(f'Saving model at global step: {global_step}')
                    checkpoint_dir = os.path.join(
                        args.save_dir, f'checkpoint-{global_step}')

                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)

                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    torch.save(optimizer.state_dict(), os.path.join(
                        checkpoint_dir, 'optimizer.pt'))

                    if args.lr_schedule:
                        torch.save(scheduler.state_dict(), os.path.join(
                            checkpoint_dir, 'scheduler.pt'))

        model.eval()
        with torch.no_grad():
            for j, batch in tqdm(enumerate(val_dataloader), total=int(len(val_dataset) / args.batch_size)):
                inputs, labels = batch.to(args.device), batch.to(args.device)

                out = model(inputs, labels=labels)
                loss = out[0]

                val_loss += loss.item()

        train_loss /= (i + 1)
        val_loss /= (j + 1)

        train_loss *= args.grad_steps

        train_perplexity = torch.exp(torch.tensor(train_loss))
        val_perplexity = torch.exp(torch.tensor(val_loss))
        adjusted_train_perplexity = torch.exp(torch.tensor(
            train_loss) * ((train_dataset.n_tokens - 1) / (train_dataset.n_original_tokens - 1)))
        adjusted_val_perplexity = torch.exp(torch.tensor(
            val_loss) * ((val_dataset.n_tokens - 1) / (val_dataset.n_original_tokens - 1)))

        sample(model, tokenizer, args)

        wandb.log({"train_epoch_loss": train_loss, "train_perplexity": train_perplexity, "adjusted_train_perplexity": adjusted_train_perplexity, 'val_epoch_loss': val_loss,
                   'val_perplexity': val_perplexity, 'adjusted_val_perplexity': adjusted_val_perplexity}, step=global_step)

        message = f'Finished epoch {epoch} | Train loss: {train_loss} | Train perplexity: {train_perplexity} | Adjusted Train perplexity: {adjusted_train_perplexity} | Val Loss: {val_loss} | Val Perplexity: {val_perplexity} | Adjusted Val Perplexity: {adjusted_val_perplexity}'
        print(message)

    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    torch.save(optimizer.state_dict(), os.path.join(
        args.save_dir, 'optimizer.pt'))

    if args.lr_schedule:
        torch.save(scheduler.state_dict(), os.path.join(
            args.save_dir, 'scheduler.pt'))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_path', default='./data/wikitext-2/wiki.train.tokens', type=str, required=False)
    parser.add_argument(
        '--val_path', default='./data/wikitext-2/wiki.valid.tokens', type=str, required=False)
    parser.add_argument('--save_dir', default=None,
                        type=str, required=False)

    parser.add_argument('--use_control_codes', default=False,
                        action="store_true", required=False)
    parser.add_argument('--control_codes', nargs='+',
                        default=['<|endoftext|>'])

    parser.add_argument('--seq_len', default=256, type=int, required=False)
    parser.add_argument('--n_tokens', default=-1, type=int, required=False)
    parser.add_argument('--n_batches', default=-1, type=int, required=False)
    parser.add_argument('--min_seq_len', default=False, action='store_true')
    # Uses fast tokenization
    parser.add_argument('--fast', default=False,
                        action="store_true", required=False)
    # Efficient for large datasets
    parser.add_argument('--efficient', default=False,
                        action="store_true", required=False)
    parser.add_argument('--detokenizer', default=False,
                        action="store_true", required=False)

    # if from scratch
    parser.add_argument('--from_scratch', default=False, action="store_true")
    parser.add_argument('--config', default='gpt2', type=str)

    # if from a pretrained model
    parser.add_argument('--checkpoint', default='distilgpt2', type=str)

    parser.add_argument('--tokenizer', default='gpt2', type=str)
    parser.add_argument('--from_tf', default=False, action="store_true")

    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr_schedule', default=True, type=bool)

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--grad_steps', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)

    # check
    # add multiple tpu cores
    parser.add_argument('--accelerator', default='GPU', type=str)
    parser.add_argument('--fp16', default=False, action="store_true")
    parser.add_argument('--apex_mode', default='O1', type=str)

    parser.add_argument('--logging_steps', default=10, type=int)
    parser.add_argument('--hist_steps', default=100, type=int)
    parser.add_argument('--save_steps', default=100, type=int)

    parser.add_argument('--do_sample', default=False, action="store_true")
    parser.add_argument('--n_samples', default=1, type=int)
    parser.add_argument('--max_length', default=256, type=int)
    parser.add_argument('--temperature', default=None, type=any)
    parser.add_argument('--top_k', default=None, type=any)
    parser.add_argument('--top_p', default=None, type=any)
    parser.add_argument('--repetition_penalty', default=None, type=any)

    parser.add_argument('--use_sliding_windows',
                        default=False, action="store_true")
    parser.add_argument('--n_sliding_windows', default=5, type=int)
    parser.add_argument('--sliding_window_size', default=128, type=int)

    parser.add_argument('--eval_only', default=False, action="store_true")
    parser.add_argument('--sample_only', default=False, action="store_true")
    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--tags', nargs='+')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.debug:
        import ptvsd
        ptvsd.enable_attach(address=('localhost', 5678),
                            redirect_output=True)
        ptvsd.wait_for_attach()
        breakpoint()

    if args.accelerator == 'TPU':
        import torch_xla.core.xla_model as xm

        args.device = xm.xla_device()
    else:
        args.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    if args.sample_only:
        run_sample(args)
    elif args.eval_only:
        run_eval(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
