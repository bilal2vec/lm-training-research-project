import os
import re
import math
import glob
import time
import argparse
import pickle
import logging

import numpy as np
from tqdm import tqdm

import tensorflow as tf

from transformers import *

from detokenizer import wikitext_detokenizer


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(inputs, labels):
    feature = {
        'inputs': _int64_feature(inputs),
        'labels': _int64_feature(labels),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def _tokenize(l, args, tokenized_control_code, tokenizer, writer):
    n_examples = 0
    if args.use_control_codes:
        seqlen = args.seq_len - 1
    else:
        seqlen = args.seq_len

    for i in range(len(l) // seqlen):
        if args.use_control_codes:
            example = tokenizer.build_inputs_with_special_tokens(
                tokenized_control_code + l[i * seqlen: (i + 1) * seqlen])
        else:
            example = tokenizer.build_inputs_with_special_tokens(
                l[i * seqlen: (i + 1) * seqlen])

        inputs = example[:-1]
        labels = example[1:]

        example = serialize_example(inputs, labels)
        writer.write(example)

        n_examples += 1

    return n_examples


def tokenize(i, paths, tokenizer, args):
    start = time.time()

    tokenized_control_code = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(args.control_codes[0]))

    n_examples = 0
    with tf.io.TFRecordWriter(os.path.join(args.save_path, f'{i}.tfrecord')) as writer:
        small_files = []
        for path in tqdm(paths):
            text = []
            with open(path, encoding="utf-8") as handle:
                if args.line_by_line:
                    for line in handle:
                        text.append(line)
                else:
                    text.append(handle.read())

            text = tokenizer.batch_encode_plus(text)["input_ids"]

            for l in text:
                if args.min_seq_len:
                    if len(l) < args.seq_len:
                        if len(small_files) == 0:
                            small_files += l
                        else:
                            small_files += tokenized_control_code + l
                        continue

                n_examples += _tokenize(l, args,
                                        tokenized_control_code, tokenizer, writer)

            if args.min_seq_len:
                if len(small_files) >= args.seq_len:
                    n_examples += _tokenize(small_files, args,
                                            tokenized_control_code, tokenizer, writer)
                    small_files = []

    end = time.time()
    print(f'#examples: {n_examples}')
    print(f'chunk processed in {int(end - start)} seconds')

    return n_examples


def main():

    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', default='./data/wikitext-2/wiki.train.tokens',
                        type=str, required=False)

    parser.add_argument('--continue_from', default=-
                        1, type=int, required=False)

    parser.add_argument('--save_path', default='./', type=str, required=False)
    parser.add_argument('--files_per_tfrecord', default=1,
                        type=int, required=False)

    parser.add_argument('--use_control_codes', default=False,
                        action="store_true", required=False)
    parser.add_argument('--control_codes', nargs='+',
                        default=['<|endoftext|>'])

    parser.add_argument('--seq_len', default=256, type=int, required=False)
    parser.add_argument('--n_examples', default=-1, type=int, required=False)
    parser.add_argument('--min_seq_len', default=False, action='store_true')
    parser.add_argument('--line_by_line', default=False, action='store_true')

    parser.add_argument('--tokenizer', default='gpt2', type=str)

    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    tf.random.set_seed(args.seed)

    if args.debug:
        import ptvsd
        ptvsd.enable_attach(address=('localhost', 5678),
                            redirect_output=True)
        ptvsd.wait_for_attach()
        breakpoint()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': args.control_codes})

    start = time.time()

    n_examples = 0

    if os.path.isdir(args.path):
        files = glob.glob(os.path.join(args.path, '*'))
        print(f'Tokenizing {len(files)} files')

        for i in range(math.ceil(len(files) / args.files_per_tfrecord)):
            if args.continue_from > -1 and i < args.continue_from:
                continue

            files_subset = files[i *
                                 args.files_per_tfrecord: (i + 1) * args.files_per_tfrecord]

            n_examples += tokenize(i, files_subset, tokenizer, args)

            if args.n_examples > -1 and n_examples >= args.n_examples:
                print(f'Stopping at {n_examples} examples')
                break

    else:
        n_examples += tokenize(0, [args.path], tokenizer, args)

    end = time.time()

    print(f'Dataset created in {int(end - start)} seconds')
    print(f'#examples: {n_examples}')


if __name__ == "__main__":
    main()
