# Pretraining

## final comments

-   parameter sharing for gpt2 doesn't work as well.
-   algpt2 plateaus a lot faster that gpt2
-   chance that something is wrong with the code for algpt2 but unlikely
-   also a small chance that more train iterations are needed

## results

-   bad on wikitext2 zero-shot
-   cant generate coherent text
-   but text shouldn't be that bad for 30/25ppl?

## todo

-   do val on separate server
    -   algpt2-medium: 3.26
-   generate with algpt2-medium
    -   still bad
-   check generation
-   get old gpt2 checkpoint
-   finetune on wikitext
    -   not good

## experiments

-   gpt2-124m
-   algpt2
    -   shared parameters
        -   about 10ppl worse
    -   dropout
        -   not much difference
    -   larger models
        -   algpt2-medium can't compare with gpt2-medium
    -   embedding size
        -   128
        -   256
        -   512
    -   distilalgpt2/smallalgpt2?
-   results won't be as good but its good for the parameter count
    -   not really

## algpt2

-   just increasing the number of layers doesn't help almost at all
-   increasing the factorized embedding size helps more but its still no where close to as good
    -   might be because bottleneck layer params arent shared
    -   not using it is almost as good as gpt2-mediums
-   compare a gpt2-124 and algpt2 on ~100k iterations?
-   remove dropout and use different factorized embeddings

## notes

-   grover uses padding
-   megatron lm: https://arxiv.org/pdf/1909.08053.pdf
-   scaling laws paper: https://arxiv.org/pdf/2001.08361.pdf
-   gpt2: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
-   algpt2-124m doesn't really go below 4.5 loss
    -   but only tried ~60k iterations so far. more might be needed
    -   gpt2-124 should get below 3
-   factorized embedding size to #params
    -   128: 6.6e6
    -   256: 13e6
    -   512: 26e6
    -   1024: 53e6
    -   none: 80e6
-   possible but v unlikely that small files are making loss not go down as fast
-   next time:
    -   use adamw
        -   a lot better
    -   gpt2
        -   no big difference
    -   use adafactor beta1: 0.9
    -   use larger learning rate: 1e-3
        -   better but not that much
-   check processing
    -   missed about <30% of small examples
-   generation with train_pt with gpt2 fails b/c tokenizer and model aren't same
-   model.generate doesn't exactly work
-   doesn't look like there is anything wrong with text generation code, its just that the model isn't very good

## dataset

-   openwebtext
-   5% val split
-   40gb text
-   8m documents

## tokenization

-   byte level bpe
-   vocab size of 50257
-   ctrl uses a 5% split of data for training tokenizer
-   i trained tokenizer on 1m files
-   same tokenizer for gpt2 and algpt2

## preprocessing

-   add `<|endoftext|>` to beginning of each sequence
-   use `--min_seq_len` (have to)
-   seqlen 1024 means that ~67% of files are too small
-   seqlen 512 means that ~35% of files are too small
-   seqlen 256 means that ~10% of files are too small
-   eventually concat files with smaller seqlens together
    -   done
-   approach:
    -   docs smaller than seqlen get concated together with eos token prepended to each
    -   eos token gets added at beginning of sequence
-   train set
    -   53gb
    -   26gb of tfrecords
    -   7913604 files
    -   split into 16 tfrecords
    -   6885495 examples
-   val set
    -   339mb
    -   99998 files
    -   split into 16 tfrecords
    -   87011 examples

## model

-   seqlen 1024
-   start with gpt2-124M
-   then also pretrain algpt2-124m

## training

-   800k iterations
-   batch size: 512
-   linear warmup for 10k iterations
-   use adafactor with beta1 = 0
-   60 epochs; ~800k iterations

```
python3 train_tfrecords.py --tpu algpt2pod --seq_len 1024 --batch_size 512 --train_len 6885495 --warmup_steps 10000 --model_type algpt2 --config_path ./algpt2/algpt2/ --epochs 60 --train_path gs://algpt2/train/0.tfrecord gs://algpt2/train/1.tfrecord gs://algpt2/train/2.tfrecord gs://algpt2/train/3.tfrecord gs://algpt2/train/4.tfrecord gs://algpt2/train/5.tfrecord gs://algpt2/train/6.tfrecord gs://algpt2/train/7.tfrecord gs://algpt2/train/8.tfrecord gs://algpt2/train/9.tfrecord gs://algpt2/train/10.tfrecord gs://algpt2/train/11.tfrecord gs://algpt2/train/12.tfrecord gs://algpt2/train/13.tfrecord gs://algpt2/train/14.tfrecord gs://algpt2/train/15.tfrecord --val_path gs://algpt2/val/0.tfrecord gs://algpt2/val/1.tfrecord gs://algpt2/val/2.tfrecord gs://algpt2/val/3.tfrecord gs://algpt2/val/4.tfrecord gs://algpt2/val/5.tfrecord gs://algpt2/val/6.tfrecord gs://algpt2/val/7.tfrecord gs://algpt2/val/8.tfrecord gs://algpt2/val/9.tfrecord gs://algpt2/val/10.tfrecord gs://algpt2/val/11.tfrecord gs://algpt2/val/12.tfrecord gs://algpt2/val/13.tfrecord gs://algpt2/val/14.tfrecord gs://algpt2/val/15.tfrecord
```
