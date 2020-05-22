# albert style lm

## notes

-   uses a factorized embedding between input one-hot encoded vectors and hidden dim
    -   default embedding parameter of 128
-   shares parameters for all layers
-   uses a sentence-order-prediction auxillary loss
-   input and output embeddings are supposed to be tied (128x50257)
    -   https://github.com/huggingface/transformers/issues/2824
    -   https://github.com/huggingface/transformers/issues/1993
    -   https://github.com/huggingface/transformers/issues/1663
    -   https://huggingface.co/transformers/model_doc/gpt2.html#transformers.GPT2LMHeadModel
-   should the projection layer (128x768) also be tied?
    -   ALBERT only ties the embedding_dim x vocab_size embeddings
-   http://jalammar.github.io/illustrated-gpt2/
-   gpt2-124 with parameter sharing now only has 12M params
-   both projection and parameter sharing work well on wikitext2
    -   check if this extends to other datasets
-   writingprompts is ~20ppl worse
    -   algpt2 generated text is a lot worse for comparable ppls

```
gpt2-124M:
50257 x 768 = ~40M

50257 x 128 = ~6M
128 x 768 = ~100k
```

```
gpt2-xl:
50257 x 1600 = ~80M
50257 x 128 = ~6M
128 x 1600 = ~200k
```

-   35M less params is good for gpt2-124M, but it will only reduce gpt2-xl's #params by 75M

-   parameter sharing will be required

-   algpt2

```
algpt2: 13M
algpt2-medium: 19M
algpt2-large: 26M
algpt2-xl: 37M
```

## Todo

-   try removing dropout
-   increase embedding size
-   deal with multihead config params
-   make and upload config files
-   make conversion script
-   tf
    -   tf model
    -   tf pretrained models
-   pt
    -   pt model
    -   pt pretrained models
-   tokenizer
    -   fast
-   optimizer
    -   tf and pt
-   how to test new ideas

## Done

-   albert citations
    -   https://scholar.google.com/scholar?sxsrf=ALeKk03GFN1HorSwdI3OK59e3PFkwR5axA:1587936934057&gs_lcp=CgZwc3ktYWIQAzIECCMQJzIECCMQJzICCAAyAggAMgIIADICCAAyAggAMgcIABAUEIcCMgIIADICCAA6BAgAEEc6BAgAEEM6BQgAEJECOgUIABCDAToKCAAQgwEQFBCHAjoECAAQCjoHCCMQsAIQJzoECAAQDVCRBliKFmDSF2gDcAJ4AIABhQGIAeEHkgEDMC44mAEAoAEBqgEHZ3dzLXdpeg&uact=5&um=1&ie=UTF-8&lr&cites=6606720413006378435
