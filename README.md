# lm-training-research-project

- blog post part 1: https://bilal.software/2020/6/22/nlp-series-1/
- blog post part 2: https://bilal.software/2020/7/17/nlp-series-2/

This is my code for my lm pretraining research project that i worked on during quarantine. The idea was to apply ALBERT-style parameter-sharing and factorized embedding to GPT-2. I was able to (almost, 21 vs 17ppl) replicate OpenAI's ppl values for gpt2-124m when i pretrained it on openwebtext for 100k training iterations on a TPU v3-128, but my ALBERT-style gpt2 model didn't train too well (the best ppl values I got were around 30) and couldn't generate coherent-sounding text.

The code for my ALBERT-style gpt2 model can be found in my fork of huggingface/transformers: https://github.com/bkkaggle/transformers/tree/albert-style. The main files for the pytorch and tf versions of the models are [1](https://github.com/bkkaggle/transformers/blob/albert-style/src/transformers/configuration_algpt2.py), [2](https://github.com/bkkaggle/transformers/blob/albert-style/src/transformers/modeling_algpt2.py), [3](https://github.com/bkkaggle/transformers/blob/albert-style/src/transformers/modeling_tf_algpt2.py), [4](https://github.com/bkkaggle/transformers/blob/albert-style/src/transformers/tokenization_algpt2.py).

My Weights&Biases project: https://app.wandb.ai/bkkaggle/lm-finetuning

My pretrained gpt2-124m model training curves and checkpoints part [1](https://app.wandb.ai/bkkaggle/lm-finetuning/runs/2rbkadpc?workspace=user-bkkaggle) and [2](https://app.wandb.ai/bkkaggle/lm-finetuning/runs/1ta3sx61?workspace=user-bkkaggle).

My pretrained ALBERT-style gpt2-124 model with parameter-sharing training curves and checkpoints part [1](https://app.wandb.ai/bkkaggle/lm-finetuning/runs/ei53u955?workspace=user-bkkaggle) and [2](https://app.wandb.ai/bkkaggle/lm-finetuning/runs/2n7j6out?workspace=user-bkkaggle)

The vocab files for my byte-level BPE tokenizer trained on a subset of openwebtext can be found at `./algpt2/algpt2-tokenizer/`

I have a colab notebook: https://colab.research.google.com/drive/1PuaMB3meZJUXy1MpaiWW0F06IUTzbYaB?usp=sharing

A rough outline of all the steps I used to process the data and train the models are in `./Markdown/CLOUD.md`.

I have pytorch and tf pretraining and finetuning scripts `train_pt.py` and `train_tf.py`, as well as `train_tfrecords.py` which is what i used to pretrain on a TPU pod.

`train_tokenizer.py` will train a byte-level BPE tokenizer on openwebtext.

`make_tfrecords.py` will process the raw openwebtext data to a set of tfrecords files that can be uploaded to a GCP bucket.

A full pretraining run with 100k iterations takes about 10-20 hours on a TPU v3-128, depending on the model that you choose.

To avoid a lot of extra charges, make sure that your GCP bucket is in the same zone as your TPU pod and turn off stackdriver logging for your GCP project.

## Acknowledgements

Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC)
