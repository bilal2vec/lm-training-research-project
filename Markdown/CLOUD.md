# Cloud

## Instructions

TF:

```
gcloud compute instances create tf-lm-finetuning --zone=europe-west4-a --machine-type=n1-standard-8 --image=debian-10-tf-2-1-v20200316 --image-project=ml-images --boot-disk-size=100GB --preemptible
gcloud compute tpus create lm-finetuning --zone=europe-west4-a --version="2.1" --accelerator-type="v3-8"

ssh -i ~/.ssh/google_compute_engine bilal@[IP]
download vscode live share remote extensions
choose python3 python interpreter

git clone https://github.com/bkkaggle/lm-training-research-project.git
git checkout dev
pip3 install -r requirements.txt
wandb login

export COLAB_TPU_ADDR="10.160.101.114:8470"

mkdir ~/.kaggle
vim ~/.kaggle/kaggle.json
cd data/imdb
kaggle datasets download lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
unzip imdb-dataset-of-50k-movie-reviews.zip
python3 process.py

```

Preprocessing:

```
(use ssd)
gcloud compute instances create preprocessing --zone=europe-west4-a --machine-type=n1-standard-16 --image=debian-10-tf-2-1-v20200316 --image-project=ml-images --boot-disk-size=200GB

ssh -i ~/.ssh/google_compute_engine bilal@[IP]
download vscode live share remote extensions
choose python3 python interpreter

git clone https://github.com/bkkaggle/lm-training-research-project.git
git checkout dev
pip3 install -r requirements.txt
export PATH=$PATH:~/.local/bin
wandb login

gdown https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx
tar -xf openwebtext.tar.xz
cat *.xz | tar -J -xf - -i

(~7 hours on hdd)

python3 train_tokenizer.py --train_path ./data/openwebtext/ --save_path ./tokenizer/ --vocab_size 50257 --n_files 1000000
(~15m)

gsutil -m cp -r tokenizer gs://algpt2/
gsutil cp ./algpt2/algpt2/algpt2-config.json gs://algpt2/tokenizer/config.json

gsutil iam ch allUsers:objectViewer gs://algpt2

ls -f | head -100000 | xargs -i mv {} ../openwebtext-valid/

python3 make_tfrecords.py --path ./data/openwebtext/ --save_path ./train/ --files_per_tfrecord 500000 --use_control_codes --seq_len 1024 --min_seq_len --tokenizer ./tokenizer/

python3 make_tfrecords.py --path ./data/openwebtext/ --save_path ./train2/ --files_per_tfrecord 500000 --use_control_codes --seq_len 1024 --min_seq_len --tokenizer ./tokenizer/

(8h)

python3 make_tfrecords.py --path ./data/openwebtext-valid/ --save_path ./val/ --files_per_tfrecord 50000 --use_control_codes --seq_len 1024 --min_seq_len --tokenizer ./tokenizer/

(5m)

import tensorflow as tf
import numpy as np

ds = tf.data.TFRecordDataset(['./train/0.tfrecord', './train/1.tfrecord', './train/2.tfrecord', './train/3.tfrecord', './train/4.tfrecord', './train/5.tfrecord', './train/6.tfrecord', './train/7.tfrecord' , './train/8.tfrecord' , './train/9.tfrecord' , './train/10.tfrecord' , './train/11.tfrecord' , './train/12.tfrecord' , './train/13.tfrecord' , './train/14.tfrecord' , './train/15.tfrecord'])
cnt = ds.reduce(np.int64(0), lambda x, _: x + 1)

print(cnt)
```
