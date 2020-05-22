import os
from tqdm import tqdm

import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv('./IMDB Dataset.csv').values

    train_df = df[:int(len(df) * 0.9)]
    val_df = df[int(len(df) * 0.9):]

    with open('imdb-train.txt', 'w') as f:
        for i in tqdm(range(len(train_df))):
            f.write(f'{train_df[i][0]}\n\n')

    with open('imdb-val.txt', 'w') as f:
        for i in tqdm(range(len(val_df))):
            f.write(f'{val_df[i][0]}\n\n')

# kaggle datasets download lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
