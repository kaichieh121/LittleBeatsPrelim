import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import torchaudio
from sklearn.model_selection import train_test_split

import os
import sys
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--data_dir', default='D:/Projects/LittleBeatsPrelim_HAL/LittleBeatsPrelim/data_30s')
    parser.add_argument('--output_dir', default='D:/Projects/LittleBeatsPrelim_HAL/LittleBeatsPrelim/manifest')
    args = parser.parse_args()
    return args

def create_df(audio_dir, label_path):
    data = []
    label_df = pd.read_csv(label_path, delimiter=",", header=None)

    for index, row in label_df.iterrows():
        idx_name = f'{str(row.iloc[0]).zfill(6)}'
        name = f'audio_{idx_name}'
        path = audio_dir / f'{name}.wav'
        label = 'wake' if row.iloc[1]==0 else 'sleep'
        data.append({
            "name": name,
            "path": path,
            "class": label
        })
    return pd.DataFrame(data)

def main():
    args = get_arguments()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = data_dir / 'audio'
    label_path = data_dir / 'label.csv'

    df = create_df(audio_dir, label_path)

    # create .csv manifest record files
    data_files = {
        "train": f"{output_dir/'train.csv'}",
        "validation": f"{output_dir/'test.csv'}",
    }
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=101, stratify=df["class"])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df.to_csv(data_files['train'], sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(data_files['validation'], sep="\t", encoding="utf-8", index=False)
    print(train_df.shape)
    print(test_df.shape)

if __name__ == "__main__":
    main()



