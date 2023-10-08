import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import torchaudio
from sklearn.model_selection import train_test_split

import os
import sys
import argparse

IDX2LABEL = {0: "Wake",
              1: "Non-REM",
              2: "REM"}

def get_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--data_dir', default='D:\\datasets\\littlebeats\\psg_data_clean')
    parser.add_argument('--output_dir', default='D:/Projects/LittleBeatsPrelim_HAL/LittleBeatsPrelim/manifest_psg')
    args = parser.parse_args()
    return args

def create_df(audio_dir, label_path, prefix='audio', postfix='wav'):
    data = []
    label_df = pd.read_csv(label_path, delimiter=",", header=None)

    for index, row in label_df.iterrows():
        idx_name = f'{str(row.iloc[0]).zfill(6)}'
        name = f'{prefix}_{idx_name}'
        path = audio_dir / f'{name}.{postfix}'
        label = IDX2LABEL[row.iloc[1]]
        data.append({
            "name": name,
            "path": path,
            "class": label
        })
    return pd.DataFrame(data)

def create_df_all(data_dir, label_path):
    data = []
    label_df = pd.read_csv(label_path, delimiter=",", header=None)

    for index, row in label_df.iterrows():
        idx_name = f'{str(row.iloc[0]).zfill(6)}'
        audio_path = data_dir / 'audio' / f'audio_{idx_name}.wav'
        ecg_path = data_dir / 'ecg' / f'ecg_{idx_name}.wav'
        # accz_path = data_dir / 'accz' / f'accz_{idx_name}.txt'
        imu_path = data_dir / 'imu' / f'imu_{idx_name}.txt'
        # uci_path = data_dir / 'uci' / f'uci_{idx_name}.npy'
        # hhar_path = data_dir / 'hhar' / f'hhar_{idx_name}.npy'
        # shoaib_path = data_dir / 'shoaib' / f'shoaib_{idx_name}.npy'
        # motion_path = data_dir / 'motion' / f'motion_{idx_name}.npy'
        label = IDX2LABEL[row.iloc[1]]
        data.append({
            "name": idx_name,
            "audio_path": audio_path,
            "ecg_path": ecg_path,
            # "accz_path": accz_path,
            # "uci_path": uci_path,
            # "hhar_path": hhar_path,
            # "shoaib_path": shoaib_path,
            # "motion_path": motion_path,
            "imu_path": imu_path,
            "class": label
        })
    return pd.DataFrame(data)

def main_full(args):
    data_top_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        data_dir = data_top_dir / split
        label_path = data_dir / 'label.csv'
        df = create_df_all(data_dir, label_path).reset_index(drop=True)
        data_file = f"{output_dir / f'{split}.csv'}"
        df.to_csv(data_file, sep="\t", encoding="utf-8", index=False)


def main_cross_validate(args):
    data_top_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = {}
    for split in ['train_1', 'train_2', 'train_3']:
        data_dir = data_top_dir / split
        label_path = data_dir / 'label.csv'
        df[split] = create_df_all(data_dir, label_path).reset_index(drop=True)
    # create .csv manifest record files
    train_df = pd.concat([df['train_1'], df['train_2']])
    data_file = f"{output_dir / f'train_1.csv'}"
    train_df.to_csv(data_file, sep="\t", encoding="utf-8", index=False)
    data_file = f"{output_dir / f'test_1.csv'}"
    df['train_3'].to_csv(data_file, sep="\t", encoding="utf-8", index=False)

    train_df = pd.concat([df['train_1'], df['train_3']])
    data_file = f"{output_dir / f'train_2.csv'}"
    train_df.to_csv(data_file, sep="\t", encoding="utf-8", index=False)
    data_file = f"{output_dir / f'test_2.csv'}"
    df['train_2'].to_csv(data_file, sep="\t", encoding="utf-8", index=False)

    train_df = pd.concat([df['train_2'], df['train_3']])
    data_file = f"{output_dir / f'train_3.csv'}"
    train_df.to_csv(data_file, sep="\t", encoding="utf-8", index=False)
    data_file = f"{output_dir / f'test_3.csv'}"
    df['train_1'].to_csv(data_file, sep="\t", encoding="utf-8", index=False)


if __name__ == "__main__":
    args = get_arguments()
    # main_cross_validate(args)
    main_full(args)



