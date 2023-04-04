'''
    for segmenting hour long audio, ecg, imu to audio chunks

'''

import argparse
import math
import time
import torch, torchaudio
from pathlib import Path
import pandas as pd
import numpy as np
from sleep_classifier_hal import read_avg_hr
from synchronize import create_chunks, load_audio_chunks, load_ecg_chunks, load_imu, load_sleep_label, align

def get_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--annotation_dir', default="F:\\datasets\\littlebeats\\sleep annotation clean")
    parser.add_argument('--data_dir', default="F:\\datasets\\littlebeats\\sleep_study_preliminary_recordings")
    parser.add_argument('--output_dir', default="F:\\datasets\\littlebeats\\segmented_data_clean")
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--dir_count', type=int, default=10000)
    args = parser.parse_args()
    return args

def load_data(imu_file, imu_timestamp_file, ecg_file, ecg_timestamp_file, audio_file, audio_timestamp_file, audio_textgrid_file, interval=30):

    audio_wav, audio_sr, audio_timestamp = load_audio_chunks([audio_file], [audio_timestamp_file], resample_rate=16000)
    ecg_wav, ecg_sr, ecg_timestamp = load_ecg_chunks(ecg_file, ecg_timestamp_file)
    imu_data, imu_sr, imu_timestamp = load_imu(imu_file, imu_timestamp_file)

    audio_wav, ecg_wav, imu_data = align(audio_wav, audio_sr, audio_timestamp, ecg_wav, ecg_sr, ecg_timestamp, imu_data, imu_sr, imu_timestamp)

    num_data = math.floor(audio_wav.shape[0] / audio_sr / interval)
    audio_data = audio_wav[:num_data * interval * audio_sr].view(num_data, interval * audio_sr)
    ecg_data = ecg_wav[:num_data * interval * ecg_sr].view(num_data, interval * ecg_sr)
    for i, (key, val) in enumerate(imu_data.items()):
        imu_data[key] = val[:num_data * interval * imu_sr].view(num_data, interval * imu_sr)
    label = load_sleep_label(num_data, audio_textgrid_file, interval)
    return audio_data.to(torch.float), audio_sr, ecg_data, ecg_sr, imu_data, imu_sr, label

def create_df(audio_dir, label_path, prefix='audio', postfix='wav'):
    data = []
    label_df = pd.read_csv(label_path, delimiter=",", header=None)

    for index, row in label_df.iterrows():
        idx_name = f'{str(row.iloc[0]).zfill(6)}'
        name = f'{prefix}_{idx_name}'
        path = audio_dir / f'{name}.{postfix}'
        label = 'wake' if row.iloc[1]==0 else 'sleep'
        data.append({
            "name": name,
            "path": path,
            "class": label
        })
    return pd.DataFrame(data)

def make_split(dir_list, data_dir, output_dir, interval):
    label_file = open(output_dir / 'label.csv', mode='w')
    avg_hr_file = open(output_dir / 'avg_hr.csv', mode='w')
    idx = 0

    for dir in dir_list:
        dir_name = dir.name
        orig_audio_folder = data_dir / dir_name / "Audio_cleaned"
        orig_ecg_folder = data_dir / dir_name / "ECG_cleaned"
        orig_imu_folder = data_dir / dir_name / "IMU_cleaned"
        for file in orig_ecg_folder.iterdir():
            if (file.match('*cleaned.wav')):
                ecg_file = file
            if ('timestamp' in file.name):
                ecg_timestamp_file = file
        for file in orig_imu_folder.iterdir():
            if (file.match('*cleaned.txt')):
                imu_file = file
            if ('timestamp' in file.name):
                imu_timestamp_file = file
        avg_hr = read_avg_hr(dir)
        for file in dir.iterdir():
            if (file.match('*.wav')):
                num = file.name.split('cleaned_')[1].split("_")[0]
                for f in orig_audio_folder.iterdir():
                    if (('_Audio_timestamps_' + num + ".txt") in f.name):
                        audio_timestamp_file = f
                for f in dir.iterdir():
                    if (f'cleaned_{num}' in f.name and "TextGrid" in f.name):
                        audio_textgrid_file = f

                audio_x, audio_sr, ecg_x, ecg_sr, imu_data, imu_sr, y = load_data(imu_file, imu_timestamp_file,
                                                                                  ecg_file, ecg_timestamp_file,
                                                                                  file, audio_timestamp_file,
                                                                                  audio_textgrid_file,
                                                                                  interval=interval)
                idx = create_chunks(audio_x, audio_sr, ecg_x, ecg_sr, imu_data, imu_sr, y, output_dir, label_file,
                                    idx, avg_hr, avg_hr_file)
                print(f'{file.name}')
    label_file.close()
    avg_hr_file.close()


def main():
    '''
        output_dir
            -audio
                -audio_000001.wav
                -audio_000002.wav
            -ecg
                -ecg_000001.wav
                -ecg_000002.wav
            -accz
                -?
            -avg_hr.csv
            -label.csv (0 non-sleep, 1 sleep)

    '''
    args = get_arguments()

    annotation_dir = Path(args.annotation_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Segmenting cross validation dataset
    dir_list = np.array([x for x in (annotation_dir / 'train').iterdir() if x.is_dir() and not (x.name == "No Sleep files") and not ('Processed ECG' in x.name)])
    n_dir = len(dir_list)
    n_per_split = n_dir // 3
    np.random.shuffle(dir_list)
    dir_split = {}
    dir_split['train_1'], dir_split['train_2'], dir_split['train_3'] = dir_list[:n_per_split], dir_list[n_per_split:2 * n_per_split], dir_list[2 * n_per_split:]

    for split in ['train_1', 'train_2', 'train_3']:
        sub_output_dir = output_dir/split
        sub_output_dir.mkdir(parents=True, exist_ok=True)
        make_split(dir_split[split], data_dir, sub_output_dir, args.interval)

    # Segmenting normal datasets
    for split in ['train', 'test', 'val']:
        dir_list = np.array([x for x in (annotation_dir / split).iterdir() if x.is_dir() and not (x.name == "No Sleep files") and not ('Processed ECG' in x.name)])
        sub_output_dir = output_dir / split
        sub_output_dir.mkdir(parents=True, exist_ok=True)
        make_split(dir_list, data_dir, sub_output_dir, args.interval)

if __name__ == '__main__':
    main()