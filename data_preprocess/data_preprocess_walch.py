'''
    for segmenting hour long audio, ecg, imu to audio chunks

'''

import argparse
import math
import time
import torch, torchaudio
import heartpy as hp
from pathlib import Path
import pandas as pd
import numpy as np
from synchronize import create_chunks, load_audio_chunks, load_ecg_chunks, load_imu, load_sleep_label, align
from tqdm import tqdm

def get_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--annotation_dir', default="D:\\datasets\\littlebeats\\sleep annotation walch")
    parser.add_argument('--data_dir', default="D:\\datasets\\littlebeats\\sleep_study_preliminary_recordings")
    parser.add_argument('--output_dir', default="D:\\datasets\\littlebeats\\segmented_data_walch")
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--dir_count', type=int, default=10000)
    args = parser.parse_args()
    return args

def write_data(ecg_wav, ecg_sr, ecg_timestamp, imu_data,imu_sr, imu_timestamp, label,interval, id, output_dir, timestamp_start):

    motion_dir = output_dir / 'motion'
    motion_dir.mkdir(parents=True, exist_ok=True)
    acce_output_file = open(motion_dir / f'{id}_acceleration.txt', mode='w')
    x_prev = None
    y_prev = None
    z_prev = None
    for i in range(imu_timestamp.shape[0]):
        time = imu_timestamp[i][0]
        idx = imu_timestamp[i][1]
        if imu_data[0][idx] != 0:
            x = imu_data[0][idx] / 10
            x_prev = x
        else:
            x = x_prev if x_prev is not None else 0
        if imu_data[1][idx] != 0:
            y = imu_data[1][idx] / 10
            y_prev = y
        else:
            y = y_prev if y_prev is not None else 0
        if imu_data[2][idx] != 0:
            z = imu_data[2][idx] / 10
            z_prev = z
        else:
            z = z_prev if z_prev is not None else 0
        acce_output_file.write(f'{time} {x} {y} {z}\n')
    acce_output_file.close()

    hr_dir = output_dir / 'heart_rate'
    hr_dir.mkdir(parents=True, exist_ok=True)
    hr_output_file = open(hr_dir / f'{id}_heartrate.txt', mode='w')
    hr_prev = None
    for i in range(ecg_timestamp.shape[0]):
        time = ecg_timestamp[i][0]
        start_idx = ecg_timestamp[i][1]
        end_idx = ecg_timestamp[i][3]
        ecg_data = ecg_wav[start_idx:end_idx]
        try:
            working_data, measures = hp.process(ecg_data.numpy(), ecg_sr)
            if(measures['bpm'] <= 250):
                hr = int(measures['bpm'])
                hr_prev = hr
            else:
                hr = hr_prev if hr_prev is not None else 150
        except:
            hr = hr_prev if hr_prev is not None else 150
        hr_output_file.write(f'{time},{hr}\n')
    hr_output_file.close()

    label_dir = output_dir / 'labels'
    label_dir.mkdir(parents=True, exist_ok=True)
    label_output_file = open(label_dir / f'{id}_labeled_sleep.txt', mode='w')
    for i in range(label.shape[0]):
        label_output_file.write(f'{(i)*30+timestamp_start} {int(label[i])}\n')
    label_output_file.close()
    return


def make_split(dir_list, data_dir, output_dir, interval):
    id_file = open(output_dir / 'id.txt', mode='w')

    for dir in tqdm(dir_list):
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

        ecg_wav, ecg_sr, ecg_timestamp = load_ecg_chunks(ecg_file, ecg_timestamp_file)
        imu_data, imu_sr, imu_timestamp = load_imu(imu_file, imu_timestamp_file)
        timestamp_start = None

        id = dir.name.split('_')[1]
        label = torch.tensor([])
        file_list = sorted(np.array([x for x in dir.iterdir()]))
        for file in file_list:
            if (file.match('*.wav')):
                num = file.name.split('cleaned_')[1].split("_")[0]
                for f in orig_audio_folder.iterdir():
                    if (('_Audio_timestamps_' + num + ".txt") in f.name):
                        audio_timestamp_file = f
                for f in dir.iterdir():
                    if (f'cleaned_{num}' in f.name and "TextGrid" in f.name):
                        audio_textgrid_file = f
                audio_wav, audio_sr, audio_timestamp = load_audio_chunks([file], [audio_timestamp_file], resample_rate=16000)
                timestamp_start = audio_timestamp[0, 0] if timestamp_start is None else timestamp_start
                num_data = math.floor(audio_wav.shape[0] / audio_sr / interval)
                tmp_label = load_sleep_label(num_data, audio_textgrid_file, interval)
                label = torch.cat((label, tmp_label), dim=0)

        write_data(ecg_wav, ecg_sr, ecg_timestamp, imu_data,imu_sr, imu_timestamp, label,interval, id, output_dir, timestamp_start)

        print(f'{id}')
        id_file.write(f'{id}, ')
    id_file.close()


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
    dir_list = np.array([x for x in annotation_dir.iterdir() if x.is_dir() and not (x.name == "No Sleep files") and not ('Processed ECG' in x.name)])

    # Segmenting normal datasets
    output_dir.mkdir(parents=True, exist_ok=True)
    make_split(dir_list, data_dir, output_dir, args.interval)

if __name__ == '__main__':
    main()