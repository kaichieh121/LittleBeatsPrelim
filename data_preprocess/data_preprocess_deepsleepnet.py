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
    parser.add_argument('--output_dir', default="D:\\datasets\\littlebeats\\segmented_data_deepsleepnet")
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--dir_count', type=int, default=10000)
    args = parser.parse_args()
    return args

def write_data(ecg_wav, ecg_sr, ecg_timestamp, label, interval, id, output_dir, timestamp_start):

    ecg_output_path = output_dir / f'{id}_ecg.npz'
    ecg_start_idx = ecg_sr*(abs(timestamp_start-ecg_timestamp[0,0]))

    ecg_wav = ecg_wav[ecg_start_idx : ecg_start_idx + ecg_sr*label.shape[0]*interval]
    x = ecg_wav.view(label.shape[0], ecg_sr * interval, 1)
    y = label
    channel = 'EEG Fpz-Cz'

    save_dict = {
        "x": x,
        "y": y,
        "fs": ecg_sr,
        "ch_label": channel,
        # "header_raw": h_raw,
        # "header_annotation": h_ann,
    }
    np.savez(ecg_output_path, **save_dict)
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
                audio_wav, audio_sr, audio_timestamp = load_audio_chunks([file], [audio_timestamp_file],
                                                                         resample_rate=16000)
                timestamp_start = audio_timestamp[0, 0] if timestamp_start is None else timestamp_start
                num_data = math.floor(audio_wav.shape[0] / audio_sr / interval)
                tmp_label = load_sleep_label(num_data, audio_textgrid_file, interval)
                label = torch.cat((label, tmp_label), dim=0)

        write_data(ecg_wav, ecg_sr, ecg_timestamp, label,interval, id, output_dir, timestamp_start)

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