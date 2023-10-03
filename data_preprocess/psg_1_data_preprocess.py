'''
    for segmenting hour long audio, ecg, imu to audio chunks

'''

import argparse
import copy
import math
import torch
from pathlib import Path
import pandas as pd
import numpy as np
from synchronize import create_chunks, load_audio_chunks, load_ecg_chunks, load_imu, align
from tqdm import tqdm

def read_avg_hr(dir):
    file = open(dir / 'avg_hr.txt', 'r')
    return float(file.read())

def get_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--data_dir', default="D:\\datasets\\littlebeats\\PSG_LB_Adult_Sleep_Study\\Data Files")
    parser.add_argument('--output_dir', default="D:\\datasets\\littlebeats\\psg_data_clean")
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--dir_count', type=int, default=10000)
    parser.add_argument('--gmt_offset', type=int, default=-5)
    args = parser.parse_args()
    return args

def load_sleep_label(num_data, labels_raw, label_start_index=0):
    if num_data is None:
        num_data = labels_raw.shape[0]
    label = torch.zeros(num_data)
    for i in range(num_data):
        if(label_start_index+i>=908):
            print()
        if("Wake" in labels_raw[label_start_index+i][1]):
            label[i] = 0
        elif("Stage 1" in labels_raw[label_start_index+i][1]):
            label[i] = 1
        elif ("Stage 2" in labels_raw[label_start_index + i][1]):
            label[i] = 1
        elif("REM" in labels_raw[label_start_index+i][1]):
            label[i] = 2
    return label

def combine_audio_timestamp(timestamp_1, timestamp_2):
    end_index = timestamp_1[-1, -1]
    for row in range(timestamp_2.shape[0]):
        timestamp_2[row][1] += end_index
        timestamp_2[row][3] += end_index
    return torch.cat((timestamp_1, timestamp_2), dim=0)

def align_lb_psg(audio_wav, audio_sr, ecg_wav, ecg_sr, imu_data, imu_sr, label, labels_start_time, start_time, interval=30):
    # start_time < labels_start_time because that's how we collect LB and PSG data
    # LB data will also end after PSG ends
    # so we will use start_time as labels_start_time as start, and end of LB data as end
    t_diff = labels_start_time - start_time
    audio_wav = audio_wav[int(t_diff * audio_sr):]
    ecg_wav = ecg_wav[int(t_diff * ecg_sr):]
    for i, (key, val) in enumerate(imu_data.items()):
        imu_data[key] = val[int(t_diff * imu_sr):]

    num_data = min(math.floor(audio_wav.shape[0] / audio_sr / interval), label.shape[0])
    audio_data = audio_wav[:num_data * interval * audio_sr].view(num_data, interval * audio_sr)
    ecg_data = ecg_wav[:num_data * interval * ecg_sr].view(num_data, interval * ecg_sr)
    for i, (key, val) in enumerate(imu_data.items()):
        imu_data[key] = val[:num_data * interval * imu_sr].view(num_data, interval * imu_sr)
    label = label[:num_data]

    return audio_data, ecg_data, imu_data, label
def load_data(imu_file, imu_timestamp_file, ecg_file, ecg_timestamp_file, audio_file, audio_timestamp_file, labels_raw, session_start_time, interval=30):

    audio_wav, audio_sr, audio_timestamp = load_audio_chunks([audio_file], [audio_timestamp_file], resample_rate=16000)
    ecg_wav, ecg_sr, ecg_timestamp = load_ecg_chunks(ecg_file, ecg_timestamp_file)
    imu_data, imu_sr, imu_timestamp = load_imu(imu_file, imu_timestamp_file)

    audio_wav, ecg_wav, imu_data = align(audio_wav, audio_sr, audio_timestamp, ecg_wav, ecg_sr, ecg_timestamp, imu_data, imu_sr, imu_timestamp)

    num_data = math.floor(audio_wav.shape[0] / audio_sr / interval)
    audio_data = audio_wav[:num_data * interval * audio_sr].view(num_data, interval * audio_sr)
    ecg_data = ecg_wav[:num_data * interval * ecg_sr].view(num_data, interval * ecg_sr)
    for i, (key, val) in enumerate(imu_data.items()):
        imu_data[key] = val[:num_data * interval * imu_sr].view(num_data, interval * imu_sr)

    file_start_time = audio_timestamp[0][0]
    label_start_index = int(np.round((file_start_time - session_start_time)/30))
    label = load_sleep_label(num_data, labels_raw, label_start_index)
    return audio_data.to(torch.float), audio_sr, ecg_data, ecg_sr, imu_data, imu_sr, label

def make_split(dir_list, data_dir, output_dir, interval, gmt_offset):


    for dir in dir_list:
        pd_file = pd.read_csv((dir/'labels.csv').__str__(), sep=',', header=None)
        labels_raw = pd_file.values
        pd_file = pd.read_csv((dir / 'psg_timestamp.csv').__str__(), sep=',', header=None)
        labels_start_time = pd_file.values[0][0] + gmt_offset * 3600

        orig_audio_folder = dir / "Audio_cleaned"
        orig_ecg_folder = dir / "ECG_cleaned"
        orig_imu_folder = dir / "IMU_cleaned"

        # find start time
        for file in orig_audio_folder.iterdir():
            if (file.match('*_Audio_timestamps_1.txt')):
                audio_timestamp_file = file
        pd_file = pd.read_csv(audio_timestamp_file.__str__(), sep=' ', header=None)
        start_time = torch.tensor(pd_file.values)[0][0]

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
        full_audio = None
        full_audio_timestamp = None
        for file in orig_audio_folder.iterdir():
            if (file.match('*with_zero.wav')):
                num = file.name.split('cleaned_')[1].split("_")[0]
                for f in orig_audio_folder.iterdir():
                    if (('_Audio_timestamps_' + num + ".txt") in f.name):
                        audio_timestamp_file = f
                audio_wav, audio_sr, audio_timestamp = load_audio_chunks([file], [audio_timestamp_file],
                                                                         resample_rate=16000)
                if full_audio_timestamp is None:
                    full_audio_timestamp = audio_timestamp
                else:
                    full_audio_timestamp = combine_audio_timestamp(full_audio_timestamp, audio_timestamp)
                if full_audio is None:
                    full_audio = audio_wav
                else:
                    full_audio = torch.cat((full_audio, audio_wav), dim=0)
                print(f'{file.name}')

        ecg_wav, ecg_sr, ecg_timestamp = load_ecg_chunks(ecg_file, ecg_timestamp_file)
        imu_data, imu_sr, imu_timestamp = load_imu(imu_file, imu_timestamp_file)
        audio_wav, ecg_wav, imu_data, start_time = align(full_audio, audio_sr, full_audio_timestamp, ecg_wav, ecg_sr,
                                                         ecg_timestamp, imu_data, imu_sr, imu_timestamp)

        label = load_sleep_label(None, labels_raw, label_start_index=0)

        audio_data, ecg_data, imu_data, label = align_lb_psg(audio_wav, audio_sr, ecg_wav, ecg_sr, imu_data, imu_sr,
                                                             label, labels_start_time, start_time)

        rand_gen_int = np.arange(label.shape[0])
        np.random.shuffle(rand_gen_int)
        val_ratio = 0.15
        test_ratio = 0.15

        train_split_idx = int(label.shape[0] * (1-val_ratio-test_ratio))
        val_split_idx = train_split_idx + int(label.shape[0] * val_ratio)

        audio_data = audio_data[rand_gen_int,:]
        ecg_data = ecg_data[rand_gen_int, :]
        for i, (key, val) in enumerate(imu_data.items()):
            imu_data[key] = imu_data[key][rand_gen_int, :]
        label = label[rand_gen_int]

        sub_output_dir = output_dir / 'train'
        sub_output_dir.mkdir(parents=True, exist_ok=True)
        audio_x = audio_data[:train_split_idx, :]
        ecg_x = ecg_data[:train_split_idx, :]
        imu_x = copy.deepcopy(imu_data)
        for i, (key, val) in enumerate(imu_data.items()):
            imu_x[key] = imu_data[key][:train_split_idx, :]
        label_x = label[:train_split_idx]
        label_file = open(sub_output_dir / 'label.csv', mode='w')
        idx = create_chunks(audio_x, audio_sr, ecg_x, ecg_sr, imu_x, imu_sr, label_x, sub_output_dir, label_file, idx=0, avg_hr=None, avg_hr_file=None)
        label_file.close()

        sub_output_dir = output_dir / 'val'
        sub_output_dir.mkdir(parents=True, exist_ok=True)
        audio_x = audio_data[train_split_idx:val_split_idx, :]
        ecg_x = ecg_data[train_split_idx:val_split_idx, :]
        imu_x = copy.deepcopy(imu_data)
        for i, (key, val) in enumerate(imu_data.items()):
            imu_x[key] = imu_data[key][train_split_idx:val_split_idx, :]
        label_x = label[train_split_idx:val_split_idx]
        label_file = open(sub_output_dir / 'label.csv', mode='w')
        idx = create_chunks(audio_x, audio_sr, ecg_x, ecg_sr, imu_x, imu_sr, label_x, sub_output_dir, label_file, idx=0, avg_hr=None, avg_hr_file=None)
        label_file.close()

        sub_output_dir = output_dir / 'test'
        sub_output_dir.mkdir(parents=True, exist_ok=True)
        audio_x = audio_data[val_split_idx:, :]
        ecg_x = ecg_data[val_split_idx:, :]
        imu_x = copy.deepcopy(imu_data)
        for i, (key, val) in enumerate(imu_data.items()):
            imu_x[key] = imu_data[key][val_split_idx:, :]
        label_x = label[val_split_idx:]
        label_file = open(sub_output_dir / 'label.csv', mode='w')
        idx = create_chunks(audio_x, audio_sr, ecg_x, ecg_sr, imu_x, imu_sr, label_x, sub_output_dir, label_file, idx=0, avg_hr=None, avg_hr_file=None)
        label_file.close()


    label_file.close()


def main():

    args = get_arguments()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dir_list = np.array([x for x in data_dir.iterdir() if x.is_dir()])
    make_split(dir_list, data_dir, output_dir, args.interval, gmt_offset=args.gmt_offset)

if __name__ == '__main__':
    main()