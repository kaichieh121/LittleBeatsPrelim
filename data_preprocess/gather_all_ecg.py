import argparse
import math
import time
import torch, torchaudio
import torchaudio.transforms as T
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--data_dir', default="D:\\datasets\\littlebeats\\sleep_study_preliminary_recordings")
    parser.add_argument('--output_dir', default="D:\\datasets\\littlebeats\\segmented_data_clean\\pretrain_ecg")
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--dir_count', type=int, default=10000)
    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_time = 0
    dir_list = np.array([x for x in data_dir.iterdir() if x.is_dir()])
    idx = 0
    for dir in tqdm(dir_list):
        dir_name = dir.name
        orig_audio_folder = data_dir / dir_name / "Audio_cleaned"
        orig_ecg_folder = data_dir / dir_name / "ECG_cleaned"
        orig_imu_folder = data_dir / dir_name / "IMU_cleaned"
        for file in orig_ecg_folder.iterdir():
            if (file.match('*cleaned.wav')):
                ecg_file = file
            if ('timestamp' in file.name):
                imu_timestamp_file = file
        ecg_data, ecg_sr = torchaudio.load(ecg_file.__str__())
        total_time += ecg_data.shape[1]/ecg_sr/3600
        resample_sr = 16000
        # resampler = T.Resample(ecg_sr, resample_sr, dtype=ecg_data.dtype)
        resampler = T.Resample(ecg_sr, resample_sr) #works on hal
        ecg_data = resampler(ecg_data)[0,:]

        interval = 30
        ecg_data = ecg_data[:int(ecg_data.shape[0]/resample_sr/interval)*resample_sr*interval]
        ecg_data = ecg_data.view(int(ecg_data.shape[0]/resample_sr/interval), -1)
        if ecg_data.sum() == 0:
            print(f'{dir_name} has corrupted IMU')
            continue
        for i in range(ecg_data.shape[0]):
            idx_name = f'{str(idx).zfill(6)}'
            path = (output_dir / f'ecg_{idx_name}.wav').__str__()
            torchaudio.save(path, ecg_data[i].unsqueeze(0), resample_sr)
            idx = idx + 1
    print(total_time)

if __name__ == '__main__':
    main()