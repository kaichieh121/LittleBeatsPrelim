import argparse
import math
import time
import torch, torchaudio
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from synchronize import create_chunks, load_audio_chunks, load_ecg_chunks, load_imu, load_sleep_label, align

def get_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--data_dir', default="D:\\datasets\\littlebeats\\sleep_study_preliminary_recordings")
    parser.add_argument('--output_dir', default="D:\\datasets\\littlebeats\\segmented_data_clean\\pretrain_limu")
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
        for file in orig_imu_folder.iterdir():
            if (file.match('*cleaned.txt')):
                imu_file = file
            if ('timestamp' in file.name):
                imu_timestamp_file = file
        pd_file = pd.read_csv(imu_file.__str__(), sep=',', header=None)
        imu_data = torch.tensor(pd_file.values)
        imu_sr = 150
        interval = 30
        total_time += imu_data.shape[0] / 67 / 3600
        # imu_data = imu_data[:int(imu_data.shape[0]/imu_sr/interval)*imu_sr*interval, 1:]
        # imu_data = imu_data.view(int(imu_data.shape[0]/imu_sr/interval), -1, 9)
        # if imu_data.sum() == 0:
        #     print(f'{dir_name} has corrupted IMU')
        #     continue
        # for i in range(imu_data.shape[0]):
        #     idx_name = f'{str(idx).zfill(6)}'
        #     path = (output_dir / f'imu_{idx_name}.txt').__str__()
        #     np.savetxt(path, imu_data[i])
        #     idx = idx + 1
    print(total_time)


if __name__ == '__main__':
    main()