import math
import time
import textgrid
import argparse
import torch, torchaudio
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
import heartpy as hp
from pathlib import Path
from helper import silence_detect
from pydub import AudioSegment, utils
from datetime import datetime, timezone
from synchronize import load_audio_chunks, load_ecg_chunks, load_imu, align

def analyze_data(audio, audio_sr, ecg, ecg_sr, imu_data, imu_sr, smoothing=False):
    bpm_visual = torch.zeros(ecg.shape[0])
    audio_visual = torch.zeros(audio.shape[0])
    # audio_visual = audio.square().mean(dim=1).sqrt()


    acc_z_mean = imu_data['acc_z'].mean(dim=1)
    acc_z_visual = (imu_data['acc_z'].abs() > 9).sum(dim=1).to(torch.float)

    for j in range(audio.shape[0]):
        # audio_wv = torch.tensor(audio[j] * 32767, dtype=torch.int16)
        audio_visual[j] = audio[j][audio[j].nonzero()].squeeze().square().mean().sqrt() * 32767

        ecg_zero_removed = ecg[j][ecg[j].nonzero()].squeeze()
        try:
            working_data, measures = hp.process(ecg_zero_removed.numpy(), ecg_sr)
            if (math.isnan(measures['bpm'])):
                bpm_visual[j] = 0
            else:
                bpm_visual[j] = min(measures['bpm'], 250)
        except:
            bpm_visual[j] = 0

    if smoothing:
        n = 5
        kernel = torch.ones(1,1,n)
        audio_visual = (F.conv1d(audio_visual.unsqueeze(0).unsqueeze(0), kernel, padding=n//2) / n).squeeze()
        bpm_visual = (F.conv1d(bpm_visual.unsqueeze(0).unsqueeze(0), kernel, padding=n//2) / n).squeeze()
        acc_z_visual = (F.conv1d(acc_z_visual.unsqueeze(0).unsqueeze(0), kernel, padding=n//2) / n).squeeze()


    return audio_visual, bpm_visual, acc_z_visual, acc_z_mean

def visualize_data(audio_visual, bpm_visual, acc_z_visual, acc_z, ibi_data):
    plt.figure()
    plt.plot(audio_visual)
    plt.title('Audio Energy')
    plt.figure()
    plt.plot(bpm_visual)
    plt.title('BPM')
    plt.figure()
    plt.plot(acc_z_visual)
    plt.title('Acceleration Z-axis Count')
    plt.figure()
    plt.plot(acc_z)
    plt.title('Acceleration Z-axis')
    plt.figure()
    plt.plot(ibi_data)
    plt.title('ibi')
    plt.figure()
    plt.plot(60/ibi_data)
    plt.title('bpm from ibi')
    plt.show()


if __name__ == '__main__':


    warnings.filterwarnings("ignore")
    data_folder = Path('./7511_demo_data')

    audio_folder = data_folder / "Audio_cleaned"
    ecg_folder = data_folder / "ECG_cleaned"
    imu_folder = data_folder / "IMU_cleaned"

    audio_wav, audio_sr, audio_timestamp = load_audio_chunks([audio_folder / 'BP_7551_2022-08-18-15-45-31_24KHz_Audio_cleaned_1.wav'], [audio_folder / 'BP_7551_2022-08-18-15-45-31_15832_Audio_timestamps_1.txt'])
    ecg_wav, ecg_sr, ecg_timestamp = load_ecg_chunks(ecg_folder/'BP_7551_2022-08-18-15-45-31_ECG_cleaned.wav', ecg_folder/'BP_7551_2022-08-18-15-45-31_ECG_timestamp.txt')
    imu_data, imu_sr, imu_timestamp = load_imu(imu_folder/'BP_7551_2022-08-18-15-45-31_IMU_cleaned.txt', imu_folder/'BP_7551_2022-08-18-15-45-31_IMU_timestamp.txt')

    audio_wav, ecg_wav, imu_data = align(audio_wav, audio_sr, audio_timestamp, ecg_wav, ecg_sr, ecg_timestamp, imu_data, imu_sr, imu_timestamp)

    interval=30
    num_data = math.floor(audio_wav.shape[0] / audio_sr / interval)
    audio_data = audio_wav[:num_data * interval * audio_sr].view(num_data, interval * audio_sr).to(torch.float)
    ecg_data = ecg_wav[:num_data * interval * ecg_sr].view(num_data, interval * ecg_sr)
    for i, (key, val) in enumerate(imu_data.items()):
        imu_data[key] = val[:num_data * interval * imu_sr].view(num_data, interval * imu_sr)

    df = pd.read_csv(data_folder/'LB_ID7551_2_Still2_edited_YH_KH.ibi', sep=' ', header=None)
    ibi_data = torch.tensor(df.values).squeeze()

    audio_energy, bpm, acc_z_percentage, acc_z_mean = analyze_data(audio_data, audio_sr, ecg_data, ecg_sr, imu_data, imu_sr, smoothing=True)

    visualize_data(audio_energy, bpm, acc_z_percentage, acc_z_mean, ibi_data)