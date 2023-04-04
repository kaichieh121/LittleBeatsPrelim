import math
import time
import textgrid
import argparse
import torch, torchaudio
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
import heartpy as hp
from pathlib import Path
from pydub import AudioSegment, utils
from datetime import datetime, timezone
from synchronize import load_audio_chunks, load_ecg_chunks, load_imu, align

def analyze_data(audio, audio_sr, ecg, ecg_sr, accz_data, imu_sr, smoothing=False):
    bpm_visual = torch.zeros(ecg.shape[0])
    audio_visual = torch.zeros(audio.shape[0])
    # audio_visual = audio.square().mean(dim=1).sqrt()


    acc_z_mean = accz_data.mean(dim=1)
    acc_z_visual = (accz_data.abs() > 9).sum(dim=1).to(torch.float)

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

def visualize_data(fig, audio_visual, bpm_visual, acc_z, ibi_data, condition, subplot_id, x_lb, x_ibi, loc_base):
    loc = plticker.MultipleLocator(base=loc_base)
    ax1 = fig.add_subplot(4, 4, 0 * 4 + subplot_id)
    ax2 = fig.add_subplot(4, 4, 1 * 4 + subplot_id)
    ax3 = fig.add_subplot(4, 4, 2 * 4 + subplot_id)
    ax4 = fig.add_subplot(4, 4, 3 * 4 + subplot_id)

    ax1.title.set_text(f'Audio Energy ({condition})')
    ax1.plot(x_lb, audio_visual)
    ax1.xaxis.set_major_locator(loc)

    ax2.title.set_text(f'BPM ({condition})')
    ax2.plot(x_lb, bpm_visual)
    ax2.xaxis.set_major_locator(loc)

    ax3.title.set_text(f'Acceleration Z-axis ({condition})')
    ax3.plot(x_lb, acc_z)
    ax3.xaxis.set_major_locator(loc)

    ax4.title.set_text(f'ibi ({condition})')
    ax4.plot(x_ibi, ibi_data)
    ax4.xaxis.set_major_locator(loc)


if __name__ == '__main__':


    warnings.filterwarnings("ignore")
    data_folder = Path('./align_demo_data')

    for dir in data_folder.iterdir():
        if dir.is_dir():
            dir_name = dir.name

            audio_folder = data_folder / dir_name / "Audio_cleaned"
            ecg_folder = data_folder / dir_name / "ECG_cleaned"
            imu_folder = data_folder / dir_name / "IMU_cleaned"

            audio_paths = []
            audio_timestamps = []
            for file in audio_folder.iterdir():
                if (file.match('*.wav') and 'zero' not in file.name):
                    audio_paths.append(file)
                if ('timestamps' in file.name):
                    audio_timestamps.append(file)
            audio_paths.sort()
            audio_timestamps.sort()

            for file in ecg_folder.iterdir():
                if (file.match('*cleaned.wav')):
                    ecg_file = file
                if ('timestamp' in file.name):
                    ecg_timestamp_file = file

            for file in imu_folder.iterdir():
                if (file.match('*cleaned.txt')):
                    imu_file = file
                if ('timestamp' in file.name):
                    imu_timestamp_file = file

            audio_wav, audio_sr, audio_timestamp = load_audio_chunks(audio_paths, audio_timestamps)
            ecg_wav, ecg_sr, ecg_timestamp = load_ecg_chunks(ecg_file, ecg_timestamp_file)
            imu_data, imu_sr, imu_timestamp = load_imu(imu_file, imu_timestamp_file)

            audio_wav, ecg_wav, imu_data = align(audio_wav, audio_sr, audio_timestamp, ecg_wav, ecg_sr, ecg_timestamp, imu_data, imu_sr, imu_timestamp)

            for file in dir.iterdir():
                if ('CPeak_LB.csv' in file.name):
                    df = pd.read_csv(file, delimiter=',')
            fig = plt.figure(figsize=(15, 15))
            fig.suptitle(f'{dir_name}')
            for i, condition in enumerate(['Baseline', 'Still1', 'Still2', 'Still3']):
                row = df.loc[df['Condition'] == condition]
                start_time = int(row.iloc[:,3].item()*60)
                end_time = int(row.iloc[:,4].item()*60)

                audio_oi = audio_wav[int(start_time*audio_sr):int(end_time*audio_sr)]
                ecg_oi = ecg_wav[int(start_time*ecg_sr):int(end_time*ecg_sr)]
                accz_oi = imu_data['acc_z'][int(start_time*imu_sr):int(end_time*imu_sr)]

                interval=10
                num_data = math.floor(audio_oi.shape[0] / audio_sr / interval)
                audio_data = audio_oi[:num_data * interval * audio_sr].view(num_data, interval * audio_sr).to(torch.float)
                ecg_data = ecg_oi[:num_data * interval * ecg_sr].view(num_data, interval * ecg_sr)
                accz_data = accz_oi[:num_data * interval * imu_sr].view(num_data, interval * imu_sr)

                audio_energy, bpm, acc_z_percentage, acc_z_mean = analyze_data(audio_data, audio_sr, ecg_data, ecg_sr, accz_data, imu_sr, smoothing=False)
                for file in dir.iterdir():
                    if (condition in file.name):
                        ibi_df = pd.read_csv(file, sep=' ', header=None)
                        ibi_data = torch.tensor(ibi_df.values).squeeze()

                x_axis_lb = torch.linspace(start_time, end_time, steps=num_data)
                x_axis_ibi = torch.linspace(start_time, end_time, steps=ibi_data.shape[0])
                if(condition=='Still1'):
                    loc_base = 25
                elif(condition=='Still2'):
                    loc_base = 25
                else:
                    loc_base = 50
                visualize_data(fig, audio_energy, bpm, acc_z_mean, ibi_data, condition, i+1, x_axis_lb, x_axis_ibi, loc_base)
            fig.tight_layout(pad=5.0)
            plt.show()