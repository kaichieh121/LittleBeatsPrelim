import math
import time
import textgrid
import argparse
import torch, torchaudio
from torch.autograd import Variable
from torch.utils.data import DataLoader
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
from synchronize import load_audio_chunks, load_ecg_chunks, load_imu, align, load_sleep_label
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import BinaryF1Score, BinaryCohenKappa
from dataloader import LittleBeatsDataset

def get_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--data_dir', default='/home/kcchang3/data/LittleBeats/data_30s')
    args = parser.parse_args()
    return args

def is_silent(audio_energy, max_amp, threshold=-35):
    silence_thresh = utils.db_to_float(threshold) * max_amp
    num_data = audio_energy.shape[0]
    res = torch.zeros(num_data)
    for i in range(num_data):
        if audio_energy[i] <= silence_thresh:
            res[i] = 1
    return res

def hr_low(bpm, threshold):
    return (bpm<threshold).to(torch.int16)

def on_back_or_stomach(acc_z, acc_var, acc_z_threshold=0.6, acc_var_threshold=0.04, imu_sr=150, interval=30):
    num_data = acc_z.shape[0]
    res = torch.zeros(num_data)
    for i in range(num_data):
        if (acc_z[i] >= acc_z_threshold*imu_sr*interval) and acc_var[i]<acc_var_threshold:
            res[i] = 1
    return res

def analyze_data(audio, audio_sr, ecg, ecg_sr, accz, imu_sr, smoothing=False):
    audio = audio.squeeze()
    ecg = ecg.squeeze()
    bpm_visual = torch.zeros(ecg.shape[0])
    audio_visual = torch.zeros(audio.shape[0])
    # audio_visual = audio.square().mean(dim=1).sqrt()


    acc_z_mean = accz.mean(dim=2)
    acc_z_visual = (accz.abs() > 9).sum(dim=2).to(torch.float).squeeze()
    z_max = 10
    z_min = -10
    norm_z = (accz - z_min) / (z_max - z_min)
    acc_z_var = torch.var(norm_z, dim=2)

    for j in range(audio.shape[0]):

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
        # acc_z_var = (F.conv1d(acc_z_var.unsqueeze(0).unsqueeze(0), kernel, padding='same') / n).squeeze()

    return audio_visual, bpm_visual, acc_z_visual, acc_z_var, acc_z_mean

def visualize_data(audio_visual, bpm_visual, acc_z_visual, acc_z_var, acc_z, label):
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
    plt.plot(acc_z_var)
    plt.title('Acceleration Z-axis variance')
    plt.figure()
    plt.plot(acc_z)
    plt.title('Acceleration Z-axis')
    plt.figure()
    plt.plot(label)
    plt.title('Label 0 wake 1 sleep')
    plt.show()


def predict(audio_energy, bpm, laying_down, acc_z_var, avg_hr, mode, threshold):

    if mode == 'audio':
        return is_silent(audio_energy, max_amp=32767, threshold=threshold)

    if mode == 'ecg':
        return hr_low(bpm, avg_hr+threshold)

    if mode == 'imu':
        return on_back_or_stomach(laying_down, threshold)

    if mode == 'audio+ecg':
        pred_audio = is_silent(audio_energy, max_amp=32767, threshold=threshold[0])
        pred_ecg = hr_low(bpm, avg_hr+threshold[1])
        y = pred_audio + pred_ecg
        return (y >= 1).to(torch.float)

    if mode == 'all':
        pred_audio = is_silent(audio_energy, max_amp=32767, threshold=threshold[0])
        pred_ecg = hr_low(bpm, avg_hr + threshold[1])
        pred_acc = on_back_or_stomach(laying_down, acc_z_var, threshold[2], threshold[3])
        y = pred_audio + pred_ecg + pred_acc
        return (y >= 1).to(torch.float)

def pred_smooth(pred, n=3):
    # k1 = torch.tensor([0.,1.,0.])
    # k2 = torch.tensor([1.,0.,1.])
    # for i in range(pred.shape[0]-2):
    #     if((pred[i:i+3]==k1).all()):
    #         pred[i+1] = 0
    #         continue
    #     if((pred[i:i+3]==k2).all()):
    #         pred[i+1] = 1
    #         continue
    # return pred

    kernel = torch.ones(1, 1, n)
    pred = (F.conv1d(pred.unsqueeze(0).unsqueeze(0), kernel, padding=n//2) / n).squeeze().round()
    return pred

def evaluate_classifier(pred, y):
    y = y.to(torch.int32)
    pred = pred.to(torch.int32)
    conf_matrix = ConfusionMatrix(num_classes=2)(pred, y)
    accuracy = 1 - (((pred - y) ** 2).sum()) / (pred.shape[0])
    f1 = BinaryF1Score()(pred, y)
    kappa = BinaryCohenKappa()(pred, y)
    return conf_matrix, accuracy, f1, kappa

if __name__ == '__main__':

    args = get_arguments()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    start_time = time.time()
    modes = {}
    audio_thresholds = []
    ecg_thresholds = []
    audio_ecg_thresholds = []
    imu_thresholds = []
    # (audio_threshold, ecg_threshold, acc_z, acc_z_var)
    all_thresholds = [(-30, -15, 0.60, 0.005), (-30, -15, 0.30, 0.05), (-30, -15, 0.60, 0.1), (-30, -15, 0.60, 20)]
    modes['audio'] = audio_thresholds
    modes['ecg'] = ecg_thresholds
    modes['audio+ecg'] = audio_ecg_thresholds
    modes['imu'] = imu_thresholds
    modes['all'] = all_thresholds

    warnings.filterwarnings("ignore")
    data_dir = args.data_dir
    audio_sr = 16000
    ecg_sr = 2381
    imu_sr = 150
    batch_size = 64

    # annotation_folder = Path('//ad.uillinois.edu/aces/hdfs/share/McElwain-MCRP-RA/LittleBeats_RA/LittleBeats_Sleep annotations/RA sleep annotations completed (for Jeff)')
    # data_folder = Path('//ad.uillinois.edu/aces/hdfs/share/McElwain-MCRP-RA/LittleBeats_RA/sleep_study_preliminary_recordings')

    output_file = open(f'./output/{datetime.now().strftime("%d-%m-%Y-%H-%M")}.txt', 'w')
    all_pred_y = {}

    for idx, (mode, thresholds) in enumerate(modes.items()):
        all_pred_y[mode] = {}
        for threshold in thresholds:
            all_pred_y[mode][threshold] = {'prediction':torch.tensor([]), 'y':torch.tensor([])}

    training_data = LittleBeatsDataset(data_dir)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)

    for batch, (audio_x, ecg_x, accz_x, avg_hr, y) in enumerate(train_dataloader):
        audio_energy, bpm, acc_z_percentage, acc_z_var, acc_z_mean = analyze_data(audio_x, audio_sr, ecg_x, ecg_sr, accz_x, imu_sr, smoothing=False)
        for idx, (mode, thresholds) in enumerate(modes.items()):
            for threshold in thresholds:
                pred = predict(audio_energy, bpm, acc_z_percentage, acc_z_var, avg_hr, mode, threshold)
                pred = pred_smooth(pred)
                all_pred_y[mode][threshold]['prediction'] = torch.cat((all_pred_y[mode][threshold]['prediction'], pred), dim=0)
                all_pred_y[mode][threshold]['y'] = torch.cat((all_pred_y[mode][threshold]['y'], y), dim=0)
        if(batch%10==0):
            print(f'Batch: {batch}, data: {batch*batch_size}')


    for idx, (mode, thresholds) in enumerate(modes.items()):
        for threshold in thresholds:
            conf_matrix, accuracy, f1, kappa = evaluate_classifier(all_pred_y[mode][threshold]['prediction'], all_pred_y[mode][threshold]['y'])
            if mode == 'audio':
                output_file.write(f'mode={mode}, audio_threshold={threshold}\n')
            if mode == 'ecg':
                output_file.write(f'mode={mode}, ecg_threshold={threshold}\n')
            if mode == 'imu':
                output_file.write(f'mode={mode}, imu_threshold={threshold}\n')
            if mode == 'audio+ecg':
                output_file.write(f'mode={mode}, audio_threshold={threshold[0]}, ecg_threshold={threshold[1]}\n')
            if mode == 'all':
                output_file.write(f'mode={mode}, audio_threshold={threshold[0]}, ecg_threshold={threshold[1]}, imu_threshold={threshold[2]} {threshold[3]}\n')
            output_file.write(f'conf_matrix={conf_matrix}\n')
            output_file.write(f'accuracy={accuracy}\n')
            output_file.write(f'f1={f1}\n')
            output_file.write(f'kappa={kappa}\n\n')
    print(f'runtime: {time.time()-start_time}')
    output_file.write(f'runtime: {time.time()-start_time}')
    output_file.close()
