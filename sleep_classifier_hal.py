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
import scipy as sp
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

def get_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--annotation_dir', default='/home/kcchang3/data/LittleBeats/LittleBeats_Sleep annotations/RA sleep annotations completed (for Jeff)')
    parser.add_argument('--data_dir', default='/home/kcchang3/data/LittleBeats/sleep_study_preliminary_recordings')
    parser.add_argument('--dir_count', type=int, default=4)
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

def hr_low(bpm, threshold=120):
    num_data = bpm.shape[0]
    res = torch.zeros(num_data)
    for i in range(num_data):
        if (bpm[i] <= threshold):
            res[i] = 1
    return res

def on_back_or_stomach(acc_z, acc_var, acc_z_threshold=0.6, acc_var_threshold=0.04, imu_sr=150, interval=30):
    num_data = acc_z.shape[0]
    res = torch.zeros(num_data)
    for i in range(num_data):
        if (acc_z[i] >= acc_z_threshold*imu_sr*interval) and acc_var[i]<acc_var_threshold:
            res[i] = 1
    return res

def read_avg_hr(dir):
    file = open(dir / 'avg_hr.txt', 'r')
    return float(file.read())

def analyze_data(audio, audio_sr, ecg, ecg_sr, imu_data, imu_sr, smoothing=False):
    bpm_visual = torch.zeros(ecg.shape[0])
    audio_visual = torch.zeros(audio.shape[0])
    # audio_visual = audio.square().mean(dim=1).sqrt()


    acc_z_mean = imu_data['acc_z'].mean(dim=1)
    acc_z_visual = (imu_data['acc_z'].abs() > 9).sum(dim=1).to(torch.float)
    z_max = 10
    z_min = -10
    norm_z = (imu_data['acc_z'] - z_min) / (z_max - z_min)
    acc_z_var = torch.var(norm_z, dim=1)

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
        # acc_z_var = (F.conv1d(acc_z_var.unsqueeze(0).unsqueeze(0), kernel, padding='same') / n).squeeze()


    return audio_visual, bpm_visual, acc_z_visual, acc_z_var, acc_z_mean

def visualize_data(audio_visual, bpm_visual, acc_z_visual, acc_z_var, acc_z, label):
    def determine_label_change(label):
        switches = []
        cur = label[0]
        for i in range(1, len(label)):
            if(label[i] != cur):
                switches.append(i)
                cur = label[i]
        return switches
    def ttest(switches, data):
        t_stat = []
        p_value = []
        last_switch = 0
        for i, switch in enumerate(switches):
            a = data[last_switch:switch]
            b = data[switch:switches[i+1]] if (i+1)<len(switches) else data[switch:]
            if len(a)<len(b):
                b = b[:len(a)]
            elif len(a)>len(b):
                a = a[-len(b):]
            tmp = sp.stats.ttest_rel(a, b)
            t_stat.append(round(tmp.statistic, 2))
            p_value.append(round(tmp.pvalue, 2))
            last_switch = switch
        return t_stat, p_value

    switches = determine_label_change(label)




    fig = plt.figure(figsize=(8, 16))
    fig.suptitle(f'Sample Visualization')

    ax1 = fig.add_subplot(5, 1, 1)
    ax2 = fig.add_subplot(5, 1, 2)
    ax3 = fig.add_subplot(5, 1, 3)
    ax4 = fig.add_subplot(5, 1, 4)
    ax5 = fig.add_subplot(5, 1, 5)

    ax1.plot(audio_visual)
    tstat, pvalue = ttest(switches, audio_visual)
    ax1.title.set_text(f'Audio Energy tstat={tstat} pvalue={pvalue}')

    ax2.plot(bpm_visual)
    tstat, pvalue = ttest(switches, bpm_visual)
    ax2.title.set_text(f'BPM tstat={tstat} pvalue={pvalue}')

    ax3.plot(acc_z_visual)
    tstat, pvalue = ttest(switches, acc_z_visual)
    ax3.title.set_text(f'Acceleration Z-axis Count tstat={tstat} pvalue={pvalue}')

    ax4.plot(acc_z)
    tstat, pvalue = ttest(switches, acc_z)
    ax4.title.set_text(f'Acceleration Z-axis tstat={tstat} pvalue={pvalue}')

    ax5.plot(label)
    ax5.title.set_text('Label 0 wake 1 sleep')

    fig.tight_layout(pad=5.0)
    plt.show()


def align_data(imu_file, imu_timestamp_file, ecg_file, ecg_timestamp_file, audio_file, audio_timestamp_file, audio_textgrid_file, interval=30):

    audio_wav, audio_sr, audio_timestamp = load_audio_chunks([audio_file], [audio_timestamp_file])
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
    annotation_folder = Path(args.annotation_dir)
    data_folder = Path(args.data_dir)

    # annotation_folder = Path('//ad.uillinois.edu/aces/hdfs/share/McElwain-MCRP-RA/LittleBeats_RA/LittleBeats_Sleep annotations/RA sleep annotations completed (for Jeff)')
    # data_folder = Path('//ad.uillinois.edu/aces/hdfs/share/McElwain-MCRP-RA/LittleBeats_RA/sleep_study_preliminary_recordings')

    output_file = open(f'./output/{datetime.now().strftime("%d-%m-%Y-%H-%M")}.txt', 'w')
    all_pred_y = {}

    for idx, (mode, thresholds) in enumerate(modes.items()):
        all_pred_y[mode] = {}
        for threshold in thresholds:
            all_pred_y[mode][threshold] = {'prediction':torch.tensor([]), 'y':torch.tensor([])}

    dir_count = 0
    for dir in annotation_folder.iterdir():
        if dir.is_dir() and not (dir.name == "No Sleep files") and not ('Processed ECG' in dir.name):
            if(dir_count >= args.dir_count):
                break
            dir_count += 1
            dir_name = dir.name
            orig_audio_folder = data_folder / dir_name / "Audio_cleaned"
            orig_ecg_folder = data_folder / dir_name / "ECG_cleaned"
            orig_imu_folder = data_folder / dir_name / "IMU_cleaned"
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
                        if(('_Audio_timestamps_' + num + ".txt") in f.name):
                            audio_timestamp_file = f
                    for f in dir.iterdir():
                        if(f'cleaned_{num}' in f.name and "TextGrid" in f.name):
                            audio_textgrid_file = f

                    audio_x, audio_sr, ecg_x, ecg_sr, imu_data, imu_sr, y = align_data(imu_file, imu_timestamp_file, ecg_file, ecg_timestamp_file, file, audio_timestamp_file, audio_textgrid_file, interval=30)
                    # audio_x, ecg_x, y = audio_x.to(device), ecg_x.to(device), y.to(device)
                    # for i, (key,val) in enumerate(imu_data.items()):
                    #     imu_data[key] = imu_data[key].to(device)
                    audio_energy, bpm, acc_z_percentage, acc_z_var, acc_z_mean = analyze_data(audio_x, audio_sr, ecg_x, ecg_sr, imu_data, imu_sr, smoothing=True)
                    visualize_data(audio_energy, bpm, acc_z_percentage, acc_z_var, acc_z_mean, y)
                    for idx, (mode, thresholds) in enumerate(modes.items()):
                        for threshold in thresholds:
                            pred = predict(audio_energy, bpm, acc_z_percentage, acc_z_var, avg_hr, mode, threshold)
                            pred = pred_smooth(pred)
                            all_pred_y[mode][threshold]['prediction'] = torch.cat((all_pred_y[mode][threshold]['prediction'], pred), dim=0)
                            all_pred_y[mode][threshold]['y'] = torch.cat((all_pred_y[mode][threshold]['y'], y), dim=0)

                    print(f'{file.name}')

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
