import math
import textgrid
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
from synchronize import load_ecg_chunks, load_imu, align
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import BinaryF1Score, BinaryCohenKappa

from predictor import AudioEnergyPredictor, EcgPredictor, AccZPredictor

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

def analyze_data(audio, audio_sr, ecg, ecg_sr, avg_hr, imu_data, imu_sr, smoothing=False):
    audio_visual = torch.zeros(ecg.shape[0])
    bpm_visual = torch.zeros(ecg_x.shape[0])
    acc_z_visual = torch.zeros(ecg_x.shape[0])
    acc_z_var = torch.zeros(ecg_x.shape[0])
    acc_z_mean = torch.zeros(ecg_x.shape[0])

    for j in range(audio.shape[0]):

        audio_visual[j] = AudioSegment(audio[j].numpy().tobytes(), frame_rate=audio_sr, sample_width=audio[j].numpy().dtype.itemsize, channels=1).rms

        try:
            working_data, measures = hp.process(ecg[j].numpy(), ecg_sr)
            if (math.isnan(measures['bpm'])):
                bpm_visual[j] = 0
            else:
                bpm_visual[j] = min(measures['bpm'], 250)
        except:
            bpm_visual[j] = 0

        acc_z_mean[j] = imu_data['acc_z'][j].mean()
        acc_z_visual[j] = (abs(imu_data['acc_z'][j]) > 9).sum()
        z_max = 10
        z_min = -10
        norm_z = (imu_data['acc_z'][j]-z_min)/(z_max-z_min)
        acc_z_var[j] = torch.var(norm_z)

    if smoothing:
        n = 5
        kernel = torch.ones(1,1,n)
        audio_visual = (F.conv1d(audio_visual.unsqueeze(0).unsqueeze(0), kernel, padding='same') / n).squeeze()
        bpm_visual = (F.conv1d(bpm_visual.unsqueeze(0).unsqueeze(0), kernel, padding='same') / n).squeeze()
        acc_z_visual = (F.conv1d(acc_z_visual.unsqueeze(0).unsqueeze(0), kernel, padding='same') / n).squeeze()
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


def align_data(imu_file, imu_timestamp_file, ecg_file, ecg_timestamp_file, audio_file, audio_timestamp_file, audio_textgrid_file, interval=30):
    audio_wav, audio_sr = torchaudio.load(audio_file.__str__(), normalization=False)
    audio_wav = audio_wav.squeeze()
    pd_file = pd.read_csv(audio_timestamp_file.__str__(), sep=' ', header=None)
    audio_timestamp = torch.tensor(pd_file.values)[:-1]

    ecg_wav, ecg_sr, ecg_timestamp = load_ecg_chunks(ecg_file, ecg_timestamp_file)

    imu_data, imu_sr, imu_timestamp = load_imu(imu_file, imu_timestamp_file)

    audio_wav, ecg_wav, imu_data = align(audio_wav, audio_sr, audio_timestamp, ecg_wav, ecg_sr, ecg_timestamp, imu_data, imu_sr, imu_timestamp)

    tg = textgrid.TextGrid.fromFile(audio_textgrid_file.__str__())
    num_data = math.floor(audio_wav.shape[0] / audio_sr / interval)
    audio_data = audio_wav[:num_data * interval * audio_sr].view(num_data, interval * audio_sr)
    ecg_data = ecg_wav[:num_data * interval * ecg_sr].view(num_data, interval * ecg_sr)
    for i, (key, val) in enumerate(imu_data.items()):
        imu_data[key] = val[:num_data * interval * imu_sr].view(num_data, interval * imu_sr)
    label = torch.zeros(num_data)
    for i in tg.getFirst('SLEEP'):
        if ('SLEEP' in i.mark):
            start = round(i.minTime / interval)
            end = min(int(i.maxTime / interval), num_data - 1)
            for j in range(start, end + 1):
                label[j] = 1
    return audio_data, audio_sr, ecg_data, ecg_sr, imu_data, imu_sr, label

def predict(audio_energy, bpm, laying_down, acc_z_var, avg_hr, mode, threshold):

    if mode == 'audio':
        y = is_silent(audio_energy, max_amp=32767, threshold=threshold)
        return y

    if mode == 'ecg':
        y = hr_low(bpm, avg_hr+threshold)
        return y

    if mode == 'imu':
        y = on_back_or_stomach(laying_down, threshold)
        return y

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
    pred = (F.conv1d(pred.unsqueeze(0).unsqueeze(0), kernel, padding='same') / n).squeeze().round()
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
    modes = {}
    audio_thresholds = []
    ecg_thresholds = []
    audio_ecg_thresholds = []
    imu_thresholds = []
    # (audio_threshold, ecg_threshold, acc_z, acc_z_var)
    all_thresholds = [(-30, -15, 0.60, 0.03), (-30, -15, 0.60, 0.04), (-30, -15, 0.60, 0.05)]




    modes['audio'] = audio_thresholds
    modes['ecg'] = ecg_thresholds
    modes['audio+ecg'] = audio_ecg_thresholds
    modes['imu'] = imu_thresholds
    modes['all'] = all_thresholds
    warnings.filterwarnings("ignore")
    annotation_folder = Path('/home/kcchang3/data/LittleBeats/LittleBeats_Sleep annotations/RA sleep annotations completed (for Jeff)')
    data_folder = Path('/home/kcchang3/data/LittleBeats/sleep_study_preliminary_recordings')

    output_file = open(f'./output/{datetime.now().strftime("%d-%m-%Y-%H-%M")}.txt', 'w')
    all_pred_y = {}

    for idx, (mode, thresholds) in enumerate(modes.items()):
        all_pred_y[mode] = {}
        for threshold in thresholds:
            all_pred_y[mode][threshold] = {'prediction':torch.tensor([]), 'y':torch.tensor([])}

    for dir in annotation_folder.iterdir():
        if dir.is_dir() and not (dir.name == "No Sleep files") and not ('Processed ECG' in dir.name):
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
                    audio_energy, bpm, laying_down, acc_z_var, acc_z = analyze_data(audio_x, audio_sr, ecg_x, ecg_sr, avg_hr, imu_data, imu_sr, smoothing=True)
                    # visualize_data(audio_energy, bpm, laying_down, acc_z_var, acc_z, y)
                    for idx, (mode, thresholds) in enumerate(modes.items()):
                        for threshold in thresholds:
                            pred = predict(audio_energy, bpm, laying_down, acc_z_var, avg_hr, mode, threshold)
                            pred = pred_smooth(pred)
                            all_pred_y[mode][threshold]['prediction'] = torch.cat((all_pred_y[mode][threshold]['prediction'], pred), dim=0)
                            all_pred_y[mode][threshold]['y'] = torch.cat((all_pred_y[mode][threshold]['y'], y), dim=0)

                    accuracy = 1 - (((pred - y) ** 2).sum()) / (pred.shape[0])
                    if(accuracy < 0.4):
                        print()
                    print(f'{file.name}: {accuracy}')

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
                output_file.write(f'mode={mode}, audio_threshold={threshold[0]}, ecg_threshold={threshold[1]}, imu_threshold={threshold[2]}\n')
            output_file.write(f'conf_matrix={conf_matrix}\n')
            output_file.write(f'accuracy={accuracy}\n')
            output_file.write(f'f1={f1}\n')
            output_file.write(f'kappa={kappa}\n\n')
    output_file.close()
