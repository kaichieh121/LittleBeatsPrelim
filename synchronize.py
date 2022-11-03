
import sys, os
import textgrid
from datetime import datetime, timezone
from pathlib import Path
import torch, torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Audio, display
from zero_insertion_audio import read_timestamps, insert_samples
from helper import plot_waveform, plot_spectrum

def load_audio_chunks(audio_paths, audio_timestamps):
    res = None
    master_timestamp = None
    for i in range(len(audio_paths)):
        waveform, sample_rate = torchaudio.load(audio_paths[i].__str__())
        waveform = torch.tensor(waveform * 32767, dtype=torch.int16)
        pd_file = pd.read_csv(audio_timestamps[i].__str__(), sep=' ', header=None)
        timestamps = torch.tensor(pd_file.values)[:-1]
        start_time = timestamps[0][0]
        if master_timestamp is None:
            master_timestamp = timestamps
        else:
            master_timestamp = torch.cat((master_timestamp, timestamps), dim=0)
        end_time = timestamps[-1][2]
        tmp_output = torch.zeros((end_time-start_time) * sample_rate)
        for j in range(timestamps.shape[0]):
            entry_start_sample = (timestamps[j][0]-start_time)*sample_rate
            entry_num_of_sample = min((timestamps[j][2]-timestamps[j][0])*sample_rate, timestamps[j][3]-timestamps[j][1])
            try:
                tmp_output[entry_start_sample:entry_start_sample+entry_num_of_sample] = waveform[0][timestamps[j][1]:timestamps[j][1]+entry_num_of_sample]
            except:
                entry_num_of_sample = waveform.shape[1] - timestamps[j][1]
                tmp_output[entry_start_sample:entry_start_sample + entry_num_of_sample] = waveform[0][timestamps[j][1]:timestamps[j][1] + entry_num_of_sample]
        if res is None:
            res = tmp_output
        else:
            res = torch.cat((res, tmp_output), dim=0)
    return res, sample_rate, master_timestamp

def load_ecg_chunks(ecg_path, audio_timestamps):
    waveform, sample_rate = torchaudio.load(ecg_path.__str__())
    pd_file = pd.read_csv(audio_timestamps.__str__(), sep=' ', header=None)
    pd_file[0] = pd_file[0].astype(str) + " " + pd_file[1]
    pd_file[1] = pd_file[2]
    pd_file[2] = pd_file[3].astype(str) + " " + pd_file[4]
    pd_file[3] = pd_file[5]
    pd_file = pd_file.drop(4, axis=1)
    pd_file = pd_file.drop(5, axis=1)
    pd_file[0] = [int(datetime.strptime(col, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()) for col in pd_file[0]]
    pd_file[2] = [int(datetime.strptime(col, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()) for col in pd_file[2]]
    timestamps = torch.tensor(pd_file.values)
    start_time = timestamps[0][0]
    end_time = timestamps[-1][2]
    tmp_output = torch.zeros((end_time - start_time) * sample_rate)
    for j in range(timestamps.shape[0]):
        entry_start_sample = (timestamps[j][0] - start_time) * sample_rate
        entry_num_of_sample = min((timestamps[j][2] - timestamps[j][0]) * sample_rate,
                                  timestamps[j][3] - timestamps[j][1])
        try:
            tmp_output[entry_start_sample:entry_start_sample + entry_num_of_sample] = waveform[0][timestamps[j][1]:
                                                                                                  timestamps[j][
                                                                                                      1] + entry_num_of_sample]
        except:
            entry_num_of_sample = waveform.shape[1] - timestamps[j][1]
            tmp_output[entry_start_sample:entry_start_sample + entry_num_of_sample] = waveform[0][timestamps[j][1]:
                                                                                                  timestamps[j][
                                                                                                      1] + entry_num_of_sample]
    return tmp_output, sample_rate, timestamps

def load_imu(input_path, timestamp_file):
    pd_file = pd.read_csv(input_path.__str__(), sep=',', header=None)
    imu_data_raw = torch.tensor(pd_file.values)

    pd_file = pd.read_csv(timestamp_file.__str__(), sep=' ', header=None)
    timestamps = torch.tensor(pd_file.values)
    start_time = timestamps[0][0]
    end_time = timestamps[-1][2]
    sample_rate = 150
    tmp_output = torch.zeros(9, (end_time - start_time) * sample_rate)
    prev_end_time = None
    for j in range(timestamps.shape[0]):

        entry_start_sample = (timestamps[j][0] - start_time) * sample_rate
        entry_num_of_sample = min((timestamps[j][2] - timestamps[j][0]) * sample_rate, timestamps[j][3] - timestamps[j][1], imu_data_raw.shape[0] - timestamps[j][1])
        for i in range(9):
            tmp_output[i][entry_start_sample:entry_start_sample + entry_num_of_sample] = imu_data_raw[timestamps[j][1]:timestamps[j][1]+entry_num_of_sample][:, i+1]

        if prev_end_time is not None and abs(timestamps[j][0]-prev_end_time) > 10:
            entry_start_time = prev_end_time
            entry_end_time = timestamps[j][0]
            entry_start_sample = (entry_start_time - start_time) * sample_rate
            entry_num_of_sample = (entry_end_time - start_time)* sample_rate-entry_start_sample
            for i in range(9):
                tmp_output[i][entry_start_sample:entry_start_sample + entry_num_of_sample//2] = tmp_output[i][entry_start_sample-entry_num_of_sample//2 : entry_start_sample]
                tmp_output[i][entry_start_sample + entry_num_of_sample // 2:entry_start_sample + entry_num_of_sample] = tmp_output[i][entry_start_sample + entry_num_of_sample: entry_start_sample+ entry_num_of_sample + entry_num_of_sample//2]
        prev_end_time = timestamps[j][2]

    return tmp_output, sample_rate, timestamps

def align(audio_wv, audio_sample_rate, audio_timestamp, ecg_wv, ecg_sample_rate, ecg_timestamp, imu_data, imu_sample_rate, imu_timestamp):

    start_time = max(audio_timestamp[0][0], ecg_timestamp[0][0], imu_timestamp[0][0])
    audio_start_time = abs(start_time-audio_timestamp[0][0])
    ecg_start_time = abs(start_time-ecg_timestamp[0][0])
    imu_start_time = abs(start_time-imu_timestamp[0][0])
    end_time = min(audio_timestamp[-1][2], ecg_timestamp[-1][2], imu_timestamp[-1][2])
    audio_end_time = abs(end_time-audio_timestamp[0][0])
    ecg_end_time = abs(end_time-ecg_timestamp[0][0])
    imu_end_time = abs(end_time-imu_timestamp[0][0])
    audio_wv = audio_wv[audio_start_time*audio_sample_rate : audio_end_time*audio_sample_rate]
    ecg_wv = ecg_wv[ecg_start_time*ecg_sample_rate : ecg_end_time*ecg_sample_rate]
    imu_out = {}
    imu_out['acc_x'] = imu_data[0][imu_start_time * imu_sample_rate: imu_end_time * imu_sample_rate]
    imu_out['acc_y'] = imu_data[1][imu_start_time * imu_sample_rate: imu_end_time * imu_sample_rate]
    imu_out['acc_z'] = imu_data[2][imu_start_time * imu_sample_rate: imu_end_time * imu_sample_rate]
    imu_out['gyr_x'] = imu_data[3][imu_start_time * imu_sample_rate: imu_end_time * imu_sample_rate]
    imu_out['gyr_y'] = imu_data[4][imu_start_time * imu_sample_rate: imu_end_time * imu_sample_rate]
    imu_out['gyr_z'] = imu_data[5][imu_start_time * imu_sample_rate: imu_end_time * imu_sample_rate]
    imu_out['mag_x'] = imu_data[6][imu_start_time * imu_sample_rate: imu_end_time * imu_sample_rate]
    imu_out['mag_y'] = imu_data[7][imu_start_time * imu_sample_rate: imu_end_time * imu_sample_rate]
    imu_out['mag_z'] = imu_data[8][imu_start_time * imu_sample_rate: imu_end_time * imu_sample_rate]
    return audio_wv, ecg_wv, imu_out

def load_sleep_label(num_data, audio_textgrid_file, interval=30):
    tg = textgrid.TextGrid.fromFile(audio_textgrid_file.__str__())
    label = torch.zeros(num_data)
    for i in tg.getFirst('SLEEP'):
        if ('SLEEP' in i.mark):
            start = round(i.minTime / interval)
            end = min(int(i.maxTime / interval), num_data - 1)
            for j in range(start, end + 1):
                label[j] = 1
    return label

def create_chunks(audio_wv, ecg_wv, target_folder, target_sr, target_chunk_size, idx):
    Path(target_folder).mkdir(parents=True, exist_ok=True)
    len = audio_wv.shape[0]
    num_chunks = len // target_sr // target_chunk_size

    for i in range(num_chunks):
        path = (target_folder / f'audio_{idx}.wav').__str__()
        torchaudio.save(path, audio_wv[i*target_sr*target_chunk_size:(i+1)*target_sr*target_chunk_size].unsqueeze(0), target_sr)
        path = (target_folder / f'ecg_{idx}.wav').__str__()
        torchaudio.save(path, ecg_wv[i * target_sr * target_chunk_size:(i + 1) * target_sr * target_chunk_size].unsqueeze(0), target_sr)
        idx += 1

    return idx
if __name__ == '__main__':
    root = Path('D:\\Projects\\LittleBeatsPrelim\\sample_data\\align_sample')
    folders = [x for x in root.iterdir() if x.is_dir()]
    target_sr = 16000
    target_chunk_size = 30 # second
    target_folder = root / 'aligned_data'
    idx = 0
    for folder in folders:
        if folder.name == 'aligned_data':
            continue
        # dealing with audio data
        audio_paths = []
        audio_timestamps = []
        audio_folder = folder / 'Audio_cleaned'
        for file in audio_folder.iterdir():
            if(file.match('*.wav') and 'zero' not in file.name):
                audio_paths.append(file)
            if('timestamps'in file.name):
                audio_timestamps.append(file)
        audio_paths.sort()
        audio_timestamps.sort()
        audio_wv, audio_sample_rate, audio_timestamp = load_audio_chunks(audio_paths, audio_timestamps)
        audio_wv = torchaudio.transforms.Resample(audio_sample_rate, target_sr)(audio_wv)

        # dealing with ecg data
        ecg_folder = folder / 'ECG_cleaned'
        for file in ecg_folder.iterdir():
            if(file.match('*.wav') and 'zero' not in file.name):
                ecg_path = file
            if('timestamp'in file.name):
                ecg_timestamp = file
        ecg_wv, ecg_sample_rate, ecg_timestamp = load_ecg_chunks(ecg_path, ecg_timestamp)
        ecg_wv = torchaudio.transforms.Resample(ecg_sample_rate, target_sr)(ecg_wv)

        # dealing with imu data
        imu_folder = folder / 'IMU_cleaned'
        for file in imu_folder.iterdir():
            if('cleaned' in file.name):
                imu_path = file
            if('timestamp'in file.name):
                imu_timestamp = file
        imu_data, imu_sample_rate, imu_timestamp = load_imu(imu_path, imu_timestamp)

        audio_wv, ecg_wv, imu_data = align(audio_wv, target_sr, audio_timestamp, ecg_wv, target_sr, ecg_timestamp, imu_data, imu_sample_rate, imu_timestamp)
        pass
        # idx = create_chunks(audio_wv, ecg_wv, target_folder, target_sr, target_chunk_size, idx)
    print()
