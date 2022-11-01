import math
import os
import torch, torchaudio
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
import heartpy as hp
from pathlib import Path


def find_avg_hr(ecg_file, interval=60):
    waveform, sample_rate = torchaudio.load(ecg_file.__str__())
    waveform = waveform[0]
    num_data = math.floor(waveform.shape[0] / sample_rate / interval)
    ecg_data = waveform[:num_data * interval * sample_rate].view(num_data, interval * sample_rate)
    hr = np.array([])
    for i in range(num_data):
        try:
            working_data, measures = hp.process(ecg_data[i].numpy(), sample_rate)
            if(measures['bpm'] <= 250):
                hr = np.append(hr, measures['bpm'])
        except:
            pass
    if math.isnan(np.nanmean(hr)):
        return 150
    else:
        return np.nanmean(hr)

if __name__ == '__main__':

    warnings.filterwarnings("ignore")
    annotation_folder = Path('//ad.uillinois.edu/aces/hdfs/share/McElwain-MCRP-RA/LittleBeats_RA/LittleBeats_Sleep annotations/RA sleep annotations completed (for Jeff)')
    data_folder = Path('//ad.uillinois.edu/aces/hdfs/share/McElwain-MCRP-RA/LittleBeats_RA/sleep_study_preliminary_recordings')


    for dir in annotation_folder.iterdir():
        if dir.is_dir() and not (dir.name == "No Sleep files") and not ('Processed ECG' in dir.name):
            if dir.name == 'JAR_4039_2022-07-10-19-33-10':
                output_path = f'{dir}/avg_hr.txt'
                if os.path.exists(output_path):
                    os.remove(output_path)
                output_file = open(output_path, 'w')
                dir_name = dir.name
                orig_audio_folder = data_folder / dir_name / "Audio_cleaned"
                orig_ecg_folder = data_folder / dir_name / "ECG_cleaned"
                for file in orig_ecg_folder.iterdir():
                    if (file.match('*cleaned.wav')):
                        ecg_file = file
                    if ('timestamp' in file.name):
                        ecg_timestamp_file = file
                avg_hr = find_avg_hr(ecg_file)
                print(f'{dir_name}/avg_hr.txt {avg_hr}')
                output_file.write(f'{avg_hr}')
                output_file.close()




