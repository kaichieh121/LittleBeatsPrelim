import sys, os
import math
import csv
from pathlib import Path
import torch, torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Audio, display

def silence_detect(audio, rate, min_silence_len=15000, silence_thresh=-35, seek_step=1000):
    from pydub import AudioSegment, silence
    data = np.asarray(audio*32767, dtype=np.int16)

    myaudio = AudioSegment(data.tobytes(), frame_rate=rate, sample_width=data.dtype.itemsize, channels=1)
    silence = silence.detect_silence(myaudio, min_silence_len=min_silence_len, silence_thresh=silence_thresh,seek_step=seek_step)

    silence = [((start/1000),(stop/1000)) for start,stop in silence] #convert to sec

    return silence

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
    if num_channels > 1:
        axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
        axes[c].set_xlim(xlim)
    if ylim:
        axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)

def plot_spectrum(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
        axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
        axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)

