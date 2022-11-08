import torch, torchaudio
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd

class LittleBeatsDataset(Dataset):
    def __init__(self, top_dir, transform=None, target_transform=None):
        self.top_dir = Path(top_dir)
        self.audio_dir = self.top_dir / 'audio'
        self.ecg_dir = self.top_dir / 'ecg'
        self.accz_dir = self.top_dir / 'accz'
        self.label_path = self.top_dir / 'label.csv'
        self.avg_hr_path = self.top_dir / 'avg_hr.csv'
        self.transform = transform
        self.target_transform = target_transform
        self.initialize()

    def initialize(self):
        self.labels = pd.read_csv(self.label_path.__str__(), sep=',', header=None)
        self.avg_hr = pd.read_csv(self.avg_hr_path.__str__(), sep=',', header=None)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        padded_idx = str(self.labels.iloc[idx, 0]).zfill(6)
        audio_path = self.audio_dir / f'audio_{padded_idx}.wav'
        ecg_path = self.ecg_dir / f'ecg_{padded_idx}.wav'
        accz_path = self.accz_dir / f'accz_{padded_idx}.txt'

        audio, audio_sr = torchaudio.load(audio_path.__str__())
        ecg, ecg_sr = torchaudio.load(ecg_path.__str__())
        pd_file = pd.read_csv(accz_path.__str__(), sep=' ', header=None)
        accz = torch.tensor(pd_file.values).squeeze().unsqueeze(dim=0)
        label = self.labels.iloc[idx, 1]
        avg_hr = self.avg_hr.iloc[idx, 1]

        return audio, ecg, accz, avg_hr, label