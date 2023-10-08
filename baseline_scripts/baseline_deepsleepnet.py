import torch, torchaudio
import torch.nn as nn
import tqdm
import pathlib as Path


class BaselineDeepSleepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential([
            nn.Conv1d()
        ])


    def forward(self, ecg_input):

        return x
        
