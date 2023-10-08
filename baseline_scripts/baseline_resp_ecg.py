import torch, torchaudio
import torch.nn as nn
import tqdm
import pathlib as Path


class BaselineRespEcg(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(409, 2)
        self.linear2 = nn.Linear(409, 2)
        # self.conv1 = nn.Conv1d(2, 1, 1)
        # self.linear3 = nn.Linear(469, 2)
        # self.linear3 = nn.LSTM(input_size=2049, batch_first=True)


    def forward(self, audio_input, ecg_input):
        audio_input = nn.functional.normalize(audio_input, dim=1)
        ecg_input = nn.functional.normalize(ecg_input, dim=1)
        audio_x = torch.abs(torch.stft(audio_input, n_fft=4096, normalized=False, return_complex=True)).sum(dim=2)
        ecg_x = torch.abs(torch.stft(ecg_input, n_fft=4096, normalized=False, return_complex=True)).sum(dim=2)
        # audio_x = self.linear1(audio_x)
        # ecg_x = self.linear2(ecg_x)
        # x = torch.cat((audio_x.transpose(1, 2), ecg_x.transpose(1, 2)), dim=1)
        # x = self.conv1(x)
        x = self.linear1(audio_x) + self.linear2(ecg_x)
        # x = self.linear3(x.squeeze())
        return x
        
