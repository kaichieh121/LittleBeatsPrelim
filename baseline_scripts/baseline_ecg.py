import torch, torchaudio
import torch.nn as nn
import tqdm
import pathlib as Path


class BaselineEcg(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=8)
        self.conv15 = nn.Conv1d(10, 10, kernel_size=8, stride=2)
        self.conv2 = nn.Conv1d(10, 10, kernel_size=8)
        self.conv25 = nn.Conv1d(10, 10, kernel_size=8, stride=3)
        self.conv3 = nn.Conv1d(10, 10, kernel_size=8)
        self.conv35 = nn.Conv1d(10, 10, kernel_size=8, stride=4)
        self.conv4 = nn.Conv1d(10, 10, kernel_size=8)
        self.conv45 = nn.Conv1d(10, 10, kernel_size=8, stride=5)
        self.conv5 = nn.Conv1d(10, 10, kernel_size=8)
        self.conv55 = nn.Conv1d(10, 1, kernel_size=8, stride=6)
        self.linear1 = nn.Linear(664, 20)
        self.linear2 = nn.Linear(20, 20)
        self.output_layer = nn.Linear(20, 2)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()


    def forward(self, input):
        # for i in range(input.shape[0]):
        #     input[i, :] = torch.median(input[i, :])
        input = input.unsqueeze(dim=1)
        x = self.conv1(input)
        x = self.conv15(x)
        x = self.conv2(x)
        x = self.conv25(x)
        x = self.conv3(x)
        x = self.conv35(x)
        x = self.conv4(x)
        x = self.conv45(x)
        x = self.conv5(x)
        x = self.conv55(x)
        x = x.squeeze()
        x = self.relu(self.dropout(x))
        x = self.linear1(x)
        x = self.relu(self.dropout(x))
        x = self.linear2(x)
        x = self.relu(self.dropout(x))
        x = self.output_layer(x)
        return x
