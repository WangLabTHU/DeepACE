import sys
import numpy as np
import h5py
import torch
import torch.nn as nn
from .lstm import LSTM, BiLSTM

class DanQ_(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=320,
                kernel_size=26,
                padding=0,
                stride=1,
                dtype=torch.float64,
            ),
            nn.ReLU(),
        )
        self.pool = nn.Sequential(
            nn.MaxPool1d(
                kernel_size=13,
                stride=13,
            ),
            nn.Dropout(
                p=0.2,
            ),
        )
        self.biLSTM = BiLSTM(
            input_size=320,
            output_size=320,
            dtype=torch.float64,
        )
        self.drop2 = nn.Dropout(
            p=0.5,
        )
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=75*640,
                out_features=925,
                dtype=torch.float64,
            ),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(
                in_features=925,
                out_features=919,
                dtype=torch.float64,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x * 0.8
        x = torch.permute(x, (0, 2, 1))
        x = self.biLSTM(x)
        x = self.drop2(x)
        x = x * 0.5
        x = torch.flatten(x, 1, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x



def seq_2_onehot_py27(seqs):
    result = []
    for seq in seqs:
        bases = 'AGCT'
        onehot = np.zeros((len(seq), 4), dtype=int)
        for i, base in enumerate(seq):
            if base in bases:
                onehot[i, bases.index(base)] = 1
            elif base == 'N':
                pass
        result.append(onehot)
    return np.array(result)


def onehot_2_seq_py27(onehots):
    bases = 'AGCT'
    seqs = []
    for onehot in onehots:
        tmp = []
        for i in range(onehot.shape[0]):
            if np.any(onehot[i]):
                j = np.argmax(onehot[i])
                tmp.append(bases[j])
            else:
                tmp.append("N")
        seq = ''.join(tmp)
        seqs.append(seq)
    return seqs

def padding_and_cropping_py27(seqs_encode, seq_len, input_len, model_name):
    if seq_len < input_len:
        print("Start padding sequences to model settings of {}: from {} to {}".format(model_name, seq_len, input_len))
        padding_length = input_len - seq_len
        pad_left = padding_length // 2
        pad_right = padding_length - pad_left
        seqs_encode = np.pad(seqs_encode, ((0, 0), (pad_left, pad_right), (0, 0)), 'constant', constant_values=0)
    elif seq_len > input_len:
        print("Start cropping sequences to model settings of {}: from {} to {}".format(model_name, seq_len, input_len))
        cropping_length = seq_len - input_len
        crop_left = cropping_length // 2
        crop_right = cropping_length - crop_left
        seqs_encode = seqs_encode[:, crop_left:-crop_right, :]
    return seqs_encode


def open_fa(file):
    record = []
    f = open(file,'r')
    for item in f:
        if '>' not in item:
            record.append(item[0:-1])
    f.close()
    return record

def write_txt(file, data):
    f = open(file,'w')
    i = 0
    while i < len(data):
        f.write(data[i] + '\n')
        i = i + 1
    f.close()