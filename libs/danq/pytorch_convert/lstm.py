import sys
import numpy as np
import h5py
import torch
import torch.nn as nn


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, dtype=torch.float):
        super().__init__()
        self.W_i = nn.Linear(input_size+hidden_size, hidden_size, dtype=dtype)
        self.W_c = nn.Linear(input_size+hidden_size, hidden_size, dtype=dtype)
        self.W_f = nn.Linear(input_size+hidden_size, hidden_size, dtype=dtype)
        self.W_o = nn.Linear(input_size+hidden_size, hidden_size, dtype=dtype)

    @staticmethod
    def hard_sigmoid(x:torch.Tensor):
        zeros = torch.zeros(x.size())
        ones = torch.ones(x.size())
        return torch.max(zeros, torch.min(ones, 0.2*x+0.5))
    
    def forward(self, x, h_t, c_t):

        xh = torch.cat((x, h_t), dim=1)
        
        i_t = self.hard_sigmoid(self.W_i(xh))
        f_t = self.hard_sigmoid(self.W_f(xh))
        g_t = torch.tanh(self.W_c(xh))
        o_t = self.hard_sigmoid(self.W_o(xh))
        
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t
        

class LSTM(nn.Module):

    def __init__(self, input_size, output_size, dtype=torch.float):
        super().__init__()
        self.output_size = output_size
        self.cell = LSTMCell(input_size, output_size, dtype)

    def forward(self, x):

        batch_size, seq_len, _ = x.size()
        output = []
        h_t = torch.zeros(batch_size, self.output_size)
        c_t = torch.zeros(batch_size, self.output_size)
        for t in range(seq_len):
            h_t, c_t = self.cell(x[:,t,:], h_t, c_t)
            output.append(h_t)
        
        output = torch.stack(output, dim=1)
        return output
    
class BiLSTM(nn.Module):

    def __init__(self, input_size, output_size, dtype=torch.float):
        super().__init__()
        self.forward_lstm = LSTM(input_size, output_size, dtype)
        self.backward_lstm = LSTM(input_size, output_size, dtype)

    def forward(self, x):
        forward = self.forward_lstm(x)
        backward = self.backward_lstm(torch.flip(x, dims=[0]))
        backward = torch.flip(backward, dims=[0])
        print(backward.shape)
        output = torch.cat((forward, backward), dim=2)
        return output

if __name__ == '__main__':
    lstm = BiLSTM(2, 3)
    print(lstm.state_dict())