from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from seya.layers.recurrent import Bidirectional

import theano
import numpy as np

class DanQ_:
    def __init__(self):
        # Define the forward and backward LSTMs
        self.forward_lstm = LSTM(input_dim=320, output_dim=320, return_sequences=True)
        self.backward_lstm = LSTM(input_dim=320, output_dim=320, return_sequences=True)
        
        # Create the bidirectional RNN
        self.brnn = Bidirectional(forward=self.forward_lstm, backward=self.backward_lstm, return_sequences=True)
        
        # Build the model
        self.model = Sequential()
        self.model.add(Convolution1D(input_dim=4,
                                      input_length=1000,
                                      nb_filter=320,
                                      filter_length=26,
                                      border_mode="valid",
                                      activation="relu",
                                      subsample_length=1))
        self.model.add(MaxPooling1D(pool_length=13, stride=13))
        self.model.add(Dropout(0.2))
        self.model.add(self.brnn)
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(input_dim=75*640, output_dim=925))
        self.model.add(Activation('relu'))
        self.model.add(Dense(input_dim=925, output_dim=919))
        self.model.add(Activation('sigmoid'))

        self.compile_model()
        
    def compile_model(self):
        print('compiling model')
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")

    def get_model(self):
        return self.model



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