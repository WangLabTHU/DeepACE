'''
seq2onehot and other utils functions
'''

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import tqdm

# allows for (batch_size, nt_map, seq_length) format
def onehot_2_seq(onehot, nt_map='ACGT'):
    if isinstance(onehot, torch.Tensor):
        onehot = onehot.cpu().numpy()  
    if isinstance(onehot, list):
        onehot = np.array(onehot)
    seqs = []
    for sample in onehot: # (4, L)
        idx = list(np.argmax(sample, axis=0))
        seq = [nt_map[i] for i in idx]
        
        ## for padding mode
        col_sums = np.sum(sample, axis=0)
        idx = np.where(col_sums == 0)[0]
        for i in idx:
            seq[i] = "N"
        
        seq = "".join(seq)    
        seqs.append(seq)
    return seqs

def padding_and_cropping(seqs_encode, seq_len, input_len, model_name):
    
    if seq_len < input_len:
        print("Start padding sequences to model settings of {}: from {} to {}".format(model_name, seq_len, input_len))
        padding_length = input_len - seq_len
        pad_left = padding_length // 2
        pad_right = padding_length - pad_left
        seqs_encode = nn.functional.pad(seqs_encode, (pad_left, pad_right), "constant", 0)
    elif seq_len > input_len:
        print("Start cropping sequences to model settings of {}: from {} to {}".format(model_name, seq_len, input_len))
        cropping_length = seq_len - input_len        
        crop_left = cropping_length // 2
        crop_right = cropping_length - crop_left
        seqs_encode = seqs_encode[:, :, crop_left:-crop_right]
    return seqs_encode    
    

def write_txt(file, data):
    f = open(file,'w')
    i = 0
    while i < len(data):
        f.write(data[i] + '\n')
        i = i + 1
    f.close()

def open_fa(file):
    record = []
    f = open(file,'r')
    for item in f:
        if '>' not in item:
            record.append(item[0:-1])
    f.close()
    return record

def pearson_corr_coef(x, y, dim = 1, reduce_dims = (-1,)):
    x_centered = x - x.mean(dim = dim, keepdim = True)
    y_centered = y - y.mean(dim = dim, keepdim = True)
    return F.cosine_similarity(x_centered, y_centered, dim = dim).mean(dim = reduce_dims)

def reverse_complement(seqs):

    rev_seqs = []
    for seq in seqs:
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        reversed_sequence = seq[::-1]
        complemented_sequence = ''.join(complement[base] for base in reversed_sequence)
        rev_seqs.append(complemented_sequence)
    return rev_seqs