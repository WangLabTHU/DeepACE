import shutil
import os, sys
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import tqdm
from scipy.stats import pearsonr, spearmanr

from .utils import *

import tensorflow as tf
from kipoiseq.transforms import ReorderedOneHot
from spliceai.utils import one_hot_encode
from tensorflow.keras.models import load_model

## Digital_Platform_lightning

class SpliceAI():
    def __init__(self, 
             model_path="./checks/SpliceAI"):

        self.pretrained_model_list = [load_model(os.path.join(model_path, "spliceai{}.h5".format(x))) for x in range(1, 6)]
        self.context = 10000
        
    def encode(self, seqs):
        
        seqs_encode = []
        for seq in seqs:
            enc = one_hot_encode('N' * (self.context//2) + seq + 'N'* (self.context//2))[None, :]
            seqs_encode.append(enc)
        
        return seqs_encode

    def predict(self, seqs):
        
        seqs_encode = self.encode(seqs)

        results = []
        with tf.device('/cpu:0'):
            for enc in seqs_encode:
                preds = np.mean([self.pretrained_model_list[m].predict(enc) for m in range(5)], axis=0)
                results.append(preds[0])
        results = torch.tensor(results).numpy()      

        return results, None
    
    def quick_valid(self):

        seqs = ['CGATCTGACGTGGGTGTAGGTAAGTGCATTATCGATATTGCAT']
        y, _ = self.predict(seqs)
        for i in range(len(y[0])):
            print(np.round(y[0, i], 2))

        if (np.round(y[0, 18:21], 2) == [0.65, 0.0, 0.35]).any():
            print("The current outputs are in alignment with the reported github issue (https://github.com/Illumina/SpliceAI/issues/39).")
        
        return
    
    def predict_bash(self, seqs, csv_features, mode="total"):
        
        seqs_encode = self.encode(seqs)
        total_samples = len(seqs_encode)
        batch_size = 128
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        results = []
        with tf.device('/cpu:0'):
            for batch_idx in range(num_batches):
                
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_samples)
                batch_enc = seqs_encode[start_idx:end_idx]
                batch_enc = np.squeeze(batch_enc, axis=1) 
                
                preds = np.mean([self.pretrained_model_list[m].predict(batch_enc) for m in range(5)], axis=0)
                preds = torch.tensor(preds) # (batch_size, seq_len, 3)

                if mode == "center":
                    out_len = preds.shape[1]
                    mid_start = (out_len - 1) // 2
                    mid_end = mid_start + 2 - (out_len % 2)
                    preds = preds[:, mid_start:mid_end, :].mean(dim=1)
                
                results += preds.tolist()
        pred_list = torch.tensor(results)
        
        anno_df = pd.read_csv(csv_features)
        anno_df = anno_df[anno_df["model"] == "SpliceAI"]
        anno_df = anno_df.drop('Unnamed: 0', axis=1).reset_index()
        return pred_list, anno_df
        