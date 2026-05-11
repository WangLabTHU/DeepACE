import shutil
import os, sys
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import tqdm
import pytorch_lightning as pl
from scipy.stats import pearsonr, spearmanr

from .utils import *

from kipoiseq.transforms import ReorderedOneHot
from libs.puffin.puffin import Puffin_
from selene_sdk import sequences

## Digital_Platform_lightning

class Puffin():
    def __init__(self, 
             model_path="./checks/Puffin"):
        
        self.input_len = 1000 # model will automatically cut to center 650
        self.pretrained_model = Puffin_(use_cuda=True)

        state_dict = torch.load( os.path.join(model_path, "puffin.pth") , map_location=torch.device("cpu"))
        self.pretrained_model.load_state_dict(state_dict, strict=False)        

    def encode(self, seqs):
        seq_len = len(seqs[0])
        seqs_encode = []
        for seq in seqs:
            enc = sequences.sequence_to_encoding(seq, bases_arr="ACGT",
                                                 base_to_index={"A": 0, "a": 0, "C": 1, "c": 1, "G": 2, "g": 2, "T": 3, "t": 3})
            seqs_encode.append(enc)
        seqs_encode = torch.tensor(seqs_encode) # (1, 1000, 4)
        seqs_encode = seqs_encode.permute(0, 2, 1) 
        
        # cropping steps, if length < 1000, pad to 1000; else, cut to 1000
        seqs_encode = padding_and_cropping(seqs_encode, seq_len, self.input_len, "Puffin") 
        seqs_encode = seqs_encode.permute(0, 2, 1) # [1, 1000, 4]
        
        return seqs_encode
    
    def predict(self, seqs):
        seqs_encode = self.encode(seqs)
        
        results = []
        pred_df_list = []
        with torch.no_grad():
            for i in range(seqs_encode.size(0)):
                v = self.pretrained_model.predict(seqs_encode[i], seqs[i])
                results.append( v.values[2:].astype(float) )
                pred_df_list.append(v)
        pred_list = torch.tensor(results)
        return pred_list, pred_df_list
    
    def quick_valid(self, csv=False):

        seqs = open_fa("./valids/Puffin/valid.fa")
        
        _, pred_df = self.predict(seqs)
        pred_df = pred_df[0].T
        offset = len(pred_df) // 2
        pred_df['Coordinate'] = pred_df['Coordinate'] - offset
        
        if(csv):
            pred_df.to_csv("./valids/Puffin/valid_local.csv")
        
        print(pred_df)
        
        tracks = ["ENCODE-CAGE", "ENCODE-RAMPAGE", "FANTOM-CAGE", "GRO-cap", "PRO-cap"]
        for track in tracks:
            real_df = pd.read_csv("./valids/Puffin/valid_server_{}.csv".format(track))
            pred_local = list(pred_df[ "Prediciton " + track.replace("-", "_").upper() ])
            pred_server = list(real_df["Prediction"])
            real_server = list(real_df["Experiment"])
            
            pcc_pred, _ = pearsonr(pred_server, pred_local)
            print("Track {}\n covariance of single nucleotide precision with server Prediction: {}".format(track, pcc_pred))
            
        return
    
    def predict_bash(self, seqs, csv_features, mode="total"):
        seqs_encode = self.encode(seqs)
        
        results = []
        with torch.no_grad():
            for i in range(seqs_encode.size(0)):
                v = self.pretrained_model.predict(seqs_encode[i], seqs[i])
                v = v.values[2:].astype(float) # (10, L-650)
                v = torch.tensor(v)
                
                if mode == "center":
                    out_len = v.shape[-1]
                    mid_start = (out_len - 1) // 2
                    mid_end = mid_start + 2 - (out_len % 2)
                    v = v[:, mid_start:mid_end ].mean(dim=-1)
                
                results.append(v.tolist())
        pred_list = torch.tensor(results)
        pred_list = pred_list[:, :pred_list.shape[1]//2]
        
        anno_df = pd.read_csv(csv_features)
        anno_df = anno_df[anno_df["model"] == "Puffin"]
        anno_df = anno_df.drop('Unnamed: 0', axis=1).reset_index()
        return pred_list, anno_df