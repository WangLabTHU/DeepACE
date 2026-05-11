import shutil
import os, sys
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import tqdm
from scipy.stats import pearsonr, spearmanr

from .utils import *

from transformers import AutoTokenizer, AutoModel
from valids.SegmentNT.plot import plot_features

## Digital_Platform_transformers


class SegmentNT():
    def __init__(self, 
             model_path="./checks/SegmentNT"):
  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained("./checks/SegmentNT", trust_remote_code=True)
        self.pretrained_model = AutoModel.from_pretrained("./checks/SegmentNT", trust_remote_code=True)
        self.pretrained_model = self.pretrained_model.eval().to(self.device)

    def encode(self, seqs):
        
        ## input length must be format: max_tokens -1 = len(seq) / 6 must be 4 * N
        seq_len = len(seqs[0])
        fmt_len = seq_len // 24 * 24
        crop_left = (seq_len - fmt_len) // 2
        crop_right = seq_len - fmt_len - crop_left
        
        if crop_left != 0 and crop_right != 0:
            seqs = [ item[crop_left : -crop_right] for item in seqs]
        
        seqs_encode = []
        for seq in seqs:
            tokens = self.tokenizer.batch_encode_plus([seq], return_tensors="pt")["input_ids"] # , padding="max_length", max_length = 1669 (default 5001)
            seqs_encode.append(tokens)  
        return seqs_encode
    
    def predict(self, seqs, csv=False):
        seqs_encode = self.encode(seqs)
        pred_list = []
        
        for tokens in seqs_encode:
            attention_mask = tokens != self.tokenizer.pad_token_id
            tokens = tokens.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            outs = self.pretrained_model(tokens, attention_mask=attention_mask, output_hidden_states=True)
            logits = outs.logits.detach()
            prob = torch.nn.functional.softmax(logits, dim=-1) 
            prob = prob[...,-1][0]
            pred_list.append(prob.cpu().numpy()) # (data_num, length, 14 channels)

        ## to csv tracks
        pred_df_list = []
        
        if csv:
            tracks_list = open_fa("./libs/segmentnt/features.txt")
            for i in range(len(pred_list)):
                v = pd.DataFrame(pred_list[i], columns=tracks_list)
                v = pd.concat([ pd.DataFrame( list(seqs[i]), columns=["Sequence"]) , v], axis=1)
                pred_df_list.append(v)
        
        return pred_list, pred_df_list
        
    def quick_valid(self):

        seqs = open_fa("./valids/SegmentNT/Homo_sapiens.GRCh38.chr20.valids.txt")
        features_config = open_fa("./libs/segmentnt/features.txt")
        features_rearranged = open_fa("./valids/SegmentNT/features_rearranged.txt")
        
        prob, _ = self.predict(seqs)
        plot_features( predicted_probabilities_all = prob[0], seq_length = len(seqs[0]), fig_width=20, 
                      features=features_config, order_to_plot=features_rearranged, plot_path = "./valids/SegmentNT/predicted_features.png")
        
        return
    
    
    def predict_bash(self, seqs, csv_features, mode="total"):
        
        seqs_encode = self.encode(seqs)
        pred_list = []
        
        for tokens in seqs_encode:
            attention_mask = tokens != self.tokenizer.pad_token_id
            tokens = tokens.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            outs = self.pretrained_model(tokens, attention_mask=attention_mask, output_hidden_states=True)
            logits = outs.logits.detach()
            prob = torch.nn.functional.softmax(logits, dim=-1) 
            prob = prob[...,-1][0]
            
            if mode == "center":
                out_len = prob.shape[1]
                mid_start = (out_len - 1) // 2
                mid_end = mid_start + 2 - (out_len % 2)
                prob = prob[mid_start:mid_end, :].mean(dim=0)
            
            pred_list.append(prob.tolist()) # (data_num, length, 14 channels)
            
        pred_list = torch.tensor(pred_list)

        anno_df = pd.read_csv(csv_features)
        anno_df = anno_df[anno_df["model"] == "SegmentNT"]
        anno_df = anno_df.drop('Unnamed: 0', axis=1).reset_index()
        return pred_list, anno_df