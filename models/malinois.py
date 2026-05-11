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

import boda

## Digital_Platform_lightning

class Malinois():
    def __init__(self, 
                 model_path="./checks/Malinois"):
        params = torch.load(os.path.join(model_path, "torch_checkpoint.pt"))
        self.input_len = params['model_hparams'].input_len # input_len=200, flank to 600
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.pretrained_model = boda.common.utils.load_model(model_path)
        self.pretrained_model = self.pretrained_model.to(self.device)

        left_pad_len = (self.input_len - 200) // 2
        right_pad_len= (self.input_len - 200) - left_pad_len
        left_flank = boda.common.utils.dna2tensor(boda.common.constants.MPRA_UPSTREAM[-left_pad_len:]).unsqueeze(0)
        right_flank= boda.common.utils.dna2tensor(boda.common.constants.MPRA_DOWNSTREAM[:right_pad_len]).unsqueeze(0)
        self.flank_builder = boda.common.utils.FlankBuilder(left_flank=left_flank, right_flank=right_flank)
        self.flank_builder = self.flank_builder.to(self.device)
    
    def encode(self, seqs):
        # nt_map = ['A','C','G','T']    
        seq_len = len(seqs[0])
        seqs_encode = torch.stack([boda.common.utils.dna2tensor(x) for x in tqdm.tqdm(seqs)], dim=0) # (717741, 4, 200)
        
        # cropping steps, if length < 200, pad to 200; else, cut to 200
        seqs_encode = padding_and_cropping(seqs_encode, seq_len, 200, "Malinois")
        
        return seqs_encode

    def predict(self, seqs, csv=False):
        seqs_encode = self.encode(seqs)
        seq_dataset = torch.utils.data.TensorDataset(seqs_encode)
        seq_loader  = torch.utils.data.DataLoader(seq_dataset, batch_size=128)
        results = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(seq_loader)):
                prepped_seq = self.flank_builder( batch[0].to(self.device) )
                predictions = self.pretrained_model( prepped_seq ) + self.pretrained_model( prepped_seq.flip(dims=[1,2]) )
                predictions = predictions.div(2.)
                results.append(predictions.detach().cpu())
        pred_list = torch.cat(results, dim=0)
        
        pred_df = []
        if csv:
            pred_df = pd.DataFrame( pred_list.numpy(), columns=['K562_preds', 'HepG2_preds', 'SKNSH_preds'] )
            pred_df = pd.concat([ pd.DataFrame(seqs, columns=["seqs"]) , pred_df], axis=1)
        return pred_list, pred_df
    
    def quick_valid(self, csv=False):
        mpra_19 = pd.read_table("./valids/Malinois/Table_S2_MPRA_dataset.txt")
        mpra_19 = mpra_19.loc[ mpra_19.loc[:, ['K562_lfcSE', 'HepG2_lfcSE', 'SKNSH_lfcSE']].max(axis=1) < 1.0 ]
        mpra_df = mpra_19.loc[ mpra_19['sequence'].str.len() == 200 ].reset_index(drop=True)
        
        seqs = list(mpra_df.loc[:,"sequence"])
        
        pred_list, pred_df = self.predict(seqs, csv=True)
        if csv:
            pred_df.to_csv("./valids/Malinois/Table_S2_prediction.csv")
        
        chr_filter = (mpra_df['chr'] == 19) | (mpra_df['chr'] == 21) | (mpra_df['chr'] == '19') | (mpra_df['chr'] == '21') | (mpra_df['chr'] == 'X')
        idx_filter = mpra_df[chr_filter].index
        mpra_df_valid = mpra_df.loc[idx_filter]
        pred_df_valid = pred_df.loc[idx_filter]
        
        for cell in ['K562', 'HepG2', 'SKNSH']:
            corr = pearsonr(mpra_df_valid[f'{cell}_log2FC'], pred_df_valid[f'{cell}_preds'])
            print(f'[chr = 19 | 21 | X, pearsonr] cell: {cell}, stat: {corr[0]:.4f}, pvalue: {corr[1]}')
        for cell in ['K562', 'HepG2', 'SKNSH']:
            corr = spearmanr(mpra_df_valid[f'{cell}_log2FC'], pred_df_valid[f'{cell}_preds'])
            print(f'[chr = 19 | 21 | X, spearmanr] cell: {cell}, stat: {corr[0]:.4f}, pvalue: {corr[1]}')
        
        return pred_list, pred_df

    def predict_bash(self, seqs, csv_features, mode="total"): 
        # single output, no "mode" parameters (mode = center, center_averaged, total)
        # return a numpy array contains prediction, and a list of annotation from csv_features

        seqs_encode = self.encode(seqs)
        seq_dataset = torch.utils.data.TensorDataset(seqs_encode)
        seq_loader  = torch.utils.data.DataLoader(seq_dataset, batch_size=128)
        results = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(seq_loader)):
                prepped_seq = self.flank_builder( batch[0].to(self.device) )
                predictions = self.pretrained_model( prepped_seq ) + self.pretrained_model( prepped_seq.flip(dims=[1,2]) )
                predictions = predictions.div(2.)
                results.append(predictions.detach().cpu())
        pred_list = torch.cat(results, dim=0)
        
        anno_df = pd.read_csv(csv_features)
        anno_df = anno_df[anno_df["model"] == "Malinois"]
        anno_df = anno_df.drop('Unnamed: 0', axis=1).reset_index()
        return pred_list, anno_df