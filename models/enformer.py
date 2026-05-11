import shutil
import os, sys
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import tqdm
from scipy.stats import pearsonr, spearmanr

from .utils import *

from kipoiseq.transforms import ReorderedOneHot
from enformer_pytorch import Enformer_

## Digital_Platform_transformers

class Enformer():
    def __init__(self, 
             model_path="./checks/Enformer"):
        
        self.input_len = 196608
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.pretrained_model = Enformer_.from_pretrained("./checks/Enformer")
        self.pretrained_model = self.pretrained_model.eval().to(self.device)
    
    def encode(self, seqs):
        # nt_map = ['A','C','G','T']
        seq_len = len(seqs[0])
        func_oh = ReorderedOneHot("ACGT")
        seqs_encode = torch.tensor([func_oh(seq) for seq in seqs])
        seqs_encode = seqs_encode.permute(0, 2, 1) # torch.Size([1, 4, 196608])
        
        # cropping steps, if length < 196608, pad to 196608; else, cut to 196608
        seqs_encode = padding_and_cropping(seqs_encode, seq_len, self.input_len, "Enformer") # torch.Size([1, 4, 196608])
        seqs_encode = seqs_encode.permute(0, 2, 1) # torch.Size([1, 196608, 4])
        return seqs_encode
    
    def predict(self, seqs, csv=False):
        seqs_encode = self.encode(seqs)
        seq_dataset = seqs_encode.to(torch.float32).to(self.device)
        seq_dataset = torch.utils.data.TensorDataset(seq_dataset)
        seq_loader  = torch.utils.data.DataLoader(seq_dataset, batch_size=4)

        ## direct prediction
        results = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(seq_loader)):
                outputs = self.pretrained_model(batch[0], head = 'human') # torch.Size([1, 896, 5313])
                results += outputs.tolist()
        pred_list = torch.tensor(results) # torch.Size([1, 896, 5313])
        
        ## to csv tracks (deprecated since time cost)
        pred_df_list = []
        if csv:
            seqs_encode = seqs_encode.permute(0, 2, 1)
            seqs_decode = onehot_2_seq(seqs_encode, nt_map='ACGT')
            tracks_csv = pd.read_csv("./libs/enformer/targets_human.txt", sep='\t')
            tracks_list = list(tracks_csv.loc[:,"description"])
            pred_df_list = []
            for i in range(pred_list.size(0)):
                v = pd.DataFrame(pred_list[i].numpy(), columns=tracks_list)
                v = pd.concat([ pd.DataFrame( list(seqs_decode[i]), columns=["Sequence"]) , v], axis=1)
                pred_df_list.append(v)

        return pred_list, pred_df_list
    
    def quick_valid(self, npy=False):
        
        seqs = open_fa("./valids/Enformer/test-sample.txt")
        data = torch.load('./valids/Enformer/test-sample.pt')
        test_target = data['target'].cpu()
        
        test_pred, _ = self.predict(seqs)
        test_pred = test_pred[0].cpu()
        
        if npy:
            np.save("./valids/Enformer/test-sample-preds.npy", test_pred.numpy())
        
        print("Correlation between Enformer and target: ", pearson_corr_coef(test_pred, test_target).item())
        return

    def predict_bash(self, seqs, csv_features, mode="total"):
        
        seqs_encode = self.encode(seqs)
        seq_dataset = seqs_encode.to(torch.float32).to(self.device)
        seq_dataset = torch.utils.data.TensorDataset(seq_dataset)
        seq_loader  = torch.utils.data.DataLoader(seq_dataset, batch_size=4)

        ## direct prediction
        results = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(seq_loader)):
                outputs = self.pretrained_model(batch[0], head = 'human') # torch.Size([1, 896, 5313])
                
                if mode == "center":
                    out_len = outputs.shape[1]
                    mid_start = (out_len - 1) // 2
                    mid_end = mid_start + 2 - (out_len % 2)
                    outputs = outputs[:, mid_start:mid_end, :].mean(dim=1)
                
                results += outputs.tolist()
        pred_list = torch.tensor(results) # torch.Size([1, 896, 5313])
        
        anno_df = pd.read_csv(csv_features)
        anno_df = anno_df[anno_df["model"] == "Enformer"]
        anno_df = anno_df.drop('Unnamed: 0', axis=1).reset_index()
        return pred_list, anno_df
    