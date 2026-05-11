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
import json
from basenji2_pytorch import Basenji2_

## Digital_Platform_lightning

class Basenji2():
    def __init__(self, 
             model_path="./checks/Basenji2"):

        self.input_len = 196608
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        
        params = os.path.join(model_path, "params_human.json")
        model_weights = os.path.join(model_path, "basenji2.pth")
        with open(params) as params_open:
            model_params = json.load(params_open)['model']
        self.pretrained_model = Basenji2_(model_params)
        self.pretrained_model.load_state_dict(torch.load(model_weights), strict=False)
        self.pretrained_model = self.pretrained_model.eval().to(self.device) # eval is very important!
    
    def encode(self, seqs):
        # nt_map = ['A','C','G','T']
        seq_len = len(seqs[0])
        func_oh = ReorderedOneHot("ACGT")
        seqs_encode = torch.tensor([func_oh(seq) for seq in seqs])
        seqs_encode = seqs_encode.permute(0, 2, 1) # torch.Size([1, 4, 196608])
        
        # cropping steps, if length < 196608, pad to 196608; else, cut to 196608
        seqs_encode = padding_and_cropping(seqs_encode, seq_len, self.input_len, "Basenji2") # torch.Size([1, 4, 196608])
        return seqs_encode
    
    def predict(self, seqs, csv=False):
        seqs_encode = self.encode(seqs)
        seq_dataset = seqs_encode.to(torch.float32).to(self.device)
        seq_dataset = torch.utils.data.TensorDataset(seq_dataset)
        seq_loader  = torch.utils.data.DataLoader(seq_dataset, batch_size=16)

        ## direct prediction
        results = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(seq_loader)):
                outputs = self.pretrained_model(batch[0]) # torch.Size([1, 1408, 5313])
                results += outputs.tolist()
        pred_list = torch.tensor(results) # torch.Size([1, 1408, 5313])
        
        ## to csv tracks (deprecated since time cost)
        pred_df_list = []
        if csv:
            seqs_decode = onehot_2_seq(seqs_encode, nt_map='ACGT')
            tracks_csv = pd.read_csv("./libs/basenji2/targets_human.txt", sep='\t')
            tracks_list = list(tracks_csv.loc[:,"description"])
            pred_df_list = []
            for i in range(pred_list.size(0)):
                v = pd.DataFrame(pred_list[i].numpy(), columns=tracks_list)
                v = pd.concat([ pd.DataFrame( list(seqs_decode[i]), columns=["Sequence"]) , v], axis=1)
                pred_df_list.append(v)
        
        return pred_list, pred_df_list
    
    def quick_valid(self, npy=False):
        
        seqs = open_fa("./valids/Basenji2/test-sample.txt")
        data = torch.load('./valids/Basenji2/test-sample.pt')
        test_target = data['target'].cpu()
        
        test_pred, _ = self.predict(seqs)
        start_idx = (1408 - 896) // 2
        end_idx = start_idx + 896
        test_pred = test_pred[0][start_idx:end_idx, :].cpu()
        
        if npy:
            np.save("./valids/Basenji2/test-sample-preds.npy", test_pred.numpy())
        
        print("Correlation between Basenji2 and target: ", pearson_corr_coef(test_pred, test_target).item())

    def predict_bash(self, seqs, csv_features, mode="total"):
        
        seqs_encode = self.encode(seqs)
        seq_dataset = seqs_encode.to(torch.float32).to(self.device)
        seq_dataset = torch.utils.data.TensorDataset(seq_dataset)
        seq_loader  = torch.utils.data.DataLoader(seq_dataset, batch_size=16)

        ## direct prediction
        results = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(seq_loader)):
                outputs = self.pretrained_model(batch[0]) # torch.Size([1, 1408, 5313])
                
                if mode == "center":
                    out_len = outputs.shape[1]
                    mid_start = (out_len - 1) // 2
                    mid_end = mid_start + 2 - (out_len % 2)
                    outputs = outputs[:, mid_start:mid_end, :].mean(dim=1)
                    
                results += outputs.tolist()
        pred_list = torch.tensor(results)

        
        anno_df = pd.read_csv(csv_features)
        anno_df = anno_df[anno_df["model"] == "Basenji2"]
        anno_df = anno_df.drop('Unnamed: 0', axis=1).reset_index()
        return pred_list, anno_df