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
from libs.others.sei import Sei_

## Digital_Platform_lightning

class Sei():
    def __init__(self, 
             model_path="./checks/Sei"):

        self.input_len = 4096
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
        model_weights = torch.load( os.path.join(model_path, "sei.pth") )
        self.pretrained_model = Sei_()
        self.pretrained_model.load_state_dict({k.replace('module.model.', ''): v for k, v in model_weights.items()})
        self.pretrained_model = self.pretrained_model.eval().to(self.device)


    def encode(self, seqs):
        # nt_map = ['A','C','G','T']
        seq_len = len(seqs[0])
        func_oh = ReorderedOneHot("ACGT", neutral_value=0)
        seqs_encode = torch.tensor([func_oh(seq) for seq in seqs])
        seqs_encode = seqs_encode.permute(0, 2, 1)
        
        # cropping steps, if length < 4096, pad to 4096; else, cut to 4096
        seqs_encode = padding_and_cropping(seqs_encode, seq_len, self.input_len, "Sei")  # torch.Size([1000, 4, 4096])     
        return seqs_encode
    

    def predict(self, seqs, csv=False):
        seqs_encode = self.encode(seqs)
        seq_dataset = seqs_encode.to(torch.float32).to(self.device)
        seq_dataset = torch.utils.data.TensorDataset(seq_dataset)
        seq_loader  = torch.utils.data.DataLoader(seq_dataset, batch_size=32)

        ## direct prediction
        results = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(seq_loader)):
                outputs = self.pretrained_model(batch[0])
                results += outputs.tolist()
        pred_list = torch.tensor(results) # torch.Size([1000, 21907])
        
        ## to csv tracks
        pred_df_list = []
        if csv:
            seqs_decode = onehot_2_seq(seqs_encode, nt_map='ACGT')
            tracks_list = list(open_fa("./libs/others/sei_profiles.txt"))

            pred_df_list = pd.DataFrame(pred_list.numpy(), columns=tracks_list)
            pred_df_list = pd.concat([ pd.DataFrame( list(seqs_decode), columns=["Sequence"]) , pred_df_list], axis=1)
        
        return pred_list, pred_df_list
    
    
    def quick_valid(self):
        
        # df = pd.read_csv("./valids/Sei/Sei_validation_dataset.csv")
        df = pd.read_csv("./valids/Sei/Sei_validation_datasets_2.csv")
        seqs = list(df.loc[:,"sequence"])
        seqs = [item.upper() for item in seqs]
        ids = list(df.loc[:,"label"]) 

        ## preprocessing
        max_len = max([len(item) for item in seqs])
        valid_seqs = []
        for seq in seqs:
            padding_length = max_len - len(seq)
            pad_left = padding_length // 2
            pad_right = padding_length - pad_left
            valid_seq = "N" * pad_left + seq + "N" * pad_right
            valid_seqs.append(valid_seq)
        
        valid_profiles = open_fa("./valids/Sei/Sei_validation_profiles.txt")
        valid_indexs = [ valid_profiles.index(str(item)) for item in ids]
        valid_targets = [1] * len(valid_indexs)
        
        ## validation
        preds, _ = self.predict(valid_seqs)

        preds = preds.cpu().numpy()
        binary_preds = (preds >= 0.5).astype(int)

        # Get the nunmber of 0/1 in preds[0]
        for i in range(binary_preds.shape[0]):
            zeros_count = np.sum(binary_preds[i] == 0)
            ones_count = np.sum(binary_preds[i] == 1)
            print(f"0 counts in preds[{i}] : {zeros_count}, 1  counts in preds[{i}]: {ones_count}")

        return
    
    def predict_bash(self, seqs, csv_features, mode="total"):
        
        seqs_encode = self.encode(seqs)
        seq_dataset = seqs_encode.to(torch.float32).to(self.device)
        seq_dataset = torch.utils.data.TensorDataset(seq_dataset)
        seq_loader  = torch.utils.data.DataLoader(seq_dataset, batch_size=32)

        ## direct prediction
        results = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(seq_loader)):
                outputs = self.pretrained_model(batch[0])
                results += outputs.tolist()
        pred_list = torch.tensor(results)
        
        anno_df = pd.read_csv(csv_features)
        anno_df = anno_df[anno_df["model"] == "Sei"]
        anno_df = anno_df.drop('Unnamed: 0', axis=1).reset_index()
        return pred_list, anno_df
        