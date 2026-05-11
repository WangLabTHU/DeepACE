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

from libs.others.basset import basset_load
from kipoiseq.transforms import ReorderedOneHot
from sklearn.metrics import roc_auc_score

## Digital_Platform_lightning

class Basset():
    def __init__(self, 
             model_path="./checks/Basset"):
        self.input_len = 600 # input_len = 200
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model = basset_load( os.path.join(model_path, 'pretrained_model_reloaded_th.pth') )
        self.pretrained_model = self.pretrained_model.to(self.device)
    
    def encode(self, seqs):
        # nt_map = ['A','C','G','T']
        seq_len = len(seqs[0])
        func_oh = ReorderedOneHot("ACGT")
        seqs_encode = torch.tensor([func_oh(seq) for seq in seqs])
        seqs_encode = seqs_encode.permute(0, 2, 1) # torch.Size([71886, 4, 600])
        
        # cropping steps, if length < 600, pad to 600; else, cut to 600
        seqs_encode = padding_and_cropping(seqs_encode, seq_len, self.input_len, "Basset")
        seqs_encode = seqs_encode.unsqueeze(3)
        
        return seqs_encode
        '''
        input should be like # (71886, 4, 600, 1), thus we should unsqueeze another dimension
        '''
        
    def predict(self, seqs, csv=False):
        seqs_encode = self.encode(seqs)
        seq_dataset = seqs_encode.to(torch.float32).to(self.device)
        seq_dataset = torch.utils.data.TensorDataset(seq_dataset)
        seq_loader  = torch.utils.data.DataLoader(seq_dataset, batch_size=32)
        
        results = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(seq_loader)):
                outputs = self.pretrained_model(batch[0])    
                results += outputs.tolist()
        pred_list = torch.tensor(results)
        
        pred_df = []
        if csv:
            pred_df = pd.DataFrame( pred_list.numpy(), columns=open_fa("./libs/others/basset_profile.txt") )
            pred_df = pd.concat([ pd.DataFrame(seqs, columns=["seqs"]) , pred_df], axis=1)
        
        return pred_list, pred_df
    
    def quick_valid(self, csv=False):
        basset_df = pd.read_csv("./valids/Basset/Table_DNase.csv")
        seqs = list(basset_df.loc[:,"seqs"])

        pred_list, pred_df = self.predict(seqs, csv=True)
        
        basset_auc = np.zeros(164)
        basset_profile = open_fa("./libs/others/basset_profile.txt")
        
        for i, track in enumerate(basset_profile):
            pred_i = list(pred_df.loc[:, track])
            target_i = list(basset_df.loc[:, track])
            basset_auc[i] = roc_auc_score(target_i, pred_i)
        
        df_valid = pd.DataFrame({"profile": basset_profile, "auc": basset_auc})
        print(df_valid)
        if csv:
            pred_df.to_csv("./valids/Basset/Table_prediction.csv")
            df_valid.to_csv("./valids/Basset/Table_profile.csv")
        
        return
    
    def predict_bash(self, seqs, csv_features, mode="total"): 
        
        self.pretrained_model.eval()
        
        seqs_encode = self.encode(seqs)
        seq_dataset = seqs_encode.to(torch.float32).to(self.device)
        seq_dataset = torch.utils.data.TensorDataset(seq_dataset)
        seq_loader  = torch.utils.data.DataLoader(seq_dataset, batch_size=32)
        
        results = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(seq_loader)):
                outputs = self.pretrained_model(batch[0])    
                results += outputs.tolist()
        pred_list = torch.tensor(results)
        
        anno_df = pd.read_csv(csv_features)
        anno_df = anno_df[anno_df["model"] == "Basset"]
        anno_df = anno_df.drop('Unnamed: 0', axis=1).reset_index()
        
        return pred_list, anno_df