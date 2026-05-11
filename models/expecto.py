import shutil
import os, sys
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import tqdm
from scipy.stats import pearsonr, spearmanr

from .utils import *

import h5py
from kipoiseq.transforms import ReorderedOneHot
from libs.others.expecto import Beluga, encodeSeqs

## Digital_Platform_lightning

class Expecto():
    def __init__(self, 
             model_path="./checks/Expecto"):

        self.input_len = 2000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.pretrained_deepsea = Beluga() 
        self.pretrained_deepsea.load_state_dict(torch.load("/home/hyu/14_Expecto/checks/deepsea.beluga.pth"))
        self.pretrained_deepsea = self.pretrained_deepsea.eval().to(self.device)
        
        # deprecated since downstream xgboost are restricted in inputs
        # modelList = pd.read_csv( "./checks/Expecto/modellist" ,sep='\t',header=0)
        # self.pretrained_xgblist = []
        # for file in modelList['ModelName']:
        #     bst = xgb.Booster({'nthread': 16})
        #     bst.load_model(file.strip())
        #     self.pretrained_xgblist.append(bst)
        
    def encode(self, seqs):
        # nt_map = ['A','G','C','T']
        seq_len = len(seqs[0])
        seqs_encode = encodeSeqs(seqs) # (10,2000) -> (20, 4, 2000)
        seqs_encode = torch.tensor(seqs_encode)
        
        # cropping steps, if length < 2000, pad to 2000; else, cut to 2000
        seqs_encode = padding_and_cropping(seqs_encode, seq_len, self.input_len, "Expecto")        
        return seqs_encode
        
    def predict(self, seqs, csv=False):
        
        seqs_encode = self.encode(seqs).unsqueeze(2) # (20, 4, 1, 2000)
        seq_dataset = seqs_encode.to(torch.float32).to(self.device)
        seq_dataset = torch.utils.data.TensorDataset(seq_dataset)
        seq_loader  = torch.utils.data.DataLoader(seq_dataset, batch_size=32)

        ## deepsea prediction
        results = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(seq_loader)):
                outputs = self.pretrained_deepsea(batch[0]) # torch.Size([20, 2002])
                results += outputs.tolist()
        deepsea_preds = torch.tensor(results)
        
        deepsea_df = []
        if csv:
            deepsea_features = pd.read_csv("./libs/others/expecto_xgboost_profiles.tsv", sep="\t")
            deepsea_celltype = list(deepsea_features.loc[:,"Cell type"])
            deepsea_assay = list(deepsea_features.loc[:,"Assay"])
            deepsea_df = pd.DataFrame( deepsea_preds.numpy(), columns=[deepsea_celltype, deepsea_assay] )
            deepsea_df = pd.concat([ pd.DataFrame(seqs + reverse_complement(seqs), columns=["seqs"]) , deepsea_df], axis=1)
        
        return deepsea_preds, deepsea_df
        
    ## xgboost only accepts 20030 length inputs (2002 * 10 + 10), inapplicable to general contexts
    
    def quick_valid(self):

        refs = open_fa("./valids/Expecto/example_refseq_shift_0.txt")
        alts = open_fa("./valids/Expecto/example_altseq_shift_0.txt")
        
        ## valids
        f = h5py.File('./valids/Expecto/example.vcf.shift_0.diff.h5')
        preds_expect = f['pred'][:]
        
        pred_refs, _ = self.predict(refs)
        pred_alts, _ = self.predict(alts)
        
        preds_delta = (pred_alts - pred_refs)
        
        profiles = open_fa("./libs/others/expecto_deepsea_profiles.txt")
        
        ## outputs
        sml_i, sml_pcc = -1, 1
        for i in range(2002):
            pcc, _ = pearsonr(preds_delta[:,i], preds_expect[:,i])
            print("PCC values of {}: {}".format(profiles[i], pcc))

            if pcc < sml_pcc:
                sml_i, sml_pcc = i, pcc
        print("-" * 100)
        print("Smallest PCC values || {}: {}".format(profiles[sml_i], sml_pcc))
        
        return
    
    def predict_bash(self, seqs, csv_features, mode="total"): 
        
        seqs_encode = self.encode(seqs).unsqueeze(2) # (20, 4, 1, 2000)
        seq_dataset = seqs_encode.to(torch.float32).to(self.device)
        seq_dataset = torch.utils.data.TensorDataset(seq_dataset)
        seq_loader  = torch.utils.data.DataLoader(seq_dataset, batch_size=32)

        ## deepsea prediction
        results = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(seq_loader)):
                outputs = self.pretrained_deepsea(batch[0]) # torch.Size([20, 2002])
                results += outputs.tolist()
        pred_list = torch.tensor(results)
        pred_list = pred_list[:len(pred_list)//2, :]
        
        anno_df = pd.read_csv(csv_features)
        anno_df = anno_df[anno_df["model"] == "Expecto"]
        anno_df = anno_df.drop('Unnamed: 0', axis=1).reset_index()
        return pred_list, anno_df