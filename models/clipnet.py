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
import h5py
import tensorflow as tf
from clipnet_mini import clipnet
from clipnet_mini import utils

## Digital_Platform_lightning

class CLIPNET():
    def __init__(self, 
             model_path="./checks/CLIPNET"):
        
        self.input_len = 1000
        nn = clipnet.CLIPNET(n_gpus=1)
        self.pretrained_model = nn.construct_ensemble(model_dir=model_path)

    def encode(self, seqs):
        # nt_map = ['A','C','G','T']
        seq_len = len(seqs[0])
        seqs_encode = [utils.TwoHotDNA(seq).twohot for seq in tqdm.tqdm(seqs, desc="Twohot encoding", disable=False)]
        seqs_encode = torch.tensor(seqs_encode)
        seqs_encode = seqs_encode.permute(0, 2, 1) 
        
        # cropping steps, if length < 1000, pad to 1000; else, cut to 1000
        seqs_encode = padding_and_cropping(seqs_encode, seq_len, self.input_len, "CLIPNET") # [5, 1000, 4]
        seqs_encode = seqs_encode.permute(0, 2, 1) 
        return seqs_encode
    
    def predict(self, seqs, csv=False):
        seqs_encode = self.encode(seqs)
        seq_dataset = tf.data.Dataset.from_tensor_slices(seqs_encode)
        seq_loader = seq_dataset.batch(32)
        
        results = []
        with tf.device('/cpu:0'):
            for i, batch in enumerate(tqdm.tqdm(seq_loader)):
                preds = self.pretrained_model.predict(batch, verbose=0) 
                results += list(preds[1])
        pred_list = torch.tensor(results)
        
        pred_df = []
        if csv:
            pred_df = pd.DataFrame( pred_list.numpy(), columns=['PRO-cap quantity'] )
            pred_df = pd.concat([ pd.DataFrame(seqs, columns=["seqs"]) , pred_df], axis=1)
        return pred_list, pred_df

    def quick_valid(self, csv=False):

        seqs = open_fa("./valids/clipnet/test.fa")
        test_preds, _ = self.predict(seqs)
        
        with h5py.File("./valids/clipnet/test_predictions_PRECOMPUTED.h5", "r") as h5file:
            dataset = h5file["quantity"]
            test_targets = dataset[:]
        
        test_targets = [item[0] for item in test_targets]
        test_preds = [item[0] for item in test_preds]
        
        if csv:
            df = pd.DataFrame( {"seqs": seqs, "quantity": test_targets, "preds": test_preds} )
            df.to_csv("./valids/clipnet/prediction.csv")
        pcc, _ = pearsonr(test_targets, test_preds)
        print("PCC values: ", pcc)
    
    def predict_bash(self, seqs, csv_features, mode="total"): 
        
        seqs_encode = self.encode(seqs)
        seq_dataset = tf.data.Dataset.from_tensor_slices(seqs_encode)
        seq_loader = seq_dataset.batch(32)
        
        results = []
        with tf.device('/cpu:0'):
            for i, batch in enumerate(tqdm.tqdm(seq_loader)):
                preds = self.pretrained_model.predict(batch, verbose=0) 
                results += list(preds[1])
        pred_list = torch.tensor(results)
        
        anno_df = pd.read_csv(csv_features)
        anno_df = anno_df[anno_df["model"] == "CLIPNET"]
        anno_df = anno_df.drop('Unnamed: 0', axis=1).reset_index()
        return pred_list, anno_df