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
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve

## Digital_Platform_lightning

class SahuCNN():
    def __init__(self, 
             model_path="./checks/SahuCNN"):
        self.input_len = 170
        self.pretrained_model_ATAC = load_model(os.path.join(model_path, "GP5d_ATAC-seq_CNN/model-331-0.83.h5"))
        self.pretrained_model_STARR = load_model(os.path.join(model_path, "GP5d_genomic_enhancer_STARR-seq_CNN/model-885-0.75.h5"))
    
    def encode(self, seqs):
        # nt_map = ['A','C','G','T']
        seq_len = len(seqs[0])
        func_oh = ReorderedOneHot("ACGT")
        seqs_encode = torch.tensor([func_oh(seq) for seq in seqs])
        seqs_encode = seqs_encode.permute(0, 2, 1) 
        
        # cropping steps, if length < 170, pad to 170; else, cut to 170
        seqs_encode = padding_and_cropping(seqs_encode, seq_len, self.input_len, "SahuCNN") # [100, 4, 170]
        seqs_encode = seqs_encode.permute(0, 2, 1) 
        return seqs_encode
    
    def predict(self, seqs, csv=False):
        
        seqs_encode = self.encode(seqs)
        seq_dataset = tf.data.Dataset.from_tensor_slices(seqs_encode)
        seq_loader = seq_dataset.batch(512)
        
        results_ATAC, results_STARR = [], []
        with tf.device('/cpu:0'):
            for i, batch in enumerate(tqdm.tqdm(seq_loader)):
                preds_ATAC = self.pretrained_model_ATAC.predict(batch, verbose=0) 
                preds_STARR = self.pretrained_model_STARR.predict(batch, verbose=0)
                results_ATAC += list(preds_ATAC[:,1])
                results_STARR += list(preds_STARR[:,1])
        results_ATAC, results_STARR = torch.tensor(results_ATAC), torch.tensor(results_STARR)
        pred_list = torch.stack([results_ATAC, results_STARR], dim=0).t()
        
        pred_df = []
        if csv:
            pred_df = pd.DataFrame( pred_list.numpy(), columns=['ATAC_preds', 'STARR_preds'] )
            pred_df = pd.concat([ pd.DataFrame(seqs, columns=["seqs"]) , pred_df], axis=1)
        return pred_list, pred_df
    
    def quick_valid(self, valid_size=5000):

        df_ATAC = pd.read_csv("./valids/SahuCNN/GP5d_ATAC-seq_data/valid_GP5d_ATAC_shuffle.csv")
        seqs = list(df_ATAC.loc[:,"seqs"])[0:valid_size]
        test_targets = list(df_ATAC.loc[:,"labels"])[0:valid_size]
        test_preds, _ = self.predict(seqs)
        fpr, tpr, thresholds = roc_curve(test_targets, test_preds[:,0])
        auc = np.trapz(tpr, fpr)
        print(f' ATAC-seq, AUC: {auc:.4f}')
        
        df_STARR = pd.read_csv("./valids/SahuCNN/GP5d_genomic_enhancer_STARR-seq_data/valid_GP5d_STARR_shuffle.csv")
        seqs = list(df_STARR.loc[:,"seqs"])[0:valid_size]
        test_targets = list(df_STARR.loc[:,"labels"])[0:valid_size]
        test_preds, _ = self.predict(seqs)
        fpr, tpr, thresholds = roc_curve(test_targets, test_preds[:,1])
        auc = np.trapz(tpr, fpr)
        print(f' STARR-seq, AUC: {auc:.4f}')
        
        return
    
    def predict_bash(self, seqs, csv_features, mode="total"): 
        
        seqs_encode = self.encode(seqs)
        seq_dataset = tf.data.Dataset.from_tensor_slices(seqs_encode)
        seq_loader = seq_dataset.batch(512)
        
        results_ATAC, results_STARR = [], []
        with tf.device('/cpu:0'):
            for i, batch in enumerate(tqdm.tqdm(seq_loader)):
                preds_ATAC = self.pretrained_model_ATAC.predict(batch, verbose=0) 
                preds_STARR = self.pretrained_model_STARR.predict(batch, verbose=0)
                results_ATAC += list(preds_ATAC[:,1])
                results_STARR += list(preds_STARR[:,1])
        results_ATAC, results_STARR = torch.tensor(results_ATAC), torch.tensor(results_STARR)
        pred_list = torch.stack([results_ATAC, results_STARR], dim=0).t()
        
        anno_df = pd.read_csv(csv_features)
        anno_df = anno_df[anno_df["model"] == "SahuCNN"]
        anno_df = anno_df.drop('Unnamed: 0', axis=1).reset_index()
        return pred_list, anno_df