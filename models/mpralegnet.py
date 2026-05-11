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
from human_legnet.trainer import LitModel
from human_legnet.training_config import TrainingConfig

## Digital_Platform_lightning

class MPRALegNet():
    def __init__(self, 
             model_path="./checks/MPRALegNet",
             test_folds = list(range(1, 10 + 1)),
             info = True):
        self.input_len = 230 # input_len=200, flank to 230
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        cell_type_list = ["HepG2", "K562", "WTC11"]
        self.total_folds = 10
        self.test_folds = test_folds
        self.pretrained_model_list = []
        self.info = info

        for cell_type in cell_type_list:
            # test_fold
            for i in range(1, self.total_folds + 1):
                if i not in self.test_folds:
                    continue
                # valid_fold
                for j in range(1, self.total_folds + 1):
                    if j!= i:
                        model_file = os.path.join( model_path, "{}/md_shift_reverse_noavg_noch/config.json".format(cell_type) )
                        weight_file = os.path.join( model_path, "{}/md_shift_reverse_noavg_noch/best_model_test{}_val{}.ckpt".format(cell_type, i, j) )
                        train_cfg = TrainingConfig.from_json(model_file)
                        checkpoint = torch.load(weight_file)
                        pretrained_model = LitModel(train_cfg)
                        pretrained_model.load_state_dict(checkpoint['state_dict'])
                        pretrained_model = pretrained_model.to(self.device)
                        if self.info:
                            print(" MPRALegNet for cell type {}, test fold {}, validation fold {} loaded".format(cell_type, i, j))
                        self.pretrained_model_list.append(pretrained_model)
        self.left_flank = "AGGACCGGATCAACT"
        self.right_flank = "CATTGCGTGAACCGA"
        
    def encode(self, seqs):
        # nt_map = ['A','G','C','T']
        seq_len = len(seqs[0])
        func_oh = ReorderedOneHot("AGCT")
        
        left_flank_encode = torch.tensor(func_oh(self.left_flank)).permute(1,0) # (4,15)
        right_flank_encode = torch.tensor(func_oh(self.right_flank)).permute(1,0) # (4,15)

        seqs_encode = torch.tensor([func_oh(seq) for seq in seqs])
        seqs_encode = seqs_encode.permute(0, 2, 1) 
        # cropping steps, if length < 200, pad to 200; else, cut to 200
        seqs_encode = padding_and_cropping(seqs_encode, seq_len, 200, "MPRALegNet") # torch.Size([245852, 4, 200])
        
        left_flank_expanded = left_flank_encode.unsqueeze(0).expand(seqs_encode.size(0), -1, -1)
        right_flank_expanded = right_flank_encode.unsqueeze(0).expand(seqs_encode.size(0), -1, -1)
        seqs_encode = torch.cat((left_flank_expanded, seqs_encode, right_flank_expanded), dim=2)
        
        return seqs_encode
    
    def predict(self, seqs, csv=False):
        
        seqs_encode = self.encode(seqs)
        seq_dataset = seqs_encode.to(torch.float32).to(self.device)
        seq_dataset = torch.utils.data.TensorDataset(seq_dataset)
        seq_loader  = torch.utils.data.DataLoader(seq_dataset, batch_size=32)
        
        # model_range = self.folds * (self.folds - 1)
        model_range = len(self.pretrained_model_list) // 3
        total_pred_list = []
        
        for pretrained_model in self.pretrained_model_list:
            results = []
            with torch.no_grad():
                if self.info:
                    seq_loader = tqdm.tqdm(seq_loader)
                for i, batch in enumerate(seq_loader):
                    outputs = pretrained_model.model(batch[0])    
                    results += outputs.tolist()
            total_pred_list.append(results)
        total_pred_list = torch.tensor(total_pred_list)
        
        mean_tensors = [torch.mean(total_pred_list[i * model_range:(i + 1) * model_range], dim=0) for i in range(3)]
        pred_list = torch.stack(mean_tensors, dim=0).t()
        
        pred_df = []
        if csv:
            pred_df = pd.DataFrame( pred_list.numpy(), columns=['HepG2_preds', 'K562_preds', 'WTC11_preds'] )
            pred_df = pd.concat([ pd.DataFrame(seqs, columns=["seqs"]) , pred_df], axis=1)
        return pred_list, pred_df
    
    def quick_valid(self):
        
        cell_type_list = ["HepG2", "K562", "WTC11"] # 245852, 393328, 92370
        ## validation on training datasets
        valid_size=1000
        self.__init__(test_folds = list(range(1, 3 + 1)), info=False)
        for i in range(len(cell_type_list)):
            df = pd.read_csv("./valids/MPRALegNet/{}.tsv".format(cell_type_list[i]), sep='\t')
            seqs = list(df.loc[:,"seq"])[:valid_size]
            test_targets = list(df.loc[:,"mean_value"])[:valid_size]
            test_preds, _ = self.predict(seqs)
            corr = pearsonr(test_targets, test_preds[:,i])
            print(f' cell type: {cell_type_list[i]}, stat: {corr[0]:.4f}, pvalue: {corr[1]}')
        
        ## 10-fold validation: separate testing
        valid_size=100
        for idx in range(1, 10 + 1):
            self.__init__(test_folds = [idx], info=False)
            for i in range(len(cell_type_list)):
                cell_type = cell_type_list[i]
                df_test = pd.read_csv(f"./valids/MPRALegNet/test_folds/{cell_type}_testfold_{idx}.csv")
                seqs = list(df_test["seq"])[:valid_size]
                test_targets = list(df_test["mean_value"])[:valid_size]
                test_preds, _ = self.predict(seqs)
                corr = pearsonr(test_targets, test_preds[:,i])
                print(f' cell type: {cell_type_list[i]}, test fold: {idx}, stat: {corr[0]:.4f}, pvalue: {corr[1]}')
        
        return
    
    def predict_bash(self, seqs, csv_features, mode="total"): 
        
        seqs_encode = self.encode(seqs)
        seq_dataset = seqs_encode.to(torch.float32).to(self.device)
        seq_dataset = torch.utils.data.TensorDataset(seq_dataset)
        seq_loader  = torch.utils.data.DataLoader(seq_dataset, batch_size=32)
        
        # model_range = self.folds * (self.folds - 1)
        model_range = len(self.pretrained_model_list) // 3
        total_pred_list = []
        
        for pretrained_model in self.pretrained_model_list:
            pretrained_model.eval()
            results = []
            with torch.no_grad():
                if self.info:
                    seq_loader = tqdm.tqdm(seq_loader)
                for i, batch in enumerate(seq_loader):
                    outputs = pretrained_model.model(batch[0])    
                    results += outputs.tolist()
            total_pred_list.append(results)
        total_pred_list = torch.tensor(total_pred_list)
        
        mean_tensors = [torch.mean(total_pred_list[i * model_range:(i + 1) * model_range], dim=0) for i in range(3)]
        pred_list = torch.stack(mean_tensors, dim=0).t()
        
        anno_df = pd.read_csv(csv_features)
        anno_df = anno_df[anno_df["model"] == "MPRALegNet"]
        anno_df = anno_df.drop('Unnamed: 0', axis=1).reset_index()
        return pred_list, anno_df