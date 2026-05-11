import os, sys
from libs.danq.danq import *
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

import pandas as pd

## Digital_Platform_danq

class DanQ():
    def __init__(self, 
             model_path="./checks/DanQ"):

        self.input_len = 1000
        self.pretrained_model = DanQ_().get_model()
        self.pretrained_model.load_weights( os.path.join(model_path, "DanQ_bestmodel.hdf5") )
        
    def encode(self, seqs):
        seq_len = len(seqs[0])
        seqs_encode = seq_2_onehot_py27(seqs)
        seqs_encode = padding_and_cropping_py27(seqs_encode, seq_len, self.input_len, "DanQ")
        return seqs_encode
    
    def predict(self, seqs, csv=False):
        seqs_encode = self.encode(seqs)
        pred_list = self.pretrained_model.predict(seqs_encode, verbose=0) 
        pred_list = np.array(pred_list)
        
        ## to_csv format
        pred_df = []
        
        if csv:
            df = pd.read_csv("./libs/danq/aucs.txt", sep="\t")
            cell_type = list(df.loc[:,"Cell Type"])
            marks = list(df.loc[:,"TF/DNase/HistoneMark"])
            
            pred_df = pd.DataFrame( pred_list, columns=[cell_type, marks] )
            pred_df = pd.concat([ pd.DataFrame(seqs, columns=["seqs"]) , pred_df], axis=1)
        
        return pred_list, pred_df

    def quick_valid(self):
        
        print("loading data")
        seqs = open_fa("./valids/DanQ/valid_seqs.txt")
        y = np.load("./valids/DanQ/valid_tracks.npy")
        
        df = pd.read_csv("./libs/danq/aucs.txt", sep="\t")
        cell_type = list(df.loc[:,"Cell Type"])
        marks = list(df.loc[:,"TF/DNase/HistoneMark"])
        
        print("start prediction")
        test_predicts, _ = self.predict(seqs)
        
        danq_auc = np.zeros(len(cell_type))
        for i in range(len(cell_type)):
            try:
                danq_auc[i] = roc_auc_score(y[:,i], test_predicts[:,i])
            except ValueError:
                pass
        
        df_valid = pd.DataFrame({"cell type": cell_type, "marks": marks, "auc": danq_auc})
        print(df_valid)
        df_valid.to_csv("./valids/DanQ/valid_auc.csv")

    def predict_bash(self, seqs, csv_features, mode="total"):
        
        seqs_encode = self.encode(seqs)
        pred_list = self.pretrained_model.predict(seqs_encode, verbose=0) 
        pred_list = np.array(pred_list)
        
        anno_df = pd.read_csv(csv_features)
        anno_df = anno_df[anno_df["model"] == "DanQ"]
        anno_df = anno_df.drop('Unnamed: 0', axis=1).reset_index()
        return pred_list, anno_df