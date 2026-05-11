
import os, sys
import keras
import pandas as pd
from libs.aparent2.utils import *

## Digital_Platform_aparent2

class APARENT2():
    def __init__(self, 
             model_path="./checks/APARENT2"):
        self.input_len = 205
        self.pretrained_model = keras.models.load_model( os.path.join(model_path, "aparent_all_libs_resnet_no_clinvar_wt_ep_5_var_batch_size_inference_mode_no_drop.h5") )
    
    def encode(self, seqs):
        # nt_map = ['A','C','G','T']
        seq_len = len(seqs[0])
        seqs_encode = seq_2_onehot_aparent2(seqs) # (2, 1, 205, 4)
        seqs_encode = padding_and_cropping_aparent2(seqs_encode, seq_len, self.input_len, "APARENT2") # (2, 1, 205, 4)   
        return seqs_encode

    def predict(self, seqs):
        
        seqs_encode = self.encode(seqs)
        
        ## library features
        lib, lib_column = np.zeros((len(seqs), 13)), 11
        lib[:, lib_column] = 1.
        
        _, cut_pred = self.pretrained_model.predict(x=[seqs_encode, lib], batch_size=32, verbose=True) # (2, 206)
        
        return cut_pred, None
    
    def quick_valid(self):
        
        # https://github.com/johli/aparent-resnet/blob/a745736a4d9fbe411d1869200f2596a02f875532/examples/aparent2_score_variants.ipynb
        
        genes = ['PTEN', 'TP53', 'F2']
        ref_seqs = open_fa("./valids/APARENT2/ref_seqs.txt")
        var_seqs = open_fa("./valids/APARENT2/var_seqs.txt")
        df = pd.DataFrame({'gene': genes, 'ref_seq': ref_seqs, 'var_seq': var_seqs})
        
        ref_cut_pred, _ = self.predict(ref_seqs)
        var_cut_pred, _ = self.predict(var_seqs)
        
        # Calculate isoform log odds ratios (cleavage downstream of core hexamer)
        isoform_start = 77
        isoform_end = 127

        ref_iso_pred_narrow = np.sum(ref_cut_pred[:, isoform_start:isoform_end], axis=1)
        var_iso_pred_narrow = np.sum(var_cut_pred[:, isoform_start:isoform_end], axis=1)

        delta_logodds_narrow = np.log(var_iso_pred_narrow / (1. - var_iso_pred_narrow)) - np.log(
            ref_iso_pred_narrow / (1. - ref_iso_pred_narrow))

        # Calculate isoform log odds ratios (cleavage anywhere in sequence)
        isoform_start = 0
        isoform_end = 205

        ref_iso_pred = np.sum(ref_cut_pred[:, isoform_start:isoform_end], axis=1)
        var_iso_pred = np.sum(var_cut_pred[:, isoform_start:isoform_end], axis=1)

        delta_logodds = np.log(var_iso_pred / (1. - var_iso_pred)) - np.log(ref_iso_pred / (1. - ref_iso_pred))
        
        pred_df = df.copy().reset_index(drop=True)

        pred_df['delta_logodds_narrow'] = delta_logodds_narrow
        pred_df['delta_logodds'] = delta_logodds
        print(pred_df)
        
        tmp = list(pred_df.loc[:,"delta_logodds"])
        if all(round(x, 6) == y for x, y in zip(tmp, [0.180913, 1.108848, -1.107177])):
            print("The current outputs are in alignment with github examples (https://github.com/johli/aparent-resnet/blob/a745736a4d9fbe411d1869200f2596a02f875532/examples/aparent2_score_variants.ipynb).")


    def predict_bash(self, seqs, csv_features, mode="total"): 
        
        seqs_encode = self.encode(seqs)
        
        ## library features
        lib, lib_column = np.zeros((len(seqs), 13)), 11
        lib[:, lib_column] = 1.
        
        _, cut_pred = self.pretrained_model.predict(x=[seqs_encode, lib], batch_size=32, verbose=True) # (2, 206)
        
        pred_list = cut_pred[:, -1]
        pred_list = pred_list.reshape(len(pred_list), 1)

        anno_df = pd.read_csv(csv_features)
        anno_df = anno_df[anno_df["model"] == "APARENT2"]
        anno_df = anno_df.drop('Unnamed: 0', axis=1).reset_index()
        return pred_list, anno_df