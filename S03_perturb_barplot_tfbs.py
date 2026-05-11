'''
perturbation of TFBS datasets

/home/hyu/Digital_Platform/manuals/figs10_batch_perturbation.py
mv /home/hyu/Digital_Platform_Dataset/DeepTFBU/training/* /home/hyu/DeepACE/Datas/D04_deeptfbu/chip-seq
mv /home/hyu/Digital_Platform/manuals/fig_dataset/perturb_batch_deeptfbu/* /home/hyu/DeepACE/Preds/D04_deeptfbu/chip-seq
'''


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import random
random.seed(42)

import os, sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyfaidx import Fasta
from collections import Counter

from scipy.spatial import distance
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr, pearsonr, rankdata

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from sklearn.preprocessing import MinMaxScaler

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def perturb_seq(ref_seq, start, end, rep=100):
    if start < 0 or end > len(ref_seq) or start >= end:
        raise ValueError("Error settings for start and end regions")
    alt_seqs = []
    for i in range(rep):
        rand_region = ''.join(random.choice('ATCG') for _ in range(end - start))
        rand_seq = ref_seq[:start] + rand_region + ref_seq[end:]
        alt_seqs.append(rand_seq)
    return alt_seqs

def write_txt(file, data):
    f = open(file,'w')
    i = 0
    while i < len(data):
        f.write(data[i] + '\n')
        i = i + 1
    f.close()

def get_matched(pred, anno, keywords, top_cols=None, match_mode="soft"):
    """
    Parameters:
    pred : ndarray (M samples x N channels)
    anno : DataFrame with metadata for N channels
    keywords: dict or list - depends on match_mode
    top_cols: int or None
    match_mode: "soft" (default) or "hard"
    """
    if match_mode == "hard":
        if not isinstance(keywords, dict):
            raise ValueError("For hard mode, keywords must be a dictionary")
        # Check for invalid keys
        validkeys = {'celltype', 'motif'}
        inputkeys = set(keywords.keys())
        invalidkeys = inputkeys - validkeys
        if invalidkeys:
            raise ValueError(f"Illegal keys: {invalidkeys}. Only 'celltype' and/or 'motif' allowed")
        # Check if at least one valid key is present
        if not inputkeys & validkeys:
            raise ValueError("At least one valid key required: 'celltype' or 'motif'")
    elif match_mode == "soft":
        if not isinstance(keywords, list):
            raise ValueError("For soft mode, keywords must be a list")
    else:
        raise ValueError("Invalid match_mode. Must be 'hard' or 'soft'")
    if len(keywords) == 0 or anno.empty:
        return np.array([]), pd.DataFrame()
    lower_keywords = {k: [x.lower() for x in v] 
                     for k,v in keywords.items()} if match_mode == "hard" else \
                    [str(kw).lower() for kw in keywords]
    if match_mode == "hard":
        mask = pd.Series(True, index=anno.index)
        if 'celltype' in lower_keywords:
            cell_mask = anno['celltype'].str.lower().str.contains(
                '|'.join(lower_keywords['celltype']), na=False)
            mask &= cell_mask
        if 'motif' in lower_keywords:
            feat_mask = anno['feature'].str.lower().apply(
                lambda x: any(kw in x for kw in lower_keywords['motif']))
            mask &= feat_mask
        matched_idx = anno[mask].index.values
    else:  # Soft mode
        match_counts = []
        for i, row in anno.iterrows():
            row_text = ' '.join(map(str, row)).lower()
            count = sum(kw in row_text for kw in lower_keywords)
            match_counts.append(count)
        match_counts = np.array(match_counts)
        sorted_idx = np.argsort(-match_counts)
        matched_idx = sorted_idx[match_counts[sorted_idx] > 0]
    if top_cols and len(matched_idx) > top_cols:
        matched_idx = matched_idx[:top_cols]
    if len(matched_idx) == 0:
        return np.array([]), pd.DataFrame()
    matched_pred = pred[:, matched_idx]
    matched_anno = anno.loc[matched_idx].reset_index(drop=False)
    return matched_pred, matched_anno


def normalize_matrix(matrix):
    scaler = MinMaxScaler()
    normalized = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        col_data = matrix[:, j].reshape(-1, 1)
        if np.ptp(col_data) == 0:
            normalized[:, j] = 0.5
        else:
            normalized[:, j] = scaler.fit_transform(col_data).flatten()
    return normalized


''' Dataset Preparation '''

# search_path = "./Datas/D04_deeptfbu/chip-seq"
# for root, dirs, files in os.walk(search_path):
#     N = 100
#     selected_motifs = ["GABPA", "BHLHE40", "SP1", "GATA2"] # 
#     for motif in tqdm(selected_motifs):
#         ## [Part1: basic models]
#         pos_file = "pos_" + motif + "_data.bed"
#         file_path = os.path.join(search_path, pos_file)
#         df = pd.read_csv(file_path, header=None, sep="\t")
#         pos_seqs = list(df.loc[:,5])[:N]
#         pos_ppms = list(df.loc[:,6])[:N]
#         start_list = [seq.find("N") for seq in pos_seqs]
#         end_list = [seq.rfind("N") + 1 for seq in pos_seqs]
#         ref, alt, index = [], [], []
#         for i in range(len(pos_seqs)):
#             seq = pos_seqs[i]
#             start = start_list[i]
#             end = end_list[i]
#             tmp = perturb_seq(seq, start, end, rep=10)
#             ref += [seq[:start] + pos_ppms[i] + seq[end:]] * len(tmp)
#             alt += tmp
#             index += [i] * len(tmp)
        
#         work_dir = f"./Preds/D04_deeptfbu/chip-seq/pos_{motif}/"
#         if not os.path.exists(work_dir):
#             os.makedirs(work_dir)
#         write_txt(os.path.join(work_dir, "ref_seqs.txt"), ref)
#         write_txt(os.path.join(work_dir, "alt_seqs.txt"), alt)


''' Barplot Analysis '''

save_dir = "./Supps/S03_perturb_barplot_tfbs/"
for motif in ["BHLHE40", "GABPA", "GATA2", "SP1"]:
    pos_pred = np.load(f"./Preds/D04_deeptfbu/chip-seq/pos_{motif}/alt_preds/uni_pred.npy")   
    neg_pred = np.load(f"./Preds/D04_deeptfbu/chip-seq/pos_{motif}/ref_preds/uni_pred.npy")  
    uni_pred = np.concatenate([pos_pred, neg_pred], axis=0)
    uni_anno = pd.read_csv("./total_features.csv").drop(columns=["Unnamed: 0"])
    keywords = {"celltype": ["HepG2", "K562"], "motif": [motif]}
    
    filt_pred, filt_anno = get_matched(uni_pred, uni_anno, keywords=keywords, match_mode="hard")
    filt_pos_pred = filt_pred[:len(pos_pred)]
    filt_neg_pred = filt_pred[len(pos_pred):]
    pos_normalized = normalize_matrix(filt_pos_pred)
    neg_normalized = normalize_matrix(filt_neg_pred)
    down_reg_ratio = (filt_pos_pred < filt_neg_pred).mean(axis=0)
    plot_anno = filt_anno.copy().reset_index(drop=True)
    plot_anno['down_reg_accuracy'] = down_reg_ratio
    plot_anno['down_reg_accuracy'] = plot_anno['down_reg_accuracy'].round(3)
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 8))
    model_colors = {
        'DanQ':     '#1f77b4',
        'Enformer': '#ff7f0e',
        'Basenji2': '#2ca02c',
        'Expecto':  '#d62728',
        'Sei':      '#9467bd',
        'Borzoi':   '#8c564b'
    }
    ax = sns.barplot(
        data=plot_anno,
        x=plot_anno.index,                  
        y='down_reg_accuracy',              
        hue='model',                        
        palette=model_colors,
        dodge=False, 
        edgecolor='black',
        linewidth=1.2
    )
    ax.axhline(y=0.5, color='k', linewidth=2.5, linestyle=':', alpha=0.6)
    ax.set_xlabel('Channel Index', fontsize=16)
    ax.set_ylabel('Down-regulation Accuracy', fontsize=16)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=12, padding=3)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"down_acc_barplot_{motif}.pdf")
    plt.savefig(plot_path, dpi=400, bbox_inches='tight')

''' Reports '''

'''
The paths in the following code need to be converted from relative addresses to absolute addresses

"BHLHE40", "GABPA", "GATA2", "SP1"

./mutation_analysis.sh \
--pos_pred ./Preds/D04_deeptfbu/chip-seq/pos_SP1/alt_preds/uni_pred.npy \
--neg_pred ./Preds/D04_deeptfbu/chip-seq/pos_SP1/ref_preds/uni_pred.npy \
--outs_path ./Supps/S03_perturb_barplot_tfbs \
--csv_path ./Preds/D04_deeptfbu/chip-seq/pos_SP1/alt_preds/uni_anno.csv \
--mode hard \
--keywords '{"celltype": ["HepG2", "K562"], "motif": ["SP1"]}' \
--top_rows 10 --top_cols 30
'''