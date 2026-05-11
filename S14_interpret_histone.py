'''
分析组蛋白的富集

/home/hyu/Digital_Platform/manuals/figs4_virtual_screen_interpret_3.py

cp /home/hyu/Digital_Platform/manuals/figs4_virtual_screen_interpret/scatters/warm_MPRA_pseudo_random_mahalanobis/regression* /home/hyu/DeepACE/Preds/D04_deeptfbu/interpret_MPRA_pseudo_random_mahalanobis/
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

''' Dataset Preparation '''

pseudo_source = "random"
mode = "warm"
ratio_df_total = []

for dataset in ["MPRA", "epigenetics"]:
    if dataset == "MPRA":
        cells = ["HepG2", "K562", "SKNSH"]
    elif dataset == "epigenetics":
        cells = ["HepG2", "HepG2", "HepG2"]
        motifs = ["ELF1_1_aim", "HNF1A_1_aim", "HNF4A_1_aim"]
    else:
        raise ValueError("Invalid dataset input!")
    for i, cell in enumerate(cells):
        motif = motifs[i] if dataset == "epigenetics" else None
        plot_tag = motif.split("_")[0] if dataset == "epigenetics" else cell
        print(f"Start Processing: {dataset}, {plot_tag}, {mode}, mahalanobis, pseudo=random")
        df_path = f"./Preds/D04_deeptfbu/interpret_{dataset}_pseudo_random_mahalanobis/regression_feature_contributions_{plot_tag}.csv"
        df = pd.read_csv(df_path).drop(columns="Unnamed: 0")
        Ks = list(range(5, 101, 5))
        tmp = []
        for K in Ks:
            top_k = df.head(K)
            top_k_count = top_k['feature_clean'].str.contains('H3K9me|H3K27me', regex=True).sum()
            top_k_ratio = top_k_count / K
            bottom_k = df.tail(K)
            bottom_k_count = bottom_k['feature_clean'].str.contains('H3K9me|H3K27me', regex=True).sum()
            bottom_k_ratio = bottom_k_count / K
            tmp.append({ 'K': K, 'top_ratio': top_k_ratio, 'bottom_ratio': bottom_k_ratio})
        ratio_df = pd.DataFrame(tmp)
        ratio_df["tag"] = f"{dataset}_{plot_tag}"
        ratio_df_total.append(ratio_df)
ratio_df_total = pd.concat(ratio_df_total, ignore_index=True)    


''' Visualization '''

line_df = ratio_df_total.copy()
df = pd.read_csv("./total_features.csv")
mask = df['feature'].str.contains("H3K9me|H3K27me", regex=True)
count_h3 = mask.sum()
total_count = len(df)
ratio = count_h3 / total_count

output_dir = "./Supps/S14_interpret_histone"
tags = line_df['tag'].unique()
x_min, x_max = line_df['K'].min(), line_df['K'].max()
y_min, y_max = -0.05, 1.05 
df_total = pd.read_csv("./total_features.csv")
mask = df_total['feature'].str.contains("H3K9me|H3K27me", regex=True)
ratio_all = mask.sum() / len(df_total)
for tag in tags:
    df_tag = line_df[line_df['tag'] == tag]
    plt.figure(figsize=(6, 6))
    sns.set(style="white")
    plt.plot(df_tag['K'], df_tag['top_ratio'], color='black', marker='o', linestyle='-', linewidth=2,
             markersize=6, label='Positive (top)')
    plt.plot(df_tag['K'], df_tag['bottom_ratio'], color='pink', marker='s', linestyle='-', linewidth=2,
             markersize=6, label='Negative (bottom)')
    plt.axhline(y=ratio_all, color='gray', linestyle='--', label='All features ratio')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("K (number of features)")
    plt.ylabel("H3K9/H3K27 ratio")
    plt.title(f"{tag} H3K9/H3K27 ratio vs Top/Bottom K features")
    plt.legend(loc='upper left')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"lineplot_histone_{tag}.pdf")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")