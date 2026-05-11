'''
/home/hyu/Digital_Platform/manuals/fig1d_bar.py
/home/hyu/Digital_Platform/manuals/fig1d_bar/mds_barplot_p10n10.pdf

mv 

mv /home/hyu/Digital_Platform/manuals/fig_dataset/control_DS0001-SID01_HepG2_pos /home/hyu/DeepACE/Preds/D05_mprabase
mv /home/hyu/Digital_Platform/manuals/fig_dataset/control_DS0001-SID01_HepG2_neg /home/hyu/DeepACE/Preds/D05_mprabase
mv /home/hyu/Digital_Platform/manuals/fig_dataset/control_DS0001-SID02_HepG2_pos /home/hyu/DeepACE/Preds/D05_mprabase
mv /home/hyu/Digital_Platform/manuals/fig_dataset/control_DS0001-SID02_HepG2_neg /home/hyu/DeepACE/Preds/D05_mprabase
mv /home/hyu/Digital_Platform/manuals/fig_dataset/control_DS0002-SID02_HepG2_pos /home/hyu/DeepACE/Preds/D05_mprabase
mv /home/hyu/Digital_Platform/manuals/fig_dataset/control_DS0002-SID02_HepG2_neg /home/hyu/DeepACE/Preds/D05_mprabase

mv /home/hyu/Digital_Platform/manuals/fig_dataset/control_MPRA_HepG2_pos /home/hyu/DeepACE/Preds/D06_mpra
mv /home/hyu/Digital_Platform/manuals/fig_dataset/control_MPRA_HepG2_neg /home/hyu/DeepACE/Preds/D06_mpra
mv /home/hyu/Digital_Platform/manuals/fig_dataset/control_MPRA_K562_pos /home/hyu/DeepACE/Preds/D06_mpra
mv /home/hyu/Digital_Platform/manuals/fig_dataset/control_MPRA_K562_neg /home/hyu/DeepACE/Preds/D06_mpra
mv /home/hyu/Digital_Platform/manuals/fig_dataset/control_MPRA_SKNSH_pos /home/hyu/DeepACE/Preds/D06_mpra
mv /home/hyu/Digital_Platform/manuals/fig_dataset/control_MPRA_SKNSH_neg /home/hyu/DeepACE/Preds/D06_mpra

mkdir -p /home/hyu/DeepACE/Preds/D04_deeptfbu/control_Epigenetics_118TF_neg && cp /home/hyu/Digital_Platform/manuals/fig_dataset/rand_Epigenetics_118TF_neg/* /home/hyu/DeepACE/Preds/D04_deeptfbu/control_Epigenetics_118TF_neg
mkdir -p /home/hyu/DeepACE/Preds/D04_deeptfbu/control_Epigenetics_118TF_pos && cp /home/hyu/Digital_Platform/manuals/fig_dataset/rand_Epigenetics_118TF_pos/* /home/hyu/DeepACE/Preds/D04_deeptfbu/control_Epigenetics_118TF_pos
mkdir -p /home/hyu/DeepACE/Preds/D07_lentimpra/control_lentiMPRA_HepG2_neg && cp /home/hyu/Digital_Platform/manuals/fig_dataset/rand_lentiMPRA_HepG2_neg/* /home/hyu/DeepACE/Preds/D07_lentimpra/control_lentiMPRA_HepG2_neg
mkdir -p /home/hyu/DeepACE/Preds/D07_lentimpra/control_lentiMPRA_HepG2_pos && cp /home/hyu/Digital_Platform/manuals/fig_dataset/rand_lentiMPRA_HepG2_pos/* /home/hyu/DeepACE/Preds/D07_lentimpra/control_lentiMPRA_HepG2_pos
mkdir -p /home/hyu/DeepACE/Preds/D07_lentimpra/control_lentiMPRA_K562_neg && cp /home/hyu/Digital_Platform/manuals/fig_dataset/rand_lentiMPRA_K562_neg/* /home/hyu/DeepACE/Preds/D07_lentimpra/control_lentiMPRA_K562_neg
mkdir -p /home/hyu/DeepACE/Preds/D07_lentimpra/control_lentiMPRA_K562_pos && cp /home/hyu/Digital_Platform/manuals/fig_dataset/rand_lentiMPRA_K562_pos/* /home/hyu/DeepACE/Preds/D07_lentimpra/control_lentiMPRA_K562_pos
mkdir -p /home/hyu/DeepACE/Preds/D07_lentimpra/control_lentiMPRA_WTC11_neg && cp /home/hyu/Digital_Platform/manuals/fig_dataset/rand_lentiMPRA_WTC11_neg/* /home/hyu/DeepACE/Preds/D07_lentimpra/control_lentiMPRA_WTC11_neg
mkdir -p /home/hyu/DeepACE/Preds/D07_lentimpra/control_lentiMPRA_WTC11_pos && cp /home/hyu/Digital_Platform/manuals/fig_dataset/rand_lentiMPRA_WTC11_pos/* /home/hyu/DeepACE/Preds/D07_lentimpra/control_lentiMPRA_WTC11_pos

cp /home/hyu/Digital_Platform/manuals/fig1d_bar/nomds_barplot_p10n10.pdf /home/hyu/DeepACE/Figs/F01_deepace_diagram/F01d_reps_distance_1.pdf
cp /home/hyu/Digital_Platform/manuals/fig1d_bar/levenshtein_barplot_p10n10.pdf /home/hyu/DeepACE/Figs/F01_deepace_diagram/F01d_reps_distance_2.pdf

cp /home/hyu/Digital_Platform/manuals/2_datasets/control_MPRA_HepG2_pos.fasta /home/hyu/DeepACE/Preds/D06_mpra
cp /home/hyu/Digital_Platform/manuals/2_datasets/control_MPRA_HepG2_neg.fasta /home/hyu/DeepACE/Preds/D06_mpra
cp /home/hyu/Digital_Platform/manuals/2_datasets/control_MPRA_K562_pos.fasta /home/hyu/DeepACE/Preds/D06_mpra
cp /home/hyu/Digital_Platform/manuals/2_datasets/control_MPRA_K562_neg.fasta /home/hyu/DeepACE/Preds/D06_mpra
cp /home/hyu/Digital_Platform/manuals/2_datasets/control_MPRA_SKNSH_pos.fasta /home/hyu/DeepACE/Preds/D06_mpra
cp /home/hyu/Digital_Platform/manuals/2_datasets/control_MPRA_SKNSH_neg.fasta /home/hyu/DeepACE/Preds/D06_mpra

cp /home/hyu/Digital_Platform/manuals/2_datasets/control_DS0001-SID01_HepG2_pos.fasta /home/hyu/DeepACE/Preds/D05_mprabase
cp /home/hyu/Digital_Platform/manuals/2_datasets/control_DS0001-SID01_HepG2_neg.fasta /home/hyu/DeepACE/Preds/D05_mprabase
cp /home/hyu/Digital_Platform/manuals/2_datasets/control_DS0001-SID02_HepG2_pos.fasta /home/hyu/DeepACE/Preds/D05_mprabase
cp /home/hyu/Digital_Platform/manuals/2_datasets/control_DS0001-SID02_HepG2_neg.fasta /home/hyu/DeepACE/Preds/D05_mprabase
cp /home/hyu/Digital_Platform/manuals/2_datasets/control_DS0002-SID02_HepG2_pos.fasta /home/hyu/DeepACE/Preds/D05_mprabase
cp /home/hyu/Digital_Platform/manuals/2_datasets/control_DS0002-SID02_HepG2_neg.fasta /home/hyu/DeepACE/Preds/D05_mprabase

cp /home/hyu/Digital_Platform/manuals/2_zeroshot/Epigenetics/rand_Epigenetics_118TF_pos.fasta /home/hyu/DeepACE/Preds/D04_deeptfbu/control_Epigenetics_118TF_pos.fasta
cp /home/hyu/Digital_Platform/manuals/2_zeroshot/Epigenetics/rand_Epigenetics_118TF_neg.fasta /home/hyu/DeepACE/Preds/D04_deeptfbu/control_Epigenetics_118TF_neg.fasta
cp /home/hyu/Digital_Platform/manuals/2_zeroshot/lentiMPRA/rand_lentiMPRA_HepG2_pos.fasta /home/hyu/DeepACE/Preds/D07_lentimpra/control_lentiMPRA_HepG2_pos.fasta
cp /home/hyu/Digital_Platform/manuals/2_zeroshot/lentiMPRA/rand_lentiMPRA_HepG2_neg.fasta /home/hyu/DeepACE/Preds/D07_lentimpra/control_lentiMPRA_HepG2_neg.fasta
cp /home/hyu/Digital_Platform/manuals/2_zeroshot/lentiMPRA/rand_lentiMPRA_K562_pos.fasta /home/hyu/DeepACE/Preds/D07_lentimpra/control_lentiMPRA_K562_pos.fasta
cp /home/hyu/Digital_Platform/manuals/2_zeroshot/lentiMPRA/rand_lentiMPRA_K562_neg.fasta /home/hyu/DeepACE/Preds/D07_lentimpra/control_lentiMPRA_K562_neg.fasta
cp /home/hyu/Digital_Platform/manuals/2_zeroshot/lentiMPRA/rand_lentiMPRA_WTC11_pos.fasta /home/hyu/DeepACE/Preds/D07_lentimpra/control_lentiMPRA_WTC11_pos.fasta
cp /home/hyu/Digital_Platform/manuals/2_zeroshot/lentiMPRA/rand_lentiMPRA_WTC11_neg.fasta /home/hyu/DeepACE/Preds/D07_lentimpra/control_lentiMPRA_WTC11_neg.fasta
        
        else:
            file_pos = f"/home/hyu/Digital_Platform/manuals/2_zeroshot/{dataset}/rand_{dataset}_{tag}_pos.fasta"
            file_neg = f"/home/hyu/Digital_Platform/manuals/2_zeroshot/{dataset}/rand_{dataset}_{tag}_neg.fasta"

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


def silhouette_single_class(X):
    dist = pairwise_distances(X, metric='euclidean')
    return np.mean(np.mean(dist, axis=1))

datasets = ["DS", "MPRA", "Epigenetics", "lentiMPRA"]
plot_tags = {
    "DS": ["DS0001-SID01", "DS0001-SID02", "DS0002-SID02"],
    "MPRA": ["HepG2", "K562", "SKNSH"],
    "Epigenetics": ["118TF"],
    "lentiMPRA": ["HepG2", "K562", "WTC11"],
}
ratio_tag = "P10-N10"    
ds_label_map = ["DS-lentiMPRA-M", "DS-lentiMPRA-WT", "DS-STARR-seq"]


'''
intra-class distance in representation space
'''

# result_labels = []
# pos_values = []
# neg_values = []
# for dataset in datasets:
#     for tag in plot_tags[dataset]:
#         if dataset == "MPRA":
#             file_pos = f"./Preds/D06_mpra/control_{dataset}_{tag}_pos/uni_pred.npy"
#             file_neg = f"./Preds/D06_mpra/control_{dataset}_{tag}_neg/uni_pred.npy"
#         elif dataset == "DS":
#             file_pos = f"./Preds/D05_mprabase/control_{tag}_HepG2_pos/uni_pred.npy"
#             file_neg = f"./Preds/D05_mprabase/control_{tag}_HepG2_neg/uni_pred.npy"
#         elif dataset == "Epigenetics":
#             file_pos = f"./Preds/D04_deeptfbu/control_{dataset}_{tag}_pos/uni_pred.npy"
#             file_neg = f"./Preds/D04_deeptfbu/control_{dataset}_{tag}_neg/uni_pred.npy"
#         elif dataset == "lentiMPRA":
#             file_pos = f"./Preds/D07_lentimpra/control_{dataset}_{tag}_pos/uni_pred.npy"
#             file_neg = f"./Preds/D07_lentimpra/control_{dataset}_{tag}_neg/uni_pred.npy"
#         data_pos = np.load(file_pos)
#         data_neg = np.load(file_neg)
#         pos_num = 85
#         neg_num = 85
#         np.random.seed(42)
#         idx_pos = np.random.choice(data_pos.shape[0], pos_num, replace=False)
#         idx_neg = np.random.choice(data_neg.shape[0], neg_num, replace=False)
#         data_pos = data_pos[idx_pos]
#         data_neg = data_neg[idx_neg]

#         data = np.vstack([data_pos, data_neg])
#         scaler = StandardScaler()
#         data = scaler.fit_transform(data)
#         labels = np.array([1]*pos_num + [0]*neg_num)
#         pca = PCA(n_components=50, random_state=42)
#         data_pca = pca.fit_transform(data)
#         emb = data_pca

#         sil_pos = silhouette_single_class(emb[labels == 1])
#         sil_neg = silhouette_single_class(emb[labels == 0])
#         if dataset == "DS":
#             result_labels.append(ds_label_map.pop(0))
#         else:
#             result_labels.append(f"{dataset}-{tag}")
#         pos_values.append(sil_pos)
#         neg_values.append(sil_neg)

# x = np.arange(len(result_labels))
# width = 0.35
# plt.figure(figsize=(9, 5))
# plt.bar(x - width/2, pos_values, width, label="Positive", alpha=0.9, color='#e5c185', edgecolor="k")
# plt.bar(x + width/2, neg_values, width, label="Negative", alpha=0.9, color='#74a892', edgecolor="k")
# plt.ylim(0, 400)
# plt.xticks(x, result_labels, rotation=45, ha="right", fontsize=16)
# plt.yticks(fontsize=16)
# plt.ylabel("Intra-class mean distance", fontsize=16)
# plt.legend(fontsize=12)
# plt.tight_layout()
# for i in range(len(x)):
#     if neg_values[i] != 0:
#         ratio = pos_values[i] / neg_values[i]
#         plt.text(x[i] - width/2, pos_values[i] + 5, f"{ratio:.1f}×", 
#                  ha='center', va='bottom', fontsize=12)
# plt.savefig("./Figs/F01_deepace_diagram/F01d_reps_distance_1.pdf", dpi=400)
# plt.close()
# print("Saved: mds_barplot_p10n10.png")


'''
conda activate Digital_Platform_lightning
cd /home/hyu/Digital_Platform/manuals
python fig1d_bar.py

intra-class distance in sequential space
'''

import Levenshtein
import itertools

def read_fasta(fasta_path):
    sequences = []
    with open(fasta_path, 'r') as f:
        seq = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                    seq = ''
            else:
                seq += line
        if seq:
            sequences.append(seq)
    return np.array(sequences)

result_labels = []
pos_values = []
neg_values = []
for dataset in datasets:
    for tag in plot_tags[dataset]:
        if dataset == "MPRA":
            file_pos = f"./Preds/D06_mpra/control_{dataset}_{tag}_pos.fasta"
            file_neg = f"./Preds/D06_mpra/control_{dataset}_{tag}_neg.fasta"
        elif dataset == "DS":
            file_pos = f"./Preds/D05_mprabase/control_{tag}_HepG2_pos.fasta"
            file_neg = f"./Preds/D05_mprabase/control_{tag}_HepG2_neg.fasta"
        elif dataset == "Epigenetics":
            file_pos = f"./Preds/D04_deeptfbu/control_{dataset}_{tag}_pos.fasta"
            file_neg = f"./Preds/D04_deeptfbu/control_{dataset}_{tag}_neg.fasta"
        elif dataset == "lentiMPRA":
            file_pos = f"./Preds/D07_lentimpra/control_{dataset}_{tag}_pos.fasta"
            file_neg = f"./Preds/D07_lentimpra/control_{dataset}_{tag}_neg.fasta"
        seq_pos = read_fasta(file_pos)
        seq_neg = read_fasta(file_neg)
        pos_num = 85
        neg_num = 85
        np.random.seed(42)
        idx_pos = np.random.choice(seq_pos.shape[0], pos_num, replace=False)
        idx_neg = np.random.choice(seq_neg.shape[0], neg_num, replace=False)
        seq_pos = seq_pos[idx_pos]
        seq_neg = seq_neg[idx_neg]

        pos_values.append(np.mean([sum(c1 != c2 for c1, c2 in zip(s1, s2)) for s1, s2 in itertools.combinations(seq_pos, 2)]))
        neg_values.append(np.mean([sum(c1 != c2 for c1, c2 in zip(s1, s2)) for s1, s2 in itertools.combinations(seq_neg, 2)]))
        if dataset == "DS":
            result_labels.append(ds_label_map.pop(0))
        else:
            result_labels.append(f"{dataset}-{tag}")

x = np.arange(len(result_labels))
width = 0.35
plt.figure(figsize=(9, 5))
plt.bar(x - width/2, pos_values, width, label="Positive", alpha=0.9, color='#e5c185', edgecolor="k")
plt.bar(x + width/2, neg_values, width, label="Negative", alpha=0.9, color='#74a892', edgecolor="k")
plt.ylim(0, 180)
plt.xticks(x, result_labels, rotation=45, ha="right", fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel("Intra-class mean distance", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
for i in range(len(x)):
    if neg_values[i] != 0:
        ratio = pos_values[i] / neg_values[i]
        plt.text(x[i] - width/2, pos_values[i] + 5, f"{ratio:.1f}×", 
                 ha='center', va='bottom', fontsize=12)
plt.savefig("./Figs/F01_deepace_diagram/F01d_reps_distance_2.pdf", dpi=400)
plt.close()
print("Saved: levenshtein_barplot_p10n10.png")