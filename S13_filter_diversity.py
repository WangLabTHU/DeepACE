'''
数据集过滤, 可视化展示diversity

/home/hyu/Digital_Platform/manuals/figs11_virtual_screen_diversity.py
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

from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
import Levenshtein

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def load_data(cell, motif=None):
    # Load primary data and labels
    if dataset == "MPRA":
        primary_data = np.load(f"./Preds/D06_mpra/valids_MPRA_AdaLead_{cell}/uni_pred.npy")
        labels_df = pd.read_csv("./Datas/D06_mpra/valids.csv")
        labels_df = labels_df[labels_df["origin"] == "AdaLead"].nlargest(500, f"{cell}_prediction")
        labels = labels_df[f"{cell}_l2fc"].to_numpy()
        seqs = labels_df["sequence"].to_numpy()
    elif dataset == "epigenetics":
        primary_data = np.load(f"./Preds/D04_deeptfbu/valids_Epigenetics_{motif}/uni_pred.npy")
        labels_df = pd.read_excel("./Datas/D04_deeptfbu/3TF_MPRA.xlsx")
        labels_df = labels_df[labels_df['sequence_name'].str.contains(motif, na=False)]
        labels = labels_df["measured enhancer activity"].to_numpy()
        labels = np.log2(labels)
        seqs = labels_df["enhancer sequence"].to_numpy()
    else:
        raise ValueError("Invalid dataset input!")
    pseudo_data = np.load("./Preds/D10_random/random_sample_1/uni_pred.npy")
    combined_data = np.vstack((primary_data, pseudo_data)) if len(pseudo_data) > 0 else primary_data
    uni_selected = PCA(n_components=50, random_state=42).fit_transform(combined_data)
    primary_data = uni_selected[:-len(pseudo_data)] if len(pseudo_data) > 0 else uni_selected
    pseudo_data = uni_selected[-len(pseudo_data):] if len(pseudo_data) > 0 else np.array([])
    return primary_data, pseudo_data, labels, seqs


def analyze_pseudo_diversity(primary_data, pseudo_data, labels, seqs, plot_tag, n_neighbors=500, metric="mahalanobis", output_dir=None):
    real_vectors = primary_data
    pseudo_vectors = pseudo_data
    n_neighbors = min(n_neighbors, len(pseudo_data))
    if metric == "mahalanobis":
        var = np.var(pseudo_vectors, axis=0)
        inv_std = 1.0 / np.sqrt(var + 1e-8)
        def diag_mahalanobis(x, y, inv_std=inv_std):
            diff = (x - y) * inv_std
            return np.sqrt(np.dot(diff, diff))
        nbrs = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=diag_mahalanobis
        ).fit(pseudo_vectors)
        distances, _ = nbrs.kneighbors(real_vectors)
        mean_distances = distances.mean(axis=1)
        pseudo_similarity = 1 - mean_distances / np.max(mean_distances)
    else:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(pseudo_vectors)
        distances, _ = nbrs.kneighbors(real_vectors)
        pseudo_similarity = 1 - np.mean(distances, axis=1)  # For cosine or other metrics
    order = np.argsort(-pseudo_similarity)
    seqs_sorted = seqs[order]
    labels_sorted = labels[order]
    group_size = 50
    def calculate_levenshtein(seq_list, step=group_size):
        num_segments = len(seq_list) // step
        levenshtein_distances = []
        for i in range(num_segments):
            start_idx = i * step
            end_idx = (i + 1) * step
            segment = seq_list[start_idx:end_idx]
            pairwise_distances = []
            for i in range(len(segment)):
                for j in range(i + 1, len(segment)):
                    pairwise_distances.append(Levenshtein.distance(segment[i], segment[j]))
            mean_distance = np.mean(pairwise_distances) if pairwise_distances else 0
            levenshtein_distances.append(mean_distance)
        return levenshtein_distances
    levenshtein_distances_sorted = calculate_levenshtein(seqs_sorted)
    
    def calculate_global_mean_levenshtein(all_seqs, max_pairs=100_000):
        n = len(all_seqs)
        if n < 2:
            return 0.0
        if n * (n - 1) // 2 > max_pairs:
            indices = np.random.choice(n, size=(max_pairs, 2), replace=True)
            indices = indices[indices[:,0] != indices[:,1]]  # 去掉自比较
            dists = [Levenshtein.distance(all_seqs[i], all_seqs[j]) for i, j in indices]
            return float(np.mean(dists))
        else:
            pairwise = [
                Levenshtein.distance(a, b)
                for i, a in enumerate(all_seqs)
                for b in all_seqs[i+1:]
            ]
            return float(np.mean(pairwise)) if pairwise else 0.0
    global_random_mean = calculate_global_mean_levenshtein(seqs, max_pairs=100_000)
    plt.figure(figsize=(10, 6))
    x = np.arange(1, len(levenshtein_distances_sorted) + 1) * group_size
    plt.plot(x, levenshtein_distances_sorted, marker='o', linestyle='-', color='b', 
             label='Sorted by pseudo-diversity')
    plt.axhline(y=global_random_mean, color='gray', linestyle='--', linewidth=1.5,
                label=f'Global random mean ({global_random_mean:.2f})')
    plt.xlim(0, x[-1] * 1.08 if len(x) > 0 else 1000)
    plt.xlabel(f'Sequence position (groups of {group_size})')
    plt.ylabel('Mean pairwise Levenshtein distance within group')
    plt.title(f'{plot_tag} — Levenshtein Distance Trend')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/diversity_{plot_tag}.pdf', dpi=400)
    plt.close()
    

''' Diversity Analysis '''    

pseudo_source = "random"
mode = "pca50"
metric_type = "mahalanobis"

output_dir = "./Supps/S13_filter_diversity"
for dataset in ["MPRA",  "epigenetics"]: #  
    if dataset == "MPRA":
        cells = ["HepG2", "K562", "SKNSH"]
    elif dataset == "epigenetics":
        cells = ["HepG2", "HepG2", "HepG2"]
        motifs = ["ELF1_1_aim", "HNF1A_1_aim", "HNF4A_1_aim"]
    else:
        raise ValueError("Invalid dataset input!")
    for i, cell in enumerate(cells):
        motif = motifs[i] if dataset == "epigenetics" else None
        plot_tag = "epigenetics_" + motif.split("_")[0] if dataset == "epigenetics" else "mpra_" + cell
        print(f"Start Processing, dataset = {dataset}, cell type / motif = {plot_tag}, processing mode = {mode}, metric type = {metric_type}, pseudo source = {pseudo_source}")
        primary_data, pseudo_data, labels, seqs = load_data(cell, motif)
        combined_data = np.vstack((primary_data, pseudo_data)) if len(pseudo_data) > 0 else primary_data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(combined_data)
        scaled_primary = scaled_data[:-len(pseudo_data)] if len(pseudo_data) > 0 else scaled_data
        scaled_pseudo = scaled_data[-len(pseudo_data):] if len(pseudo_data) > 0 else np.array([])
        primary_data = scaled_primary
        pseudo_data = scaled_pseudo
        analyze_pseudo_diversity(primary_data, pseudo_data, labels, seqs, plot_tag, output_dir=output_dir)