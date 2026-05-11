'''
分析HNF4A的扰动

/home/hyu/Digital_Platform/manuals/figs4_virtual_screen_interpret.py
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

import logomaker
import collections, itertools

def classify_feature(f):
    if f.startswith("CHIP-seq:H3K"):
        return "Histone"
    elif f.startswith("CHIP-seq:"):
        return "Motif"
    elif "CAGE" in f:
        return "RNA"
    elif "ATAC-seq" in f:
        return "Accessibility"
    else:
        return "Other"
    
def plot_seqlogo_comparison(top_seqs, bottom_seqs, title="SeqLogo Comparison", save_name="seqlogo_comparison.pdf"):
    top_matrix = logomaker.alignment_to_matrix(top_seqs, to_type='counts')
    bottom_matrix = logomaker.alignment_to_matrix(bottom_seqs, to_type='counts')
    top_info = logomaker.transform_matrix(top_matrix, from_type='counts', to_type='information')
    bottom_info = logomaker.transform_matrix(bottom_matrix, from_type='counts', to_type='information')
    fig, axes = plt.subplots(2, 1, figsize=(30, 6))
    logomaker.Logo(top_info,
                   ax=axes[0],
                   shade_below=0.5,
                   fade_below=0.5,
                   color_scheme='classic')
    axes[0].set_title('Top 50 (Highest Predicted)', fontsize=14, pad=15)
    axes[0].set_ylabel('Information (bits)', fontsize=12)
    logomaker.Logo(bottom_info,
                   ax=axes[1],
                   shade_below=0.5,
                   fade_below=0.5,
                   color_scheme='classic')
    axes[1].set_title('Bottom 50 (Lowest Predicted)', fontsize=14, pad=15)
    axes[1].set_ylabel('')
    axes[1].set_yticklabels([]) 
    plt.tight_layout()
    plt.savefig(save_name, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Comparison SeqLogo -> {save_name}")

def compute_kmer_freqs(seqs, K):
    freq_matrix = np.zeros((len(seqs), n_kmers))
    for i, seq in enumerate(seqs):
        seq = seq.upper()
        kmer_counts = collections.Counter(seq[j:j+K] for j in range(len(seq)-K+1))
        total_kmers = sum(kmer_counts.values())
        if total_kmers == 0:
            continue
        for j, kmer in enumerate(all_kmers):
            freq_matrix[i, j] = kmer_counts.get(kmer, 0) / total_kmers
    return freq_matrix


def plot_kmer_pcc_scatter(kmer_corr_dict, output_dir, filename_prefix="kmer_pcc_scatter"):
    os.makedirs(output_dir, exist_ok=True)
    df_deepace = kmer_corr_dict.get("deepace")
    df_deeptfbu = kmer_corr_dict.get("deeptfbu")
    df_mpra = kmer_corr_dict.get("mpra")
    if df_deepace is None or df_mpra is None:
        print("[Warning] Missing deepace or mpra data, skipping deepace vs mpra plot")
    else:
        _plot_one_scatter(
            df_x=df_deepace, df_y=df_mpra,
            x_label="DeepACE kmer PCC",
            y_label="MPRA kmer PCC",
            title="DeepACE vs MPRA: kmer Pearson Correlation Comparison",
            save_name=os.path.join(output_dir, f"{filename_prefix}_deepace_vs_mpra.pdf")
        )
    if df_deeptfbu is None or df_mpra is None:
        print("[Warning] Missing deeptfbu or mpra data, skipping deeptfbu vs mpra plot")
    else:
        _plot_one_scatter(
            df_x=df_deeptfbu, df_y=df_mpra,
            x_label="DeepTFBU kmer PCC",
            y_label="MPRA kmer PCC",
            title="DeepTFBU vs MPRA: kmer Pearson Correlation Comparison",
            save_name=os.path.join(output_dir, f"{filename_prefix}_deeptfbu_vs_mpra.pdf")
        )

def _plot_one_scatter(df_x, df_y, x_label, y_label, title, save_name):
    merged = df_x.merge(df_y, on='kmer', suffixes=('_x', '_y'))
    x = merged['pearson_r_x'].values
    y = merged['pearson_r_y'].values
    kmers = merged['kmer'].values
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, alpha=0.7, color='steelblue', s=50, edgecolor='k')
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y = x')
    r, p = pearsonr(x, y)
    plt.text(0.05, 0.95, f'Pearson r = {r:.3f}\np = {p:.2e}',
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    threshold = 0.2
    for i in range(len(kmers)):
        if abs(x[i]) > threshold or abs(y[i]) > threshold:
            plt.annotate(kmers[i], (x[i], y[i]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8, ha='left', va='bottom',
                         bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"[Saved] kmer PCC scatter -> {save_name}")


def plot_three_mode_weblogo(top_region_seqs, output_dir, save_name="seqlogo_95_120_all_top.pdf"):
    order = ["mpra", "deepace", "deeptfbu"]
    titles = {
        "mpra": "MPRA (by enhancer activity)",
        "deepace": "DeepACE (by prediction)",
        "deeptfbu": "DeepTFBU (by prediction)"
        }
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    for idx, plot_tag in enumerate(order):
        seqs = top_region_seqs[plot_tag]
        ax = axes[idx]
        counts_mat = logomaker.alignment_to_matrix(seqs, to_type='counts')
        info_mat = logomaker.transform_matrix(counts_mat, from_type='counts', to_type='information')
        logomaker.Logo(info_mat,
                       ax=ax,
                       shade_below=0.5,
                       fade_below=0.5,
                       color_scheme='classic')
        ax.set_title(titles[plot_tag], fontsize=14, pad=15)
        ax.set_ylabel('Information (bits)', fontsize=12)
        ax.set_xticks(range(0, 26, 5))
        ax.set_xticklabels(range(95, 121, 5))
        if idx < 2:
            ax.set_xlabel('')
            ax.set_xticklabels([])
    axes[-1].set_xlabel('Sequence Position (bp)', fontsize=12)    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    full_path = os.path.join(output_dir, save_name)
    plt.savefig(full_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"[Saved] 3-row WebLogo (positions 95-120) -> {full_path}")


''' Basic Settings '''

output_dir = f"./Supps/S16_interpret_weblogo/"
df_focus = pd.read_csv(f"./total_features.csv")
df_focus["original_index"] = df_focus.index.tolist()
df_focus["feature_clean"] = df_focus["feature"].replace({"CHIP:": "CHIP-seq:","CEBPb": "CEBPB","CHIP-seq:3xFLAG-": "CHIP-seq:"}, regex=True)
df_focus["feature_group"] = df_focus["feature_clean"].apply(classify_feature)
df_focus["feature_channel"] = df_focus.apply(lambda row: f"({row['model']})-({row.name})-{row['feature_clean']}", axis=1)
df_focus = df_focus.drop(columns=["Unnamed: 0"])
primary_data = np.load(f"./Preds/D04_deeptfbu/valids_Epigenetics_HNF4A_1_aim/uni_pred.npy")
labels_df = pd.read_excel("./Datas/D04_deeptfbu/3TF_MPRA.xlsx")
labels_df = labels_df[labels_df['sequence_name'].str.contains("HNF4A_1_aim", na=False)]
idx = 3939
pred_list = primary_data[:, idx]
task_name = df_focus.iloc[idx]["feature_channel"]
labels_df[task_name] = pred_list
labels_df["preds"] = [float(item.split("_")[0]) for item in labels_df["sequence_name"]]

''' Visualization '''

kmer_corr_results = {}
top_region_seqs = {}
bottom_region_seqs = {}
for tag in [task_name, "preds", "measured enhancer activity"]:
    if tag == task_name:
        plot_tag = "deepace"
    elif tag == "preds":
        plot_tag = "deeptfbu"
    else:
        plot_tag = "mpra"
    sorted_labels_df = labels_df.sort_values(by=tag, ascending=True) # task_name, "preds", 
    sorted_labels = sorted_labels_df["measured enhancer activity"].to_numpy()
    sorted_labels = np.log2(sorted_labels)
    sorted_seqs = sorted_labels_df['enhancer sequence'].tolist()
    sorted_preds = sorted_labels_df[tag].to_numpy()

    top50_seqs = sorted_seqs[-50:]
    bottom50_seqs = sorted_seqs[:50]
    top50_region = [seq[94:120].upper() for seq in top50_seqs]
    top_region_seqs[plot_tag] = top50_region
    bottom50_region = [seq[94:120].upper() for seq in bottom50_seqs]
    bottom_region_seqs[plot_tag] = bottom50_region

    bases = 'ACGT'
    K = 6
    all_kmers = [''.join(kmer) for kmer in itertools.product(bases, repeat=K)]
    n_kmers = len(all_kmers)  # 4096
    kmer_freq_matrix = compute_kmer_freqs(sorted_seqs, K)
    occurrence_count = np.sum(kmer_freq_matrix > 0, axis=0)
    n_seqs = kmer_freq_matrix.shape[0]
    min_occurrence = max(50, int(0.05 * n_seqs))
    valid_mask = occurrence_count >= min_occurrence
    print(f"Filtering kmers: keeping {valid_mask.sum()}/{n_kmers} kmers that appear in at least {min_occurrence} sequences")
    correlations = np.full(n_kmers, np.nan)
    
    for j in np.where(valid_mask)[0]:
        r, _ = pearsonr(kmer_freq_matrix[:, j], sorted_preds)
        correlations[j] = r
    correlations = np.array(correlations)
    kmer_corr_df = pd.DataFrame({'kmer': all_kmers, 'pearson_r': correlations})
    kmer_corr_df = kmer_corr_df.sort_values('pearson_r', ascending=False).reset_index(drop=True)
    kmer_corr_df = kmer_corr_df[kmer_corr_df["pearson_r"].notna()]
    kmer_corr_results[plot_tag] = kmer_corr_df.copy()

plot_kmer_pcc_scatter(
    kmer_corr_dict=kmer_corr_results,
    output_dir=output_dir,
    filename_prefix="kmer_pcc"
)
plot_three_mode_weblogo(
    top_region_seqs=top_region_seqs,
    output_dir=output_dir,
    save_name="seqlogo_95_120_top.pdf"
)
plot_three_mode_weblogo(
    top_region_seqs=bottom_region_seqs,
    output_dir=output_dir,
    save_name="seqlogo_95_120_bottom.pdf"
)