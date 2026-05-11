'''
画序列的突变后保守性结果, 在CAGI5的15个数据集

/home/hyu/Digital_Platform/manuals/fig2f_point_mutation_final.py
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
from umap import UMAP
from matplotlib.colors import TwoSlopeNorm

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

BASE_ORDER = ["A", "C", "G", "T"]
BASE2IDX = {b: i for i, b in enumerate(BASE_ORDER)}


def build_mutation_matrix(df_seqs, values):
    ref_seq = df_seqs["ref_seq"].iloc[0]
    L = len(ref_seq)
    mat = [[] for _ in range(L * 4)]
    mat = np.full((L, 4), np.nan)
    for i, row in df_seqs.iterrows():
        ref_s = row["ref_seq"]
        alt_s = row["alt_seq"]
        val = values[i]
        diff_pos = np.where(np.array(list(ref_s)) != np.array(list(alt_s)))[0]        
        if len(diff_pos) != 1:
            continue
        pos = diff_pos[0]
        alt_base = alt_s[pos]
        if alt_base not in BASE2IDX:
            continue
        j = BASE2IDX[alt_base]
        if np.isnan(mat[pos, j]):
            mat[pos, j] = val
        else:
            mat[pos, j] = (mat[pos, j] + val) / 2.0 
    return mat, ref_seq


def plot_heat_row(ax, mat, ref_seq, title, algo_name):
    vals = mat[~np.isnan(mat)]
    if len(vals) > 0:
        vmin = np.nanpercentile(vals, 1)
        vmax = np.nanpercentile(vals, 99)
    else:
        vmin, vmax = -1, 1
    mat = (mat - vmin) / (vmax - vmin) * 2 - 1
    sns.heatmap(
        mat.T,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        cbar=False,
        linewidths=0.05,
        linecolor="gray"
    )
    L = len(ref_seq)
    for pos in range(L):
        base = ref_seq[pos]
        j = BASE2IDX[base]
        ax.add_patch(plt.Rectangle(
            (pos, j),
            1, 1,
            fill=False,
            edgecolor="black",
            linewidth=1.0
        ))
    ax.set_ylabel("Base")
    ax.set_yticks(np.arange(4) + 0.5)
    ax.set_yticklabels(BASE_ORDER, rotation=0)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_title(f"{title} ({algo_name})", fontsize=12)

def find_min_lfc_window(mat_lfc, ref_seq, window_size=None):
    L = len(ref_seq)
    if window_size is None or window_size >= L:
        return mat_lfc, ref_seq, 0, L
    lfc_vals = np.nan_to_num(mat_lfc, nan=0.0)  # shape (L,4)
    # finding the most conservative mutations
    pos_lfc = np.nanmin(lfc_vals, axis=1)  # shape (L,)
    window_sum = np.convolve(pos_lfc, np.ones(window_size), mode='valid')
    start = np.argmin(window_sum)
    end = start + window_size
    return mat_lfc[start:end, :], ref_seq[start:end], start, end

def plot_motif_mutation_heatmaps( motif, df_seqs, df_pca, df_evo2, df_promoterAI, 
                                 output_dir, window_size=60):
    mat_lfc, ref_seq = build_mutation_matrix( df_seqs, df_seqs["VariantExpressionEffect (log2)"].values)
    mat_pca, _ = build_mutation_matrix(df_seqs, df_pca["scores"].values)
    mat_evo2, _ = build_mutation_matrix(df_seqs, df_evo2["scores"].values)
    mat_promAI, _ = build_mutation_matrix(df_seqs, df_promoterAI["scores"].values)
    mat_lfc_slice, ref_seq_slice, s, e = find_min_lfc_window(mat_lfc, ref_seq, window_size)
    mat_pca_slice = mat_pca[s:e, :]
    mat_evo2_slice = mat_evo2[s:e, :]
    mat_promAI_slice = mat_promAI[s:e, :]
    
    heatmap_data = {
        'Reference_Sequence': list(ref_seq_slice),
        'Position': range(s, e)
    }
    for i, nuc in enumerate(['A', 'C', 'G', 'T']):
        heatmap_data[f'LFC_{nuc}'] = mat_lfc_slice[:, i]
        heatmap_data[f'PCA_{nuc}'] = mat_pca_slice[:, i]
        heatmap_data[f'Evo2_{nuc}'] = mat_evo2_slice[:, i]
        heatmap_data[f'PromAI_{nuc}'] = mat_promAI_slice[:, i]
    save_name_base = f"heatmap_{motif}_combined"
    csv_save_path = os.path.join(output_dir, f"{save_name_base}.csv")
    # pd.DataFrame(heatmap_data).to_csv(csv_save_path, index=False)
    print(f"Saved heatmap data for {motif} → {csv_save_path}")
    
    L2 = len(ref_seq_slice)
    figsize = (L2 / 3, 8)
    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=figsize,
        dpi=200,
        sharex=True
    )
    plot_heat_row(axes[0], mat_lfc_slice,  ref_seq_slice, "Variant Expression Effect (log2)", "LFC")
    plot_heat_row(axes[1], mat_pca_slice,  ref_seq_slice, "Model Score", "PCA")
    plot_heat_row(axes[2], mat_evo2_slice, ref_seq_slice, "Model Score", "Evo2")
    plot_heat_row(axes[3], mat_promAI_slice, ref_seq_slice, "Model Score", "promoterAI")
    axes[-1].set_xticks(np.arange(L2) + 0.5)
    axes[-1].set_xticklabels(list(ref_seq_slice), fontsize=6)
    axes[-1].set_xlabel(f"Position (window {s} - {e})")
    fig.suptitle(f"{motif}: Mutation Heatmaps", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"conservation_heatmap_{motif}.pdf")
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Saved slice heatmap {motif}: {s}-{e} → {save_path}")
    


def plot_motif_per_base_conservation(motif, df_seqs, df_pca, df_evo2, df_promoterAI, output_dir):
    mat_lfc, ref_seq = build_mutation_matrix(
        df_seqs,
        df_seqs["VariantExpressionEffect (log2)"].values
    )
    mat_pca, _ = build_mutation_matrix(df_seqs, df_pca["scores"].values)
    mat_evo2, _ = build_mutation_matrix(df_seqs, df_evo2["scores"].values)
    mat_promAI, _ = build_mutation_matrix(df_seqs, df_promoterAI["scores"].values)
    cons_lfc      = np.nanmin(mat_lfc, axis=1)
    cons_pca      = np.nanmin(mat_pca, axis=1)
    cons_evo2     = np.nanmin(mat_evo2, axis=1)
    cons_promAI   = np.nanmin(mat_promAI, axis=1)
    L = len(ref_seq)
    x = np.arange(L)
    conservation_data = {
        'Reference_Sequence': list(ref_seq),
        'Position': x,
        'LFC_Conservation': cons_lfc,
        'PCA_Conservation': cons_pca,
        'Evo2_Conservation': cons_evo2,
        'PromAI_Conservation': cons_promAI
    }
    save_name_base = f"per_base_conservation_{motif}"
    csv_save_path = os.path.join(output_dir, f"{save_name_base}.csv")
    # pd.DataFrame(conservation_data).to_csv(csv_save_path, index=False)
    print(f"Saved per-base conservation data for {motif} → {csv_save_path}")

    figsize = (L / 20, 8)
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=figsize, dpi=200, sharex=True)
    axes[0].plot(x, cons_lfc, color="red", linewidth=1.5)
    axes[0].set_ylabel("LFC\nConservation")
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, cons_pca, color="blue", linewidth=1.5)
    axes[1].set_ylabel("PCA\nConservation")
    axes[1].grid(alpha=0.3)
    axes[2].plot(x, cons_evo2, color="green", linewidth=1.5)
    axes[2].set_ylabel("Evo2\nConservation")
    axes[2].grid(alpha=0.3)
    axes[3].plot(x, cons_promAI, color="purple", linewidth=1.5)
    axes[3].set_ylabel("promoterAI\nConservation")
    axes[3].grid(alpha=0.3)
    axes[3].set_xlabel("Position (nt)")
    plt.suptitle(f"{motif}: Per-base Conservation", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"conservation_lineplot_{motif}.pdf")
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Saved per-base conservation plot → {save_path}")



''' Conservation Analysis '''

metric = "mahalanobis"
dataset = "MPRABase"
motif_list = ["TERT", "HBG1", "LDLR", "F9", "GP1BA", "IRF4", "IRF6", "PKLR", "ZFAND3", "SORT1", 
              "HBB", "UC88", "MYC_rs6983267", "RET", "TCF7L2"] 

output_dir = f"./Supps/S07_variant_cons_cagi5/"
print(f"Processing dataset: {dataset}")
for motif in motif_list:
    df_pca = pd.read_csv(f"./Preds/D05_mprabase/analysis_mahalanobis/pca50_variant_scores_{motif}.csv")
    df_evo2 = pd.read_csv(f"./Preds/D05_mprabase/analysis_evo2/evo2_variant_scores_{motif}.csv")
    df_promoterAI = pd.read_csv(f"./Preds/D05_mprabase/analysis_promoterai/promoterAI_variant_scores_{motif}.csv")
    df_pca = df_pca[["scores", "variant_effects"]]
    df_pca["scores"] = -df_pca["scores"]
    df_evo2 = df_evo2[["scores", "variant_effects"]]
    df_promoterAI = df_promoterAI[["scores", "variant_effects"]]
    df_seqs = pd.read_csv(f"./Datas/D05_mprabase/point_{dataset}_{motif}_saturation.tsv", sep="\t")
    plot_motif_mutation_heatmaps(
        motif, 
        df_seqs,
        df_pca,
        df_evo2,
        df_promoterAI,
        output_dir=output_dir)
    plot_motif_per_base_conservation(
        motif,
        df_seqs,
        df_pca,
        df_evo2,
        df_promoterAI,
        output_dir=output_dir)

'''
cd /home/hyu/DeepACE
conda activate Digital_Platform_lightning

'''