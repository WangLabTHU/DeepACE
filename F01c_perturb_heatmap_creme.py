'''
/home/hyu/Digital_Platform/manuals/figs9_seq_perturbation_3.py

mv /home/hyu/Digital_Platform/manuals/fig_dataset/perturb_seq_creme_2/perturb_datasets/* /home/hyu/DeepACE/Preds/D03_creme/
cp -r /home/hyu/Digital_Platform/manuals/figs9_seq_perturbation/perturb_bar_3/* /home/hyu/DeepACE/Figs/F01_deepace_diagram/F01c_perturb_heatmap_creme
mv /home/hyu/Digital_Platform/manuals/fig_dataset/perturb_seq_creme/gencode_tss_summary/* /home/hyu/DeepACE/Preds/D03_creme/
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

import re

BASE_DIR = os.path.abspath("./")
sys.path.append(BASE_DIR)
from functions import get_matched, open_fa

save_dir = "./Preds/D03_creme/"
celltypes = ["4824_PC-3", "5110_GM12878", "5111_K562",]
df_list = [
    "4824_PC-3_perturb_merged_alltiles",
    "5110_GM12878_perturb_merged_alltiles",
    "5111_K562_perturb_merged_alltiles",
]

feature_map = {
    "4824_PC-3": "prostate cancer cell line:PC-3",
    "5110_GM12878": "B lymphoblastoid cell line: GM12878 ENCODE, biol_",
    "5111_K562": "chronic myelogenous leukemia cell line:K562 ENCODE, biol_",
}

def extract_region_bounds(region_str):
    match = re.match(r"\[(\d+), (\d+)\)", region_str)
    if match:
        start, end = map(int, match.groups())
        return start, end
    return None, None

for i in range(len(df_list)):
    df_summary = pd.read_csv(f"{save_dir}/{celltypes[i]}/{df_list[i]}.csv")
    uni_pred = np.load(f"{save_dir}/{celltypes[i]}/{df_list[i]}/uni_pred.npy")
    uni_anno = pd.read_csv(f"{save_dir}/{celltypes[i]}/{df_list[i]}/uni_anno.csv")
    feature = feature_map[celltypes[i]]
    sel_col = uni_anno[uni_anno["celltype"] == feature].index
    sel_anno = uni_anno.loc[sel_col].reset_index(drop=True)
    sel_pred = uni_pred[:, sel_col]
    celltypes_name = celltypes[i]
    bar_dir = f"./Figs/F01_deepace_diagram/F01c_perturb_heatmap_creme/{celltypes_name}"
    os.makedirs(bar_dir, exist_ok=True)
    tile_size = 500

    df_tile = df_summary[df_summary["tile_size"] == tile_size]
    regions = df_tile["region"].unique()
    regions = sorted(regions, key=lambda x: extract_region_bounds(x)[0])
    sel_indices = df_tile.index
    alt_pred = sel_pred[sel_indices, :]

    ref_indices = []
    for sel_idx in sel_indices:
        mut_id = df_tile.loc[sel_idx, "mut_id"]
        seq_id = mut_id.split("randmut")[1].split("_")[0]
        ref_idx = df_summary[df_summary["mut_id"] == f"{celltypes_name}_ref{seq_id}"].index[0]
        ref_indices.append(ref_idx)
    ref_pred = sel_pred[ref_indices, :]

    delta_pred = alt_pred - ref_pred
    df_tile = df_tile.reset_index(drop=True)
    sel_anno = sel_anno.copy()
    sel_anno["rep_id"] = sel_anno.groupby("model").cumcount() + 1
    labels = sel_anno["model"] + " (" + sel_anno["rep_id"].astype(str) + ")"
    center_idx = len(regions) // 2
    offsets = [-2, -1, 0, 1, 2]
    for offset in offsets:
        region_idx = center_idx + offset
        if region_idx < 0 or region_idx >= len(regions):
            continue
        target_region = regions[region_idx]
        region_rows = df_tile[df_tile["region"] == target_region].index
        delta_pred_region = delta_pred[region_rows, :]
        corr_matrix = np.corrcoef(delta_pred_region.T)
        corr_df = pd.DataFrame(corr_matrix, index=labels, columns=labels)
        plt.figure(figsize=(6, 5))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        ax = sns.heatmap(
            corr_df,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            mask=mask,
            cbar_kws={"label": "Correlation", 'shrink': 0.5}
        )
        for i in range(corr_df.shape[0]):
            for j in range(corr_df.shape[1]):
                if not mask[i, j] and corr_df.iloc[i, j] <= 0:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=1.5))

        plt.title(
            f"{celltypes_name} (tile_size=5000)\nRegion offset: {offset}",
            fontsize=14
        )
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plot_path = os.path.join(
            bar_dir,
            f"{celltypes_name}_corr_heatmap_tile5000_region_{offset}.pdf"
        )
        plt.savefig(plot_path, dpi=400, bbox_inches='tight')
        plt.close()