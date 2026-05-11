'''
perturbation analysis of CREME dataset

/home/hyu/Digital_Platform/manuals/figs9_seq_perturbation_2.py
/home/hyu/Digital_Platform/manuals
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


def open_fa(file):
    record = []
    f = open(file,'r')
    for item in f:
        if '>' not in item:
            record.append(item[0:-1])
    f.close()
    return record

def perturb_seq(ref_seq, start, end, rep=100):
    """Randomly mutate region [start, end) in ref_seq."""
    if start < 0 or end > len(ref_seq) or start >= end:
        raise ValueError("Invalid start or end for perturbation region.")
    alt_seqs = []
    for _ in range(rep):
        rand_region = ''.join(random.choice('ATCG') for _ in range(end - start))
        alt_seqs.append(ref_seq[:start] + rand_region + ref_seq[end:])
    return alt_seqs

def write_fasta(path, seqs, tags):
    """Write sequences and headers to FASTA."""
    with open(path, "w") as f:
        for tag, seq in zip(tags, seqs):
            f.write(f">{tag}\n{seq}\n")
    print(f"[✓] FASTA written to: {path}")

def extract_region_bounds(region_str):
    match = re.match(r"\[(\d+), (\d+)\)", region_str)
    if match:
        start, end = map(int, match.groups())
        return start, end
    return None, None


''' Dataset Preparation '''

# seq_len = 196608
# tiles = [100, 500, 5000]
# rep = 1
# for celltype in ["5111_K562", "5110_GM12878", "4824_PC-3"]:
#     out_dir = f"./Preds/D03_creme/{celltype}"
#     os.makedirs(out_dir, exist_ok=True)
#     print(f"\nProcessing {celltype} ...")
#     fa_path = f"./Preds/D03_creme/{celltype}_selected_genes.fa"
#     ori_seqs = open_fa(fa_path)
#     ref_seqs = ori_seqs[0:20]
#     merged_alt_seqs, merged_tags, merged_records = [], [], []
#     for i, ref_seq in enumerate(ref_seqs):
#         for tile in tiles:
#             print(f"Generating tile={tile} ...")
#             start_positions = range(seq_len // 2 - tile * 10, seq_len // 2 + tile * 10 + 1, tile)
#             for start in tqdm(start_positions, desc=f"Tile {tile}", leave=False):
#                 end = start + tile
#                 mutated = perturb_seq(ref_seq, start, end, rep=rep)
#                 for r_idx, mut_seq in enumerate(mutated, start=1):
#                     tag = f"{celltype}_randmut{i}_{start}_{end}_tile{tile}_rep{r_idx}"
#                     merged_alt_seqs.append(mut_seq)
#                     merged_tags.append(tag)
#                     merged_records.append({ "celltype": celltype, "mut_id": tag, "region": f"[{start}, {end})", "tile_size": tile, "rep_id": r_idx})
#         ref_tag = f"{celltype}_ref{i}"
#         merged_alt_seqs.append(ref_seq)
#         merged_tags.append(ref_tag)
#         merged_records.append({ "celltype": celltype, "mut_id": ref_tag, "region": "ref", "tile_size": "NA", "rep_id": "NA"})
#     merged_fasta_path = os.path.join(out_dir, f"{celltype}_perturb_merged_alltiles.fa")
#     merged_csv_path = os.path.join(out_dir, f"{celltype}_perturb_merged_alltiles.csv")
#     write_fasta(merged_fasta_path, merged_alt_seqs, merged_tags)
#     pd.DataFrame(merged_records).to_csv(merged_csv_path, index=False)


''' Perturbation Barplot '''

load_dir = "./Preds/D03_creme"
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

for i in range(len(df_list)):
    df_summary = pd.read_csv(f"{load_dir}/{celltypes[i]}/{df_list[i]}.csv")
    uni_pred = np.load(f"{load_dir}/{celltypes[i]}/{df_list[i]}/uni_pred.npy")
    uni_anno = pd.read_csv(f"{load_dir}/{celltypes[i]}/{df_list[i]}/uni_anno.csv")
    feature = feature_map[celltypes[i]]
    sel_col = uni_anno[uni_anno["celltype"] == feature].index
    sel_anno = uni_anno.loc[sel_col].reset_index(drop=True)
    sel_anno['celltype'] = (
        sel_anno.groupby('model')['celltype']
        .transform(lambda x: x + (x.groupby(x).cumcount() + 1).astype(str))
    )
    sel_pred = uni_pred[:, sel_col]
    celltypes_name = celltypes[i]
    save_dir = f"./Supps/S02_perturb_barplot_creme/"
    tile_sizes = [100, 500, 5000]
    corr_matrix_all_tiles = np.zeros((len(tile_sizes), 21))
    corr_std_all_tiles = np.zeros((len(tile_sizes), 21))
    for idx, tile_size in enumerate(tile_sizes):
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
        for j, region in enumerate(regions):
            region_rows = df_tile[df_tile["region"] == region].index
            delta_pred_region = delta_pred[region_rows, :] # (20, 4) = (seq_num, channels)
            corr_matrix = np.corrcoef(delta_pred_region.T)
            n = corr_matrix.shape[0]
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            vals = corr_matrix[mask]
            avg_corr = vals.mean()
            std_corr = vals.std()
            corr_matrix_all_tiles[idx, j] = avg_corr
            corr_std_all_tiles[idx, j] = std_corr
    tile_sizes = [int(item) for item in tile_sizes]
    corr_data = pd.DataFrame(corr_matrix_all_tiles, index=tile_sizes, columns=[f"Region {i}" for i in range(len(regions))])
    corr_std_data = pd.DataFrame(corr_std_all_tiles, index=tile_sizes, columns=corr_data.columns)
    num_regions = corr_data.shape[1]
    region_labels = [f"{k}" for k in range(-10, 11)]
    corr_data.columns = region_labels
    selected_sizes = [100, 500, 5000]
    plot_data = corr_data.loc[selected_sizes]    
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True, sharey=True,
                         gridspec_kw={'hspace': 0.3})
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    x_positions = np.arange(len(region_labels))
    for idx, (ax, tile_size) in enumerate(zip(axes, selected_sizes)):
        values = corr_data.loc[tile_size].values
        errors = corr_std_data.loc[tile_size].values
        ax.bar(x_positions, values, width=0.8, color=colors[idx], 
            edgecolor='black', linewidth=1.0, alpha=0.95,
            yerr=errors, capsize=4, ecolor='black')
        ax.set_title(f'tile_size = {tile_size}', fontsize=16, pad=12, fontweight='bold')
        ax.set_ylim(-1.05, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.tick_params(axis='y', labelsize=16)
        if idx == 1:
            ax.set_ylabel('Average Correlation', fontsize=16)
        else:
            ax.set_ylabel('')
    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(region_labels, ha='right', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(save_dir, f"feat_corr_barplot_{celltypes_name}.pdf")
    plt.savefig(plot_path, dpi=400, bbox_inches='tight')
    print(f"Three normal bar plots saved to: {plot_path}")





