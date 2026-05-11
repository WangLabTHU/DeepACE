'''
数据集过滤, 可视化展示数据集

/home/hyu/Digital_Platform/manuals/fig2c_virtual_screen.py
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

def load_data(cell, motif=None, mode="total"):
    """Load primary and pseudo data based on dataset, cell/motif, and mode.
    
    Args:
        cell (str): Cell type (e.g., 'HepG2', 'K562', 'SKNSH', 'WTC11').
        motif (str, optional): Motif name for epigenetics dataset (e.g., 'ELF1_1_aim').
        mode (str): Processing mode ('standard', 'total', 'xfilt', 'pca50', 'pca100'). Default is 'total'.
    
    Returns:
        tuple: (primary_data, pseudo_data, labels)
    """
    # Load primary data and labels
    if dataset == "MPRA":
        primary_data = np.load(f"./Preds/D06_mpra/valids_MPRA_AdaLead_{cell}/uni_pred.npy")
        labels_df = pd.read_csv("./Datas/D06_mpra/valids.csv")
        labels_df = labels_df[labels_df["origin"] == "AdaLead"].nlargest(500, f"{cell}_prediction")
        labels = labels_df[f"{cell}_l2fc"].to_numpy()
    elif dataset == "epigenetics":
        primary_data = np.load(f"./Preds/D04_deeptfbu/valids_Epigenetics_{motif}/uni_pred.npy")
        labels_df = pd.read_excel("./Datas/D04_deeptfbu/3TF_MPRA.xlsx")
        labels_df = labels_df[labels_df['sequence_name'].str.contains(motif, na=False)]
        labels = labels_df["measured enhancer activity"].to_numpy()
        labels = np.log2(labels)
    else:
        raise ValueError("Invalid dataset input!")
    pseudo_data = np.load("./Preds/D10_random/random_sample_1/uni_pred.npy")
    combined_data = np.vstack((primary_data, pseudo_data)) if len(pseudo_data) > 0 else primary_data
    anno_df = pd.read_csv(f"./total_features.csv")
    match_tag = "SK-N-SH" if cell == "SKNSH" else cell
    uni_selected = PCA(n_components=50, random_state=42).fit_transform(combined_data)
    primary_data = uni_selected[:-len(pseudo_data)] if len(pseudo_data) > 0 else uni_selected
    pseudo_data = uni_selected[-len(pseudo_data):] if len(pseudo_data) > 0 else np.array([])
    return primary_data, pseudo_data, labels

def preprocess_data(primary_data, pseudo_data, labels):
    """Scale data and categorize into positive, negative, and mid groups."""
    combined_data = np.vstack((primary_data, pseudo_data)) if len(pseudo_data) > 0 else primary_data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)
    # Separate scaled data
    scaled_primary = scaled_data[:-len(pseudo_data)] if len(pseudo_data) > 0 else scaled_data
    scaled_pseudo = scaled_data[-len(pseudo_data):] if len(pseudo_data) > 0 else np.array([])
    # Categorize labels
    n_total = len(labels)
    n_top = int(n_total * 0.2)
    indices = np.argsort(labels)
    neg_data = scaled_primary[indices[:n_top]]
    pos_data = scaled_primary[indices[-n_top:]]
    mid_data = scaled_primary[indices[n_top:-n_top]]
    sorted_labels = np.concatenate([labels[indices[:n_top]], labels[indices[-n_top:]], labels[indices[n_top:-n_top]]])
    # Combine samples and create labels
    sample_data = np.vstack((neg_data, pos_data, mid_data, scaled_pseudo)) if len(pseudo_data) > 0 else np.vstack((neg_data, pos_data, mid_data))
    sample_labels = (['Negative'] * len(neg_data) + ['Positive'] * len(pos_data) + 
                     ['Mid'] * len(mid_data) + ['Pseudo'] * len(scaled_pseudo))
    return sample_data, sample_labels, sorted_labels



def plot_umap(sample_data, sample_labels, sorted_labels, plot_tag, output_dir=None):
    pseudo_mask = np.array(sample_labels) == 'Pseudo'
    umap = UMAP(n_components=2, random_state=42)
    embedding = umap.fit_transform(sample_data)
    df_plot = pd.DataFrame({
        'Dim1': embedding[:, 0],
        'Dim2': embedding[:, 1],
        'Group': sample_labels
    })
    pseudo_mask_df = df_plot['Group'] == 'Pseudo'
    real_mask = ~pseudo_mask_df
    df_plot.loc[real_mask, 'Expression'] = sorted_labels
    df_plot.loc[pseudo_mask_df, 'Expression'] = np.nan
    plt.figure(figsize=(8, 6))
    xy = df_plot.loc[real_mask, ['Dim1', 'Dim2']].values.T
    weights = df_plot.loc[real_mask, 'Expression'].values
    expr_raw = df_plot.loc[real_mask, 'Expression'].values
    expr_median = np.median(expr_raw)
    expr_mean = np.mean(expr_raw)
    norm = TwoSlopeNorm(vmin=expr_raw.min(), vcenter=expr_median, vmax=expr_raw.max())
    weights = weights - weights.min() + 1e-6  
    if len(weights) > 5:
        kde = gaussian_kde(xy, weights=weights, bw_method=0.15)
        xgrid = np.linspace(df_plot['Dim1'].min(), df_plot['Dim1'].max(), 400)
        ygrid = np.linspace(df_plot['Dim2'].min(), df_plot['Dim2'].max(), 400)
        X, Y = np.meshgrid(xgrid, ygrid)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        Z_masked = np.ma.masked_where(Z < 0.001 * Z.max(), Z)  
        Z_expr = Z_masked / Z_masked.max() * (expr_raw.max() - expr_raw.min()) + expr_raw.min()
        contour_real = plt.contourf(X, Y, Z_expr, levels=20, cmap='Greens', alpha=0.7, norm=norm)
    plt.scatter(
        df_plot.loc[pseudo_mask_df, 'Dim1'],
        df_plot.loc[pseudo_mask_df, 'Dim2'],
        color='k',
        s=2,
        alpha=0.9,
        edgecolor="None",
        label='Pseudo'
    )
    cbar = plt.colorbar(contour_real)
    cbar.set_label("Expression-weighted KDE", fontsize=12)
    plt.title(f"UMAP (fit on Pseudo, contour real) - {plot_tag}")
    plt.xlabel('UMAP Dim1')
    plt.ylabel('UMAP Dim2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/proj_umap_{plot_tag}.pdf", dpi=400, bbox_inches='tight')
    # df_plot.to_csv(f"{output_dir}/umap2d_{plot_tag}.csv")
    plt.close()



''' UMAP Analysis '''

pseudo_source = "random"
mode = "pca50"
metric_type = "mahalanobis"

for dataset in ["MPRA", "epigenetics"]: #   
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
        output_dir = "./Supps/S12_filter_umap"
        primary_data, pseudo_data, labels = load_data(cell, motif, mode)
        sample_data, sample_labels, sorted_labels = preprocess_data(primary_data, pseudo_data, labels)
        plot_umap(sample_data, sample_labels, sorted_labels, plot_tag, output_dir=output_dir)