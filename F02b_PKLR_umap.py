'''
/home/hyu/Digital_Platform/manuals/fig1e_mutational_effects_umap.py

cp /home/hyu/Digital_Platform/manuals/fig1e_mutational_effects/umap2d_PKLR.pdf /home/hyu/DeepACE/Figs/F02_variant_effects/F02b_PKLR_umap.pdf
cp /home/hyu/Digital_Platform/manuals/fig1e_mutational_effects/umap2d_PKLR.csv /home/hyu/DeepACE/Figs/F02_variant_effects/F02b_PKLR_umap.csv
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


def plot_umap(sample_data, sample_labels, sorted_labels, plot_tag, idx_min, idx_max,
              output_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    pseudo_mask = np.array(sample_labels) == 'Pseudo'
    rand_mask = np.array(sample_labels) == 'Rand'
    real_mask = (~pseudo_mask) & (~rand_mask)
    pseudo_point = sample_data[pseudo_mask].mean(axis=0, keepdims=True)
    real_data = sample_data[real_mask]
    rand_point = sample_data[rand_mask]
    sample_data_reduced = np.vstack([pseudo_point, real_data, rand_point])
    sample_labels_reduced = ['Pseudo'] + ['Real'] * len(real_data) + ['Rand'] * len(rand_point)
    umap_model = UMAP(n_components=2, random_state=42)
    embedding = umap_model.fit_transform(sample_data_reduced)
    df_plot = pd.DataFrame({
        'Dim1': embedding[:, 0],
        'Dim2': embedding[:, 1],
        'Group': sample_labels_reduced
    })
    df_plot['Expression'] = np.nan
    df_plot.loc[df_plot['Group'] == 'Real', 'Expression'] = sorted_labels
    plt.figure(figsize=(5, 4))
    real_mask_df = df_plot['Group'] == 'Real'
    xy = df_plot.loc[real_mask_df, ['Dim1', 'Dim2']].values.T
    weights = df_plot.loc[real_mask_df, 'Expression'].values
    expr_raw = df_plot.loc[real_mask_df, 'Expression'].values
    expr_median = np.median(expr_raw)
    expr_mean = np.mean(expr_raw)
    norm = TwoSlopeNorm(vmin=expr_raw.min(), vcenter=expr_median, vmax=expr_raw.max())
    weights = weights - np.nanmin(weights) + 1e-6
    if len(weights) > 5:
        kde = gaussian_kde(xy, weights=weights, bw_method=0.15)
        xgrid = np.linspace(df_plot['Dim1'].min() - 1, df_plot['Dim1'].max() + 1, 400)
        ygrid = np.linspace(df_plot['Dim2'].min() - 1, df_plot['Dim2'].max() + 1, 400)
        X, Y = np.meshgrid(xgrid, ygrid)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)        
        Z_masked = np.ma.masked_where(Z < 0.001 * Z.max(), Z)  
        Z_expr = Z_masked / Z_masked.max() * (expr_raw.max() - expr_raw.min()) + expr_raw.min()
        contour_real = plt.contourf(X, Y, Z_expr, levels=20, cmap='Greens', alpha=0.7, norm=norm)
    plt.scatter(
        df_plot.loc[df_plot['Group'] == 'Pseudo', 'Dim1'],
        df_plot.loc[df_plot['Group'] == 'Pseudo', 'Dim2'],
        color='#249875', s=50, alpha=0.9, edgecolor="white", label='Backbone'
    )
    plt.scatter(
        df_plot.loc[df_plot['Group'] == 'Rand', 'Dim1'],
        df_plot.loc[df_plot['Group'] == 'Rand', 'Dim2'],
        color='k', s=2, alpha=0.9, edgecolor="None", label='Pseudo'
    )
    idx_min = np.array(idx_min)
    plt.scatter(
        df_plot.loc[idx_min + 1, 'Dim1'],
        df_plot.loc[idx_min + 1, 'Dim2'],
        color='blue', s=50, alpha=0.9, edgecolor="white", label='Min ΔlogFC'
    )
    idx_max = np.array(idx_max)
    plt.scatter(
        df_plot.loc[idx_max + 1, 'Dim1'],
        df_plot.loc[idx_max + 1, 'Dim2'],
        color='red', s=50, alpha=0.9, edgecolor="white", label='Max ΔlogFC'
    )
    plt.xlabel('UMAP Dim1', fontsize=16)
    plt.ylabel('UMAP Dim2', fontsize=16)
    plt.legend(loc='lower left', fontsize=12)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cbar = plt.colorbar(contour_real, shrink=0.8)
    cbar.ax.tick_params(labelsize=16)
    # new_ticks = [2.4, 1.2, 0.0, -1.2, -2.4]
    # cbar.set_ticks(new_ticks)
    # cbar.set_ticklabels([f'{t:.1f}' for t in new_ticks])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/F02b_PKLR_umap.pdf", dpi=400, bbox_inches='tight')
    # df_plot.to_csv(f"{output_dir}/proj_umap_{plot_tag}.csv", index=False)
    plt.close()


''' Single Dataset UMAP '''

for motif in ["PKLR"]: 
    uni_path = f"./Preds/D05_mprabase/point_MPRABase_{motif}_saturation/uni_pred.npy"
    df_path = f"./Preds/D05_mprabase/point_MPRABase_{motif}_saturation.tsv"
    uni_list = np.load(uni_path)
    df = pd.read_csv(df_path, sep="\t")
    variant_effects = df['VariantExpressionEffect (log2)'].to_numpy()
    df_min = df.nsmallest(20, "VariantExpressionEffect (log2)")
    idx_min = list(df_min.index)
    df_max = df.nlargest(20, "VariantExpressionEffect (log2)")
    idx_max = list(df_max.index)
    pred_alt = uni_list[:-1]
    pred_ref = np.repeat(uni_list[-1][np.newaxis, :], len(pred_alt), axis=0)
    valid_mask = np.isfinite(pred_alt).any(axis=0) & np.isfinite(pred_ref).any(axis=0)
    pred_alt = pred_alt[:, valid_mask]
    pred_ref = pred_ref[:, valid_mask]
    pred_rand = np.load(f"./Preds/D05_mprabase/point_MPRABase_{motif}_randaug/uni_pred.npy")[:-1]
    pred_rand = pred_rand[:, valid_mask]
    
    combined = np.vstack([pred_alt, pred_ref, pred_rand])
    combined_pca = PCA(n_components=50, random_state=42).fit_transform(combined)
    pred_alt_pca = combined_pca[:len(pred_alt)] # (1407, 50)
    pred_ref_pca = combined_pca[len(pred_alt):-len(pred_rand)] # (1407, 50)
    pred_rand_pca = combined_pca[-len(pred_rand):] # (500, 50)
    output_dir = "./Figs/F02_variant_effects"
    sample_data = np.vstack([pred_ref_pca, pred_alt_pca, pred_rand_pca])
    sample_labels = ['Pseudo'] * len(pred_ref_pca) + ['Real'] * len(pred_alt_pca) + ['Rand'] * len(pred_rand_pca)
    plot_umap(sample_data, sample_labels, variant_effects, plot_tag=motif, output_dir=output_dir, 
              idx_min = idx_min, idx_max = idx_max)