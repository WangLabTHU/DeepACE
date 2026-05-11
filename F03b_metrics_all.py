'''
/home/hyu/Digital_Platform/manuals/fig2c_virtual_screen.py

cp /home/hyu/DeepACE/Supps/S19_robust_anchors/results_anchors/anchors_500/radar_* /home/hyu/DeepACE/Figs/F03_virtual_screen/
cp /home/hyu/Figures/DeepACE/Fig3/Fig3d* /home/hyu/DeepACE/Figs/F03_virtual_screen/
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
from scipy.ndimage import gaussian_filter1d

def load_data(cell, motif=None):
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



''' Radar Preparation for algos '''

pseudo_source = "random"
metric_type = "mahalanobis"
mode = "pca50"
n_neighbors = 500

pseudo_400, pseudo_300, pseudo_200, pseudo_100 = [], [], [], []
screen_400, screen_300, screen_200, screen_100 = [], [], [], []
posrat_400, posrat_300, posrat_200, posrat_100 = [], [], [], []
all_labels = []
for dataset in ["MPRA", "epigenetics"]: 
    if dataset == "MPRA":
        cells = ["HepG2", "K562", "SKNSH"]
    elif dataset == "epigenetics":
        cells = ["HepG2", "HepG2", "HepG2"]
        motifs = ["ELF1_1_aim", "HNF1A_1_aim", "HNF4A_1_aim"]
    
    if dataset == "MPRA":
        output_dir = "./Preds/D04_deeptfbu/pca50_epigenetics_pseudo_random_mahalanobis"
    elif dataset == "epigenetics":
        output_dir = "./Preds/D06_mpra/pca50_MPRA_pseudo_random_mahalanobis"
    
    for i, cell in enumerate(cells):
        motif = motifs[i] if dataset == "epigenetics" else None
        plot_tag = motif.split("_")[0] if dataset == "epigenetics" else cell
        df_pseudo_effect = pd.read_csv(f"{output_dir}/pseudo_effect_{plot_tag}.csv")
        df_screen_effect = pd.read_csv(f"{output_dir}/screen_effect_{plot_tag}.csv")
        df_positive_ratio = pd.read_csv(f"{output_dir}/positive_ratio_{plot_tag}.csv")
        baseline_log2 = df_screen_effect.iloc[0]["mean_remaining"]

        for topk in [400, 300, 200, 100]:
            p_val = get_values(df_pseudo_effect, topk, "prop_fin")
            p_val = np.log2(p_val + 1) 
            raw_log2 = get_values(df_screen_effect, -topk, "mean_remaining")
            r_val = get_values(df_positive_ratio, -topk, "positive_ratio_remaining")
            s_val = 2 ** (raw_log2 - baseline_log2)
            if topk == 400:
                pseudo_400.append(p_val); screen_400.append(s_val); posrat_400.append(r_val)
            elif topk == 300:
                pseudo_300.append(p_val); screen_300.append(s_val); posrat_300.append(r_val)
            elif topk == 200:
                pseudo_200.append(p_val); screen_200.append(s_val); posrat_200.append(r_val)
            else:
                pseudo_100.append(p_val); screen_100.append(s_val); posrat_100.append(r_val)
                all_labels.append(f"{dataset}_{plot_tag}")

topks = [400, 300, 200, 100]
N_points = len(all_labels)
pseudo_exp = [pseudo_400, pseudo_300, pseudo_200, pseudo_100]
screen_exp = [screen_400, screen_300, screen_200, screen_100]
posrat_exp = [posrat_400, posrat_300, posrat_200, posrat_100]
exps = [pseudo_exp, screen_exp, posrat_exp]
names = ["pseudo_effect", "screen_effect", "positive_ratio"]
controls = [1.0, 1.0, 0.2]
for exp, name, ctrl in zip(exps, names, controls):
    df_save = pd.DataFrame({
        "Label": all_labels * 4,
        "TopK": sum([[k] * N_points for k in topks], []),
        "Value": sum(exp, []),
        "Control": [ctrl] * (N_points * 4)
    })
    df_save.to_csv(f"{output_dir}/radar_{name}.csv", index=False)


''' Lineplot for algos '''

save_dir = "./Figs/F03_virtual_screen/"
file_map = {
        "Filter_Efficiency": f"./Figs/F03_virtual_screen/radar_pseudo_effect.csv",
        "Retained_Activity": f"./Figs/F03_virtual_screen/radar_screen_effect.csv"
}
for metric_name, path in file_map.items():
    if not os.path.exists(path): continue
    df = pd.read_csv(path)
    df['Category'] = df['Label'].apply(lambda x: 'MPRA' if x.startswith('MPRA_') else 'Epigenetics')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    sns.set_theme(style="ticks")
    categories = ['MPRA', 'Epigenetics']
    palette_map = {'MPRA': sns.color_palette("Blues_d", n_colors=3), 'Epigenetics': sns.color_palette("YlOrRd", n_colors=3)}
    y_min, y_max = df['Value'].min() * 0.9, df['Value'].max() * 1.1
    for i, cat in enumerate(categories):
        ax = axes[i]
        sub_df = df[df['Category'] == cat]
        sns.lineplot(data=sub_df, x='TopK', y='Value', hue='Label', marker='o', linewidth=3, markersize=10, palette=palette_map[cat], ax=ax)
        control_val = sub_df['Control'].iloc[0]
        ax.axhline(y=control_val, color='black', linestyle='--', linewidth=2, label='Control')
        ax.set_title(f'{metric_name} - {cat}', fontsize=20, pad=15)
        ax.set_xlabel('TopK samples', fontsize=20)
        ax.set_ylabel('Metric value', fontsize=20)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(450, 50)
        ax.set_xticks([400, 300, 200, 100])
        ax.tick_params(axis='both', labelsize=20, width=2, length=8)
        ax.legend(title=None, frameon=False, fontsize=18, loc='upper left')
        sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"F03b_{metric_name}_LinePlot.svg"), bbox_inches='tight')
    plt.close()