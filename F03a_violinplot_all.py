'''
/home/hyu/Digital_Platform/manuals/fig2c_virtual_screen.py

cp /home/hyu/2_Basset/hg19/hg19.fa /home/hyu/DeepACE/Datas/D02_grch

mv /home/hyu/Digital_Platform/manuals/fig_dataset/valids_MPRA_AdaLead_* /home/hyu/DeepACE/Preds/D06_mpra/
cp /home/hyu/Digital_Platform_Dataset/Malinois/valids.csv /home/hyu/DeepACE/Datas/D06_mpra

mv /home/hyu/Digital_Platform/manuals/fig_dataset/valids_Epigenetics_* /home/hyu/DeepACE/Preds/D04_deeptfbu/
cp /home/hyu/Digital_Platform_Dataset/DeepTFBU/3TF_MPRA.xlsx /home/hyu/DeepACE/Datas/D04_deeptfbu

cp -r /home/hyu/Digital_Platform/manuals/fig2c_virtual_screen/pca50_epigenetics_pseudo_random_mahalanobis /home/hyu/DeepACE/Preds/D04_deeptfbu/
cp -r /home/hyu/Digital_Platform/manuals/fig2c_virtual_screen/pca50_MPRA_pseudo_random_mahalanobis /home/hyu/DeepACE/Preds/D06_mpra/

cp /home/hyu/Figures/DeepACE/Fig3/Fig3c_* /home/hyu/DeepACE/Figs/F03_virtual_screen/
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import TruncatedSVD
from scipy.stats import pearsonr, spearmanr
import random
import os, sys
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.manifold import MDS
from scipy.spatial.distance import cdist
from numpy.linalg import inv
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import mahalanobis
from pyfaidx import Fasta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score

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



def analyze_pseudo_similarity(sample_data, sample_labels, labels, plot_tag, n_neighbors=500, 
                              metric="cosine", output_dir="./Figs/F03_virtual_screen"):

    pseudo_mask = np.array([g == 'Pseudo' for g in sample_labels])
    real_idx = np.where(~pseudo_mask)[0]
    pseudo_idx = np.where(pseudo_mask)[0]
    real_vectors = sample_data[real_idx]
    pseudo_vectors = sample_data[pseudo_idx]
    n_neighbors = min(n_neighbors, len(pseudo_idx))
    
    # distance calculation
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
    
    # 1. Proportion analysis
    order = np.argsort(-pseudo_similarity)
    neg_flags = np.array([sample_labels[i] == 'Negative' for i in real_idx])
    pos_flags = np.array([sample_labels[i] == 'Positive' for i in real_idx])

    # Compute cumulative sums and proportions
    cum_neg = np.cumsum(neg_flags[order])
    cum_pos = np.cumsum(pos_flags[order])
    total_samples = np.arange(1, len(order) + 1)
    prop_fin = (cum_neg + 1) / (cum_pos + 1)
    pd.DataFrame({
        "order": np.arange(1, len(order) + 1),
        "cum_neg": cum_neg,
        "cum_pos": cum_pos,
        "pseudo_similarity": pseudo_similarity[order],
        "prop_fin": prop_fin
    }).to_csv(f"{output_dir}/pseudo_effect_{plot_tag}.csv", index=False)
    
    # 2. Mean expression after removal
    cut_size = 100
    mean_remaining = []
    std_remaining = []
    for k in range(1, len(order) + 1):
        remaining_idx = order[k:]
        if len(remaining_idx) > 0:
            mean_val = labels[remaining_idx].mean()
            std_val = labels[remaining_idx].std()
        else:
            mean_val = np.nan
            std_val = np.nan
        mean_remaining.append(mean_val)
        std_remaining.append(std_val)
    pd.DataFrame({
        "removed_top_n": np.arange(1, len(order) + 1),
        "mean_remaining": mean_remaining,
        "std_remaining": std_remaining 
    }).to_csv(f"{output_dir}/screen_effect_{plot_tag}.csv", index=False)
    
    # 3. Pseudo similarity vs expression
    expression = labels[order]    
    pcc, _ = pearsonr(distances.mean(axis=1), expression)
    group_size = 100
    distances_sorted = distances.mean(axis=1)[order]
    groups = []
    group_labels = []
    medians = []
    for i in range(0, len(distances_sorted), group_size):
        end_idx = min(i + group_size, len(distances_sorted))
        group_expr = expression[i:end_idx]
        groups.append(group_expr)
        group_labels.append(f"{i+1}-{end_idx}")
        medians.append(np.median(group_expr) if len(group_expr) > 0 else np.nan)  
    df_violin = pd.DataFrame({
        'Expression': np.concatenate(groups),
        'Group': np.repeat(group_labels, [len(g) for g in groups])
    })
    df_violin.to_csv(f"{output_dir}/scatter_distance_expr_{plot_tag}.csv", index=False)
    
    
    # 4. Positive ratio among remaining samples
    cut_size = 100
    pos_ratio_remaining = []
    for k in range(1, len(order) + 1):
        remaining_idx = order[k:]
        if len(remaining_idx) > 0:
            pos_ratio = np.mean(pos_flags[remaining_idx]) 
        else:
            pos_ratio = np.nan
        pos_ratio_remaining.append(pos_ratio)
    pd.DataFrame({
        "removed_top_n": np.arange(1, len(order) + 1),
        "positive_ratio_remaining": pos_ratio_remaining
    }).to_csv(f"{output_dir}/positive_ratio_{plot_tag}.csv", index=False)
    
'''
Dataset Preparation
'''

pseudo_source = "random"
mode = "pca50"
metric_type = "mahalanobis"

for dataset in ["MPRA", "epigenetics"]: 
    if dataset == "MPRA":
        cells = ["HepG2", "K562", "SKNSH"]
    elif dataset == "epigenetics":
        cells = ["HepG2", "HepG2", "HepG2"]
        motifs = ["ELF1_1_aim", "HNF1A_1_aim", "HNF4A_1_aim"]
    else:
        raise ValueError("Invalid dataset input!")
    
    if dataset == "MPRA":
        output_dir = "./Preds/D04_deeptfbu/pca50_epigenetics_pseudo_random_mahalanobis"
    elif dataset == "epigenetics":
        output_dir = "./Preds/D06_mpra/pca50_MPRA_pseudo_random_mahalanobis"
    
    for i, cell in enumerate(cells):
        motif = motifs[i] if dataset == "epigenetics" else None
        plot_tag = motif.split("_")[0] if dataset == "epigenetics" else cell
        print(f"Start Processing, dataset = {dataset}, cell type / motif = {plot_tag}, processing mode = {mode}, metric type = {metric_type}, pseudo source = {pseudo_source}")
        primary_data, pseudo_data, labels = load_data(cell, motif)
        sample_data, sample_labels, sorted_labels = preprocess_data(primary_data, pseudo_data, labels)
        analyze_pseudo_similarity(sample_data, sample_labels, sorted_labels, plot_tag, metric=metric_type, output_dir=output_dir)

'''
Plotting Figures
'''


def Fig3c_plot_mpra_violin():
    files = {
        "HepG2": "./Preds/D06_mpra/pca50_MPRA_pseudo_random_mahalanobis/scatter_distance_expr_HepG2.csv",
        "K562": "./Preds/D06_mpra/pca50_MPRA_pseudo_random_mahalanobis/scatter_distance_expr_K562.csv",
        "SKNSH": "./Preds/D06_mpra/pca50_MPRA_pseudo_random_mahalanobis/scatter_distance_expr_SKNSH.csv"
        "ELF1": "./Preds/D04_deeptfbu/pca50_epigenetics_pseudo_random_mahalanobis/scatter_distance_expr_ELF1.csv",
        "HNF1A": "./Preds/D04_deeptfbu/pca50_epigenetics_pseudo_random_mahalanobis/scatter_distance_expr_HNF1A.csv",
        "HNF4A": "./Preds/D04_deeptfbu/pca50_epigenetics_pseudo_random_mahalanobis/scatter_distance_expr_HNF4A.csv"
    }
    save_dir = "./Figs/F03_virtual_screen"
    group_order = ['1-100', '101-200', '201-300', '301-400', '401-500']
    for cell_line, path in files.items():
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        df['Group'] = pd.Categorical(df['Group'], categories=group_order, ordered=True)
        first_vals = df.loc[df['Group'] == group_order[0], 'Expression'].dropna()
        last_vals = df.loc[df['Group'] == group_order[-1], 'Expression'].dropna()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.set_theme(style="ticks")
        ax = sns.violinplot(data=df, x='Group', y='Expression', palette="Greens", inner=None, alpha=0.7, ax=ax)
        sns.boxplot(data=df, x='Group', y='Expression', width=0.12, color='white', linewidth=2, showfliers=False, ax=ax)
        medians = df.groupby('Group')['Expression'].median()
        ax.plot(range(len(group_order)), medians, color='k', marker='o', linestyle='--', linewidth=2.5, markersize=8)
        ax.set_title(cell_line, fontsize=20, pad=10)
        ax.set_xticklabels(group_order, rotation=45, horizontalalignment='right', fontsize=20)
        ax.set_xlabel('Distance to random anchors', fontsize=20, labelpad=12)
        ax.set_ylabel('Expression value', fontsize=20, labelpad=12)
        ax.set_ylim(-3, 2)
        ax.set_yticks([-2, -1, 0, 1])
        ax.tick_params(axis='both', labelsize=20)
        sns.despine()
        plt.savefig(os.path.join(save_dir, f"Fig3a_Trend_{cell_line}.svg"), bbox_inches='tight')
        plt.close()