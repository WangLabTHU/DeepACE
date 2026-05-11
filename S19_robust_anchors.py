'''
分析鲁棒性

/home/hyu/Digital_Platform/manuals/figs12_random_sample_generation_radar.py

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

BASE_DIR = os.path.abspath("/home/hyu/Digital_Platform")
sys.path.append(BASE_DIR)
from functions import get_matched, open_fa


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
    
    pseudo_data_1 = np.load("/home/hyu/Digital_Platform/manuals/fig_dataset/random_sample_1/uni_pred.npy")
    pseudo_data_2 = np.load("/home/hyu/Digital_Platform/manuals/fig_dataset/random_sample_2/uni_pred.npy")
    pseudo_data_3 = np.load("/home/hyu/Digital_Platform/manuals/fig_dataset/random_sample_3/uni_pred.npy")
    pseudo_data_4 = np.load("/home/hyu/Digital_Platform/manuals/fig_dataset/random_sample_4/uni_pred.npy")
    pseudo_data = np.vstack((pseudo_data_1, pseudo_data_2, pseudo_data_3, pseudo_data_4)) 
    
    combined_data = np.vstack((primary_data, pseudo_data)) if len(pseudo_data) > 0 else primary_data
    anno_df = pd.read_csv(f"/home/hyu/Digital_Platform/modals/total_features.csv")
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
    scaled_primary = scaled_data[:-len(pseudo_data)] if len(pseudo_data) > 0 else scaled_data
    scaled_pseudo = scaled_data[-len(pseudo_data):] if len(pseudo_data) > 0 else np.array([])
    n_total = len(labels)
    n_top = int(n_total * 0.2)
    indices = np.argsort(labels)
    neg_data = scaled_primary[indices[:n_top]]
    pos_data = scaled_primary[indices[-n_top:]]
    mid_data = scaled_primary[indices[n_top:-n_top]]
    sorted_labels = np.concatenate([labels[indices[:n_top]], labels[indices[-n_top:]], labels[indices[n_top:-n_top]]])
    sample_data = np.vstack((neg_data, pos_data, mid_data, scaled_pseudo)) if len(pseudo_data) > 0 else np.vstack((neg_data, pos_data, mid_data))
    sample_labels = (['Negative'] * len(neg_data) + ['Positive'] * len(pos_data) + 
                     ['Mid'] * len(mid_data) + ['Pseudo'] * len(scaled_pseudo))
    return sample_data, sample_labels, sorted_labels

def analyze_pseudo_similarity(sample_data, sample_labels, labels, plot_tag, n_neighbors=500, metric="cosine", output_dir=None):
    """Analyze pseudo-sample similarity and its effect on expression."""
    pseudo_mask = np.array([g == 'Pseudo' for g in sample_labels])
    real_idx = np.where(~pseudo_mask)[0]
    pseudo_idx = np.where(pseudo_mask)[0]
    real_vectors = sample_data[real_idx]
    pseudo_vectors = sample_data[pseudo_idx]
    n_neighbors = min(n_neighbors, len(pseudo_idx))
    
    # Nearest neighbors analysis
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
    cum_neg = np.cumsum(neg_flags[order])
    cum_pos = np.cumsum(pos_flags[order])
    total_samples = np.arange(1, len(order) + 1)
    prop_fin = (cum_neg + 1) / (cum_pos + 1)
    kernel_size = 5
    prop_fin_smooth = gaussian_filter1d(prop_fin, sigma=kernel_size, mode='nearest')
    # plt.figure(figsize=(8, 6))
    # plt.plot(np.arange(1, len(order) + 1), prop_fin_smooth, label="Negative / Positive Proportion", color='red', linewidth=2)
    # plt.axhline(y=1, color='black', linestyle='--', linewidth=1)
    # plt.xlabel("Number of Real Samples (sorted by pseudo similarity)")
    # plt.ylabel("Proportion of Samples")
    # plt.title(f"Proportion of Negative/Positive Samples ({plot_tag})")
    # plt.grid(True, linestyle="--", alpha=0.5)
    # plt.legend()
    # plt.savefig(f"{output_dir}/pseudo_effect_{plot_tag}.pdf", dpi=400, bbox_inches="tight")
    # plt.close()
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
    kernel_size = 5
    mean_remaining_smooth = gaussian_filter1d(mean_remaining, sigma=kernel_size, mode='nearest')
    std_remaining_smooth = gaussian_filter1d(std_remaining, sigma=kernel_size, mode='nearest')
    # plt.figure(figsize=(8, 6))
    # plt.plot(np.arange(0, len(order) - cut_size + 1), mean_remaining_smooth[:-cut_size+1], 
    #          color='purple', linewidth=2, label='Mean Expression')
    # plt.xlabel("Number of Real Samples Removed")
    # plt.ylabel("Mean Expression of Remaining Samples")
    # plt.title(f"Mean Expression after Removing Top Samples ({plot_tag})")
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.legend()
    # plt.savefig(f"{output_dir}/screen_effect_{plot_tag}.pdf", dpi=400, bbox_inches='tight')
    # plt.close()
    pd.DataFrame({
        "removed_top_n": np.arange(1, len(order) + 1),
        "mean_remaining": mean_remaining,
        "std_remaining": std_remaining  # 保存原始标准差
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
    # plt.figure(figsize=(10, 6))
    # sns.violinplot(x='Group', y='Expression', hue='Group', data=df_violin, palette='RdBu_r', inner='quartile', legend=False)
    # plt.axhline(y=np.mean(expression), color='black', linestyle='--', linewidth=1, label='Global Mean')
    # valid_mask = ~np.isnan(medians)  # 过滤无效中位数
    # plt.plot(np.arange(len(group_labels))[valid_mask], np.array(medians)[valid_mask], 
    #          color='red', linewidth=2, marker='o', label='Median Expression')
    # for i, (x, y) in enumerate(zip(np.arange(len(group_labels))[valid_mask], np.array(medians)[valid_mask])):
    #     plt.text(x, y, f'{y:.2f}', fontsize=8, ha='center', va='bottom', color='red',
    #              bbox=dict(facecolor='white', edgecolor='white', alpha=1.0, boxstyle='round,pad=0.2'))
    # plt.xlabel("Sample Groups (sorted by increasing distance to pseudo)")
    # plt.ylabel("Measured Expression (2^L2FC)")
    # plt.title(f"Violin Plot of Expression by Distance Group ({plot_tag})")
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.legend()
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/scatter_distance_expr_{plot_tag}.pdf", dpi=400, bbox_inches='tight')
    # plt.close()
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
    kernel_size = 5
    pos_ratio_smooth = gaussian_filter1d(pos_ratio_remaining, sigma=kernel_size, mode='nearest')
    # plt.figure(figsize=(8, 6))
    # plt.plot(np.arange(0, len(order) - cut_size + 1), pos_ratio_smooth[:-cut_size+1], 
    #          color='green', linewidth=2, label='Positive Ratio')
    # plt.xlabel("Number of Real Samples Removed")
    # plt.ylabel("Positive Sample Ratio (Remaining)")
    # plt.title(f"Positive Ratio in Remaining Samples ({plot_tag})")
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.axhline(y=0.2, color='black', linestyle='--', linewidth=1)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/positive_ratio_{plot_tag}.pdf", dpi=400, bbox_inches='tight')
    # plt.close()
    pd.DataFrame({
        "removed_top_n": np.arange(1, len(order) + 1),
        "positive_ratio_remaining": pos_ratio_remaining
    }).to_csv(f"{output_dir}/positive_ratio_{plot_tag}.csv", index=False)
    
def get_values(df, idx, column_name):
    return df.iloc[idx][column_name]


''' Dataset Preparation for anchors '''

pseudo_source = "random"
metric_type = "mahalanobis"
mode = "pca50"
output_base = "./Supps/S19_robust_anchors/results_anchors"

for dataset in ["MPRA", "epigenetics"]: 
    if dataset == "MPRA":
        cells = ["HepG2", "K562", "SKNSH"]
    elif dataset == "epigenetics":
        cells = ["HepG2", "HepG2", "HepG2"]
        motifs = ["ELF1_1_aim", "HNF1A_1_aim", "HNF4A_1_aim"]
    else:
        raise ValueError("Invalid dataset input!")
    for n_neighbors in [10, 20, 50, 100, 200, 500, 1000, 2000]:
            for i, cell in enumerate(cells):
                motif = motifs[i] if dataset == "epigenetics" else None
                plot_tag = motif.split("_")[0] if dataset == "epigenetics" else cell
                print(f"Start Processing, dataset = {dataset}, cell type / motif = {plot_tag}, processing mode = {mode}, metric type = {metric_type}, pseudo source = {pseudo_source}")
                # Load and preprocess data
                primary_data, pseudo_data, labels = load_data(cell, motif)
                sample_data, sample_labels, sorted_labels = preprocess_data(primary_data, pseudo_data, labels)
                # Analyze pseudo-sample similarity
                output_dir = f"{output_base}/anchors_{n_neighbors}"
                os.makedirs(output_dir, exist_ok=True)
                analyze_pseudo_similarity(sample_data, sample_labels, sorted_labels, plot_tag, metric=metric_type, output_dir=output_dir, n_neighbors=n_neighbors)



''' Radar Preparation for algos '''

pseudo_source = "random"
metric_type = "mahalanobis"
mode = "pca50"

for n_neighbors in [10, 20, 50, 100, 200, 500, 1000, 2000]: 
    pseudo_400, pseudo_300, pseudo_200, pseudo_100 = [], [], [], []
    screen_400, screen_300, screen_200, screen_100 = [], [], [], []
    posrat_400, posrat_300, posrat_200, posrat_100 = [], [], [], []
    all_labels = []
    output_dir = f"./Supps/S19_robust_anchors/results_anchors/anchors_{n_neighbors}/"
    for dataset in ["MPRA", "epigenetics"]: 
        if dataset == "MPRA":
            cells = ["HepG2", "K562", "SKNSH"]
        elif dataset == "epigenetics":
            cells = ["HepG2", "HepG2", "HepG2"]
            motifs = ["ELF1_1_aim", "HNF1A_1_aim", "HNF4A_1_aim"]

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

save_dir = "./Supps/S19_robust_anchors/"
for n_neighbors in [10, 20, 50, 100, 200, 500, 1000, 2000]: 
    file_map = {
            "Filter_Efficiency": f"./Supps/S19_robust_anchors/results_anchors/anchors_{n_neighbors}/radar_pseudo_effect.csv",
            "Retained_Activity": f"./Supps/S19_robust_anchors/results_anchors/anchors_{n_neighbors}/radar_screen_effect.csv"
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
        plt.savefig(os.path.join(save_dir, f"lineplot_{n_neighbors}_{metric_name}.pdf"), bbox_inches='tight')
        plt.close()

