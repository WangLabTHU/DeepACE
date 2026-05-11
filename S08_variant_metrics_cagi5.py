'''
画序列的突变后保守性结果, 在CAGI5的15个数据集

/home/hyu/Digital_Platform/manuals/fig2e_point_mutation.py
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
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import mahalanobis

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def compute_sample_similarity(pred_alt, pred_ref):
    pred_alt = np.asarray(pred_alt)
    pred_ref = np.asarray(pred_ref)
    cov = np.cov(pred_ref, rowvar=False)
    cov_inv = np.linalg.pinv(cov)
    sims = []
    for x in pred_alt:
        dists = [-mahalanobis(x, y, cov_inv) for y in pred_ref]
        dists = np.array(dists)
        sims.append(dists.mean())
    sims = np.array(sims)
    sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-12)
    return sims
        
def plot_similarity_vs_expression(similarities, variant_effects, motif, output_dir, smooth_sigma=5, prefix="backbone"):
    similarities = np.asarray(similarities, dtype=float)
    variant_effects = np.asarray(variant_effects, dtype=float)
    mask = np.isfinite(similarities) & np.isfinite(variant_effects)
    similarities = similarities[mask]
    variant_effects = variant_effects[mask]
    order = np.argsort(-similarities)
    sorted_effects = variant_effects[order]
    smooth_effects = gaussian_filter1d(sorted_effects, sigma=smooth_sigma, mode='nearest')

    plt.figure(figsize=(8, 6))
    x = np.arange(1, len(sorted_effects) + 1)
    plt.scatter(x, sorted_effects, color='gray', s=8, alpha=0.5, label='Raw variant effects')
    plt.plot(x, smooth_effects, color='red', linewidth=2, label='Smoothed trend')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Samples (sorted by similarity)")
    plt.ylabel("Variant Effect (log2 fold change)")
    plt.title(f"{motif}: Expression Trend along Similarity Rank")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_similarity_vs_expression_{motif}.pdf", dpi=400, bbox_inches="tight")
    plt.close()
    # pd.DataFrame({
    #     "order": np.arange(1, len(order)+1),
    #     "sorted_effects": sorted_effects,
    #     "smooth_effects": smooth_effects
    # }).to_csv(f"{output_dir}/{prefix}_similarity_vs_expression_{motif}.csv", index=False)


    
def plot_screen_effect(similarity_all, variant_effects, motif, output_dir, cut_size=100, kernel_size=5, prefix="randaug"):
    similarity_all = np.asarray(similarity_all)
    variant_effects = np.asarray(variant_effects)
    order = np.argsort(-similarity_all)
    mean_remaining, std_remaining = [], []
    for k in range(1, len(order) + 1):
        remaining_idx = order[k:]
        if len(remaining_idx) > 0:
            mean_val = variant_effects[remaining_idx].mean()
            std_val = variant_effects[remaining_idx].std()
        else:
            mean_val = np.nan
            std_val = np.nan
        mean_remaining.append(mean_val)
        std_remaining.append(std_val)
    mean_remaining_smooth = gaussian_filter1d(mean_remaining, sigma=kernel_size, mode='nearest')
    std_remaining_smooth = gaussian_filter1d(std_remaining, sigma=kernel_size, mode='nearest')

    if cut_size is None or cut_size <= 0:
        x = np.arange(len(order))
        mean_to_plot = mean_remaining_smooth
        std_to_plot = std_remaining_smooth
    else:
        x = np.arange(0, len(order) - cut_size + 1)
        mean_to_plot = mean_remaining_smooth[:-cut_size+1]
        std_to_plot = std_remaining_smooth[:-cut_size+1]
    plt.figure(figsize=(8, 6))
    plt.plot(x, mean_to_plot, color='purple', linewidth=2, label='Mean Expression')
    plt.xlabel("Number of High-Similarity Samples Removed")
    plt.ylabel("Mean Expression of Remaining Samples")
    plt.title(f"Expression Stability after Removing Top Similar Samples ({motif})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{prefix}_screen_effect_{motif}.pdf")
    plt.savefig(fig_path, dpi=400, bbox_inches='tight')
    plt.close()
    # df = pd.DataFrame({
    #     "removed_top_n": np.arange(1, len(order) + 1),
    #     "mean_remaining": mean_remaining,
    #     "std_remaining": std_remaining,
    #     "mean_smooth": mean_remaining_smooth,
    #     "std_smooth": std_remaining_smooth
    # })
    # csv_path = os.path.join(output_dir, f"{prefix}_screen_effect_{motif}.csv")
    # df.to_csv(csv_path, index=False)


def plot_pseudo_effect(similarity_all, variant_effects, motif, output_dir, kernel_size=5, prefix="randaug"):
    os.makedirs(output_dir, exist_ok=True)
    similarity_all = np.asarray(similarity_all)
    variant_effects = np.asarray(variant_effects)
    order = np.argsort(-similarity_all)
    pos_flags = variant_effects > 0
    neg_flags = variant_effects < 0
    cum_neg = np.cumsum(neg_flags[order])
    cum_pos = np.cumsum(pos_flags[order])
    total_samples = np.arange(1, len(order) + 1)
    prop_fin = (cum_neg + 1) / (cum_pos + 1)  # normalize
    prop_fin_smooth = gaussian_filter1d(prop_fin, sigma=kernel_size, mode='nearest')

    plt.figure(figsize=(8, 6))
    plt.plot(total_samples, prop_fin_smooth,
             label="Negative / Positive Proportion",
             color='red', linewidth=2)
    plt.axhline(y=prop_fin[-1], color='black', linestyle='--', linewidth=1, label="Neutral Line")
    plt.xlabel("Number of Samples (sorted by similarity)")
    plt.ylabel("Neg / Pos Proportion")
    plt.title(f"Negative/Positive Ratio along Similarity Ranking ({motif})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{prefix}_pseudo_effect_{motif}.pdf")
    plt.savefig(fig_path, dpi=400, bbox_inches="tight")
    plt.close()
    # df = pd.DataFrame({
    #     "rank": np.arange(1, len(order) + 1),
    #     "cum_neg": cum_neg,
    #     "cum_pos": cum_pos,
    #     "similarity_sorted": similarity_all[order],
    #     "variant_effects_sorted": variant_effects[order],
    #     "prop_fin": prop_fin,
    #     "prop_fin_smooth": prop_fin_smooth
    # })
    # csv_path = os.path.join(output_dir, f"{prefix}_pseudo_effect_{motif}.csv")
    # df.to_csv(csv_path, index=False)



def plot_positive_ratio(similarity_all, variant_effects, motif, output_dir, cut_size=100, kernel_size=5, prefix="randaug"):
    similarity_all = np.asarray(similarity_all)
    variant_effects = np.asarray(variant_effects)
    order = np.argsort(-similarity_all)
    pos_flags = variant_effects > 0
    pos_ratio_remaining = []
    for k in range(1, len(order) + 1):
        remaining_idx = order[k:]
        if len(remaining_idx) > 0:
            pos_ratio = np.mean(pos_flags[remaining_idx])
        else:
            pos_ratio = np.nan
        pos_ratio_remaining.append(pos_ratio)
    pos_ratio_smooth = gaussian_filter1d(pos_ratio_remaining, sigma=kernel_size, mode='nearest')
    if cut_size is None or cut_size <= 0:
        x = np.arange(len(order))
        y = pos_ratio_smooth
    else:
        x = np.arange(0, len(order) - cut_size + 1)
        y = pos_ratio_smooth[:-cut_size+1]
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color='green', linewidth=2, label='Positive Ratio')
    pos_ratio_fin = np.sum(pos_flags) / len(variant_effects)
    plt.axhline(y=pos_ratio_fin, color='black', linestyle='--', linewidth=1, label="Neutral Line")
    plt.xlabel("Number of High-Similarity Samples Removed")
    plt.ylabel("Positive Sample Ratio (Remaining)")
    plt.title(f"Positive Ratio in Remaining Samples ({motif})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{prefix}_positive_ratio_{motif}.pdf")
    plt.savefig(fig_path, dpi=400, bbox_inches='tight')
    plt.close()
    # df = pd.DataFrame({
    #     "removed_top_n": np.arange(1, len(order) + 1),
    #     "positive_ratio_remaining": pos_ratio_remaining,
    #     "positive_ratio_smooth": pos_ratio_smooth
    # })
    # csv_path = os.path.join(output_dir, f"{prefix}_positive_ratio_{motif}.csv")
    # df.to_csv(csv_path, index=False)

def plot_distance_violin(similarity_all, variant_effects, motif, output_dir, group_size=250, prefix="randaug"):
    similarity_all = np.asarray(similarity_all)
    lfc = np.asarray(variant_effects)
    order = np.argsort(-similarity_all)
    lfc_sorted = lfc[order]
    groups, group_labels, medians = [], [], []
    for i in range(0, len(lfc_sorted), group_size):
        end_idx = min(i + group_size, len(lfc_sorted))
        group_lfc = lfc_sorted[i:end_idx]
        groups.append(group_lfc)
        group_labels.append(f"{i+1}-{end_idx}")
        medians.append(np.median(group_lfc) if len(group_lfc) > 0 else np.nan)
    df_violin = pd.DataFrame({
        'LFC': np.concatenate(groups),
        'Group': np.repeat(group_labels, [len(g) for g in groups])
    })
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x='Group', y='LFC', hue='Group', data=df_violin, 
        palette='RdBu_r', inner='quartile', legend=False
    )
    plt.axhline(y=np.mean(lfc_sorted), color='black', linestyle='--', linewidth=1, label='Global Mean')
    valid_mask = ~np.isnan(medians)
    plt.plot(np.arange(len(group_labels))[valid_mask], np.array(medians)[valid_mask], 
             color='red', linewidth=2, marker='o', label='Median LFC')
    for x, y in zip(np.arange(len(group_labels))[valid_mask], np.array(medians)[valid_mask]):
        plt.text(x, y, f'{y:.2f}', fontsize=8, ha='center', va='bottom', color='red',
                 bbox=dict(facecolor='white', edgecolor='white', alpha=1.0, boxstyle='round,pad=0.2'))
    plt.xlabel("Sample Groups (sorted by decreasing similarity)")
    plt.ylabel("Measured log fold change (LFC)")
    plt.title(f"Violin Plot of LFC by Similarity Group ({motif})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{prefix}_distance_lfc_{motif}.pdf")
    plt.savefig(fig_path, dpi=400, bbox_inches='tight')
    plt.close()
    # csv_path = os.path.join(output_dir, f"{prefix}_distance_lfc_{motif}.csv")
    # df_violin.to_csv(csv_path, index=False)


''' Lineplot Visualization '''

metric = "mahalanobis"
dataset = "MPRABase"
motif_list = ["TERT", "HBG1", "LDLR", "F9", "GP1BA", "IRF4", "IRF6", "PKLR", "ZFAND3", "SORT1", 
              "HBB", "UC88", "MYC_rs6983267", "RET", "TCF7L2"] 

output_dir = f"./Supps/S08_variant_metrics_cagi5/"
print(f"Processing dataset: {dataset}")
for motif in motif_list:
    print(f"Processing motif/background: {motif}")
    uni_path = f"./Preds/D05_mprabase/point_{dataset}_{motif}_saturation/uni_pred.npy"
    df_path = f"./Preds/D05_mprabase/point_{dataset}_{motif}_saturation.tsv"
    uni_list = np.load(uni_path)
    df = pd.read_csv(df_path, sep="\t")
    variant_effects = df['VariantExpressionEffect (log2)'].to_numpy()
    pred_alt = uni_list[:-1]
    pred_ref = np.repeat(uni_list[-1][np.newaxis, :], len(pred_alt), axis=0)
    valid_mask = np.isfinite(pred_alt).any(axis=0) & np.isfinite(pred_ref).any(axis=0)
    pred_alt = pred_alt[:, valid_mask]
    pred_ref = pred_ref[:, valid_mask]
    pred_rand = np.load("./Preds/D10_random/random_sample_1/uni_pred.npy")
    pred_rand = pred_rand[:, valid_mask]
    combined = np.vstack([pred_alt, pred_ref, pred_rand])
    combined_pca = PCA(n_components=50, random_state=42).fit_transform(combined)
    pred_alt_pca = combined_pca[:len(pred_alt)]
    pred_ref_pca = combined_pca[len(pred_alt):-len(pred_rand)]
    pred_rand_pca = combined_pca[-len(pred_rand):]
    # lineplot-randaug
    similarity_all = compute_sample_similarity(pred_alt_pca, pred_rand_pca)
    plot_similarity_vs_expression(similarity_all, variant_effects, motif, output_dir=output_dir, prefix="randaug")
    plot_screen_effect(similarity_all, variant_effects, motif, output_dir, prefix="randaug")
    plot_pseudo_effect(similarity_all, variant_effects, motif, output_dir, prefix="randaug")
    plot_positive_ratio(similarity_all, variant_effects, motif, output_dir, prefix="randaug")
    plot_distance_violin(similarity_all, variant_effects, motif, output_dir, prefix="randaug")
    # pd.DataFrame({"scores": similarity_all, "variant_effects": variant_effects}).to_csv(f"{output_dir}/pca50_variant_scores_{motif}.csv")