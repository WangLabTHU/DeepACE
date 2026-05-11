'''
/home/hyu/Figures/DeepACE/Fig2.py
/home/hyu/Digital_Platform/manuals/fig2f_point_mutation_evo2.py
/home/hyu/Digital_Platform/manuals/fig2g_point_mutation_cold.py
/home/hyu/Digital_Platform/manuals/xfig2b_point_mutation_final.py

cp /home/hyu/Figures/DeepACE/Fig2/Fig2d_curve_IRF4.svg /home/hyu/DeepACE/Figs/F02_variant_effects/F02d_IRF4_quantile.svg
cp /home/hyu/Digital_Platform/manuals/fig2e_point_mutation/MPRABase_cosine/pca50_* /home/hyu/DeepACE/Preds/D05_mprabase/analysis_cosine
cp /home/hyu/Digital_Platform/manuals/fig2e_point_mutation/MPRABase_mahalanobis/pca50_* /home/hyu/DeepACE/Preds/D05_mprabase/analysis_mahalanobis
cp -r /home/hyu/Digital_Platform/manuals/fig_dataset/random_sample_* /home/hyu/DeepACE/Preds/D10_random
cp /home/hyu/Digital_Platform/manuals/fig2g_point_mutation_cold/MPRABase_mahalanobis/pca50_* /home/hyu/DeepACE/Preds/D05_mprabase/analysis_cold
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

def compute_sample_similarity(pred_alt, pred_ref, metric="cosine", mode="pairwise"):
    pred_alt = np.asarray(pred_alt)
    pred_ref = np.asarray(pred_ref)
    if mode not in ["pairwise", "batch"]:
        raise ValueError(f"Unknown mode '{mode}', must be 'pairwise' or 'batch'")
    if mode == "pairwise":
        n_samples = len(pred_alt)
        if metric == "cosine":
            sims = np.array([
                cosine_similarity(pred_alt[i].reshape(1, -1), pred_ref[i].reshape(1, -1))[0, 0]
                for i in range(n_samples)
            ])
            return sims
        elif metric == "mahalanobis":
            cov = np.cov(pred_ref, rowvar=False)
            cov_inv = np.linalg.pinv(cov)
            sims = np.array([
                -mahalanobis(pred_alt[i], pred_ref[i], cov_inv)
                for i in range(n_samples)
            ])
            sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-12)
            return sims
    elif mode == "batch":
        if metric == "cosine":
            sims_matrix = cosine_similarity(pred_alt, pred_ref)
            sims = sims_matrix.mean(axis=1) 
            return sims
        elif metric == "mahalanobis":
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
    

def plot_umap(sample_data, sample_labels, sorted_labels, plot_tag, output_dir="./fig2c_virtual_screen"):
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

    plt.figure(figsize=(8, 6))
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
        xgrid = np.linspace(df_plot['Dim1'].min(), df_plot['Dim1'].max(), 400)
        ygrid = np.linspace(df_plot['Dim2'].min(), df_plot['Dim2'].max(), 400)
        X, Y = np.meshgrid(xgrid, ygrid)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)        
        Z_masked = np.ma.masked_where(Z < 0.001 * Z.max(), Z)  
        Z_expr = Z_masked / Z_masked.max() * (expr_raw.max() - expr_raw.min()) + expr_raw.min()
        contour_real = plt.contourf(X, Y, Z_expr, levels=20, cmap='Greens', alpha=0.7, norm=norm)

    plt.scatter(
        df_plot.loc[df_plot['Group'] == 'Pseudo', 'Dim1'],
        df_plot.loc[df_plot['Group'] == 'Pseudo', 'Dim2'],
        color='#249875', s=50, alpha=0.9, edgecolor="white", label='Original backbone'
    )
    
    plt.scatter(
        df_plot.loc[df_plot['Group'] == 'Rand', 'Dim1'],
        df_plot.loc[df_plot['Group'] == 'Rand', 'Dim2'],
        color='k', s=2, alpha=0.9, edgecolor="None", label='Random mutation'
    )
    cbar = plt.colorbar(contour_real)
    cbar.set_label("Expression-weighted KDE", fontsize=12)
    plt.title(f"UMAP (Pseudo anchor, real contour) - {plot_tag}")
    plt.xlabel('UMAP Dim1')
    plt.ylabel('UMAP Dim2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/umap2d_{plot_tag}.pdf", dpi=400, bbox_inches='tight')
    df_plot.to_csv(f"{output_dir}/umap2d_{plot_tag}.csv", index=False)
    plt.close()
    
def plot_similarity_vs_expression(similarities, variant_effects, motif, output_dir, smooth_sigma=5, prefix="backbone"):
    os.makedirs(output_dir, exist_ok=True)

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

    pd.DataFrame({
        "order": np.arange(1, len(order)+1),
        "sorted_effects": sorted_effects,
        "smooth_effects": smooth_effects
    }).to_csv(f"{output_dir}/{prefix}_similarity_vs_expression_{motif}.csv", index=False)
    
def plot_screen_effect(similarity_all, variant_effects, motif, output_dir, cut_size=100, kernel_size=5, prefix="randaug"):
    os.makedirs(output_dir, exist_ok=True)
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
    df = pd.DataFrame({
        "removed_top_n": np.arange(1, len(order) + 1),
        "mean_remaining": mean_remaining,
        "std_remaining": std_remaining,
        "mean_smooth": mean_remaining_smooth,
        "std_smooth": std_remaining_smooth
    })
    csv_path = os.path.join(output_dir, f"{prefix}_screen_effect_{motif}.csv")
    df.to_csv(csv_path, index=False)


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
    prop_fin = (cum_neg + 1) / (cum_pos + 1)  # 避免除零

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

    df = pd.DataFrame({
        "rank": np.arange(1, len(order) + 1),
        "cum_neg": cum_neg,
        "cum_pos": cum_pos,
        "similarity_sorted": similarity_all[order],
        "variant_effects_sorted": variant_effects[order],
        "prop_fin": prop_fin,
        "prop_fin_smooth": prop_fin_smooth
    })
    csv_path = os.path.join(output_dir, f"{prefix}_pseudo_effect_{motif}.csv")
    df.to_csv(csv_path, index=False)



def plot_positive_ratio(similarity_all, variant_effects, motif, output_dir, cut_size=100, kernel_size=5, prefix="randaug"):
    os.makedirs(output_dir, exist_ok=True)
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

    df = pd.DataFrame({
        "removed_top_n": np.arange(1, len(order) + 1),
        "positive_ratio_remaining": pos_ratio_remaining,
        "positive_ratio_smooth": pos_ratio_smooth
    })
    csv_path = os.path.join(output_dir, f"{prefix}_positive_ratio_{motif}.csv")
    df.to_csv(csv_path, index=False)

def plot_distance_violin(similarity_all, variant_effects, motif, output_dir, group_size=250, prefix="randaug"):
    os.makedirs(output_dir, exist_ok=True)
    
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
    csv_path = os.path.join(output_dir, f"{prefix}_distance_lfc_{motif}.csv")
    df_violin.to_csv(csv_path, index=False)
    

'''
[01] PCA50  cosine / manalanobis
'''


metrics = ["cosine", "mahalanobis"]
datasets = ["MPRABase"] #   
motif_map = {
    "MPRABase": ["TERT", "HBG1", "LDLR", "F9", "GP1BA", "IRF4", "IRF6", "PKLR", "ZFAND3", "SORT1",
                 "HBB", "HNF4A", "ZRS", "UC88", "MSMB", "MYC_rs6983267", "RET", "TCF7L2"] 
}


for dataset in datasets:
    for metric in metrics:
        print(f"Processing dataset: {dataset}")
        output_dir = f"./Preds/D05_mprabase/analysis_{metric}"
        all_similarities, all_groups, all_sources = [], [], []
        motif_list = motif_map[dataset]
        for motif in motif_list:
            print(f"Processing motif/background: {motif}")
            uni_path = f"./Preds/D05_mprabase/point_{dataset}_{motif}_saturation/uni_pred.npy"
            df_path = f"./Preds/D05_mprabase/point_{dataset}_{motif}_saturation.tsv"
            uni_list = np.load(uni_path)
            df = pd.read_csv(df_path, sep="\t")
            variant_effects = df['VariantExpressionEffect (log2)'].to_numpy()

            # non nan screening
            pred_alt = uni_list[:-1]
            pred_ref = np.repeat(uni_list[-1][np.newaxis, :], len(pred_alt), axis=0)
            valid_mask = np.isfinite(pred_alt).any(axis=0) & np.isfinite(pred_ref).any(axis=0)
            pred_alt = pred_alt[:, valid_mask]
            pred_ref = pred_ref[:, valid_mask]
            
            # random sample visualization
            if metric == "cosine":
                pred_rand = np.load(f"./Preds/D05_mprabase/point_MPRABase_{motif}_randaug/uni_pred.npy")[:-1]
                pred_rand = pred_rand[:, valid_mask]
            elif metric == "mahalanobis":
                pred_rand = np.load("./Preds/D10_random/random_sample_1/uni_pred.npy")
                pred_rand = pred_rand[:, valid_mask]
            
            combined = np.vstack([pred_alt, pred_ref, pred_rand])
            combined_pca = PCA(n_components=50, random_state=42).fit_transform(combined)
            pred_alt_pca = combined_pca[:len(pred_alt)]
            pred_ref_pca = combined_pca[len(pred_alt):-len(pred_rand)]
            pred_rand_pca = combined_pca[-len(pred_rand):]

            sample_data = np.vstack([pred_ref_pca, pred_alt_pca, pred_rand_pca])
            sample_labels = ['Pseudo'] * len(pred_ref_pca) + ['Real'] * len(pred_alt_pca) + ['Rand'] * len(pred_rand_pca)
            plot_umap(sample_data, sample_labels, variant_effects, plot_tag=motif, output_dir=output_dir)
            similarity_all = compute_sample_similarity(pred_alt_pca, pred_rand_pca, metric=metric, mode="batch")
            # plot_similarity_vs_expression(similarity_all, variant_effects, motif, output_dir=output_dir, prefix="randaug")
            # plot_screen_effect(similarity_all, variant_effects, motif, output_dir, prefix="randaug")
            # plot_pseudo_effect(similarity_all, variant_effects, motif, output_dir, prefix="randaug")
            # plot_positive_ratio(similarity_all, variant_effects, motif, output_dir, prefix="randaug")
            # plot_distance_violin(similarity_all, variant_effects, motif, output_dir, prefix="randaug")
            pd.DataFrame({"scores": similarity_all, "variant_effects": variant_effects}).to_csv(f"{output_dir}/pca50_variant_scores_{motif}.csv")


'''
[02] PCA50 cold
'''

metrics = ["mahalanobis"]
datasets = ["MPRABase"] #   
motif_map = {
    "MPRABase": ["TERT", "HBG1", "LDLR", "F9", "GP1BA", "IRF4", "IRF6", "PKLR", "ZFAND3", "SORT1",
                 "HBB", "HNF4A", "ZRS", "UC88", "MSMB", "MYC_rs6983267", "RET", "TCF7L2"] 
}

for dataset in datasets:
    for metric in metrics:
        print(f"Processing dataset: {dataset}")
        output_dir = f"./Preds/D05_mprabase/analysis_cold"
        all_similarities, all_groups, all_sources = [], [], []
        motif_list = motif_map[dataset]
        for motif in motif_list:
            print(f"Processing motif/background: {motif}")
            uni_path = f"./Preds/D05_mprabase/point_{dataset}_{motif}_saturation/uni_pred.npy"
            df_path = f"./Preds/D05_mprabase/point_{dataset}_{motif}_saturation.tsv"
            uni_list = np.load(uni_path)
            df = pd.read_csv(df_path, sep="\t")
            variant_effects = df['VariantExpressionEffect (log2)'].to_numpy()

            # non nan screening
            pred_alt = uni_list[:-1]
            pred_ref = np.repeat(uni_list[-1][np.newaxis, :], len(pred_alt), axis=0)
            valid_mask = np.isfinite(pred_alt).any(axis=0) & np.isfinite(pred_ref).any(axis=0)
            pred_alt = pred_alt[:, valid_mask]
            pred_ref = pred_ref[:, valid_mask]
            
            # random sample visualization
            if metric == "cosine":
                pred_rand = np.load(f"./Preds/D05_mprabase/point_MPRABase_{motif}_randaug/uni_pred.npy")[:-1]
                pred_rand = pred_rand[:, valid_mask]
            elif metric == "mahalanobis":
                pred_rand = np.load("./Preds/D10_random/random_sample_1/uni_pred.npy")
                pred_rand = pred_rand[:, valid_mask]
            
            combined = np.vstack([pred_alt, pred_ref, pred_rand])
            pca = joblib.load('./Preds/D01_screens/pca_model.pkl')
            combined_pca = pca.transform(combined)
            pred_alt_pca = combined_pca[:len(pred_alt)]
            pred_ref_pca = combined_pca[len(pred_alt):-len(pred_rand)]
            pred_rand_pca = combined_pca[-len(pred_rand):]

            sample_data = np.vstack([pred_ref_pca, pred_alt_pca, pred_rand_pca])
            sample_labels = ['Pseudo'] * len(pred_ref_pca) + ['Real'] * len(pred_alt_pca) + ['Rand'] * len(pred_rand_pca)
            plot_umap(sample_data, sample_labels, variant_effects, plot_tag=motif, output_dir=output_dir)
            similarity_all = compute_sample_similarity(pred_alt_pca, pred_rand_pca, metric=metric, mode="batch")
            # plot_similarity_vs_expression(similarity_all, variant_effects, motif, output_dir=output_dir, prefix="randaug")
            # plot_screen_effect(similarity_all, variant_effects, motif, output_dir, prefix="randaug")
            # plot_pseudo_effect(similarity_all, variant_effects, motif, output_dir, prefix="randaug")
            # plot_positive_ratio(similarity_all, variant_effects, motif, output_dir, prefix="randaug")
            # plot_distance_violin(similarity_all, variant_effects, motif, output_dir, prefix="randaug")
            pd.DataFrame({"scores": similarity_all, "variant_effects": variant_effects}).to_csv(f"{output_dir}/pca50_variant_scores_{motif}.csv")





