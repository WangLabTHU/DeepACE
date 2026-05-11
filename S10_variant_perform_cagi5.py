'''
画序列的突变后quantile discov / PCC, 在CAGI5的15个数据集

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

def plot_quantile_discover_rate_per_motif(
    motif, df_dict, output_dir,
    quantile_grid=None,
    save_name=None):
    if quantile_grid is None:
        quantile_grid = np.linspace(0.01, 0.99, 99)
    algo_list = list(df_dict.keys())
    plt.figure(figsize=(8, 6))
    auc_dict = {}
    df_plot = {"quantile": quantile_grid}
    for algo in algo_list:
        df = df_dict[algo][motif]
        scores = np.asarray(df["scores"])
        effects = np.asarray(df["variant_effects"])
        discover_rates = []
        nan_mask = ~np.isnan(scores)
        scores = scores[nan_mask]
        effects = effects[nan_mask]
        for q in quantile_grid:
            score_q = np.quantile(scores, q)
            mask_S = scores <= score_q
            S_size = mask_S.sum()
            if S_size == 0:
                discover_rates.append(0.0)
                continue
            effect_q = np.quantile(effects, q)
            hits = np.sum(effects[mask_S] <= effect_q)
            discover_rates.append(hits / S_size)
        auc = np.trapz(discover_rates, quantile_grid)
        auc_dict[algo] = auc
        df_plot[f"discover_rate_{algo}"] = discover_rates
        plt.plot(quantile_grid, discover_rates, label=f"{algo} (AUC={auc:.3f})", linewidth=2)
    plt.title(f"Quantile-aligned Discover Rate — {motif}")
    plt.xlabel("Score quantile")
    plt.ylabel("Discover rate")
    plt.grid(alpha=0.3)
    plt.legend()
    save_path = os.path.join(output_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Saved quantile-aligned discover-rate curve for {motif} → {save_path}")
    df_plot = pd.DataFrame(df_plot)
    # df_plot.to_csv(f"{output_dir}/{save_name[:-4] + '.csv'}") 
    return auc_dict



def plot_pcc_per_motif(motif, df_dict, output_dir, save_name="pcc_plot.pdf"):
    df_pca = df_dict["PCA"][motif]
    df_evo2 = df_dict["Evo2"][motif]
    df_promoterAI = df_dict["promoterAI"][motif]
    scaler = StandardScaler()
    df_pca["normalized_scores"] = scaler.fit_transform(df_pca[["scores"]])
    df_evo2["normalized_scores"] = scaler.fit_transform(df_evo2[["scores"]])
    df_promoterAI["normalized_scores"] = scaler.fit_transform(df_promoterAI[["scores"]])
    corr_pca, _ = pearsonr(df_pca["variant_effects"], df_pca["normalized_scores"])
    corr_evo2, _ = pearsonr(df_evo2["variant_effects"], df_evo2["normalized_scores"])
    corr_promoterAI, _ = pearsonr(df_promoterAI["variant_effects"], df_promoterAI["normalized_scores"])
    plt.figure(figsize=(8, 6))
    plt.scatter(df_pca["normalized_scores"], df_pca["variant_effects"],  label=f"PCA (PCC={corr_pca:.2f})", color='tab:blue', alpha=0.4, edgecolor=None, s=10)
    plt.scatter(df_evo2["normalized_scores"], df_evo2["variant_effects"],  label=f"Evo2 (PCC={corr_evo2:.2f})", color='tab:orange', alpha=0.4, edgecolor=None, s=10)
    plt.scatter(df_promoterAI["normalized_scores"], df_promoterAI["variant_effects"],  label=f"promoterAI (PCC={corr_promoterAI:.2f})", color='tab:green', alpha=0.4, edgecolor=None, s=10)

    x_min = min(df_pca["normalized_scores"].min(), df_evo2["normalized_scores"].min(), df_promoterAI["normalized_scores"].min())
    x_max = max(df_pca["normalized_scores"].max(), df_evo2["normalized_scores"].max(), df_promoterAI["normalized_scores"].max())
    x_values = np.linspace(x_min, x_max, 100)
    y_values_pca = corr_pca * (np.std(df_pca["variant_effects"]) / np.std(df_pca["normalized_scores"])) * (x_values - np.mean(df_pca["normalized_scores"])) + np.mean(df_pca["variant_effects"])
    y_values_evo2 = corr_evo2 * (np.std(df_evo2["variant_effects"]) / np.std(df_evo2["normalized_scores"])) * (x_values - np.mean(df_evo2["normalized_scores"])) + np.mean(df_evo2["variant_effects"])
    y_values_promoterAI = corr_promoterAI * (np.std(df_promoterAI["variant_effects"]) / np.std(df_promoterAI["normalized_scores"])) * (x_values - np.mean(df_promoterAI["normalized_scores"])) + np.mean(df_promoterAI["variant_effects"])
    plt.plot(x_values, y_values_pca, label=None, color='tab:blue', linewidth=2)
    plt.plot(x_values, y_values_evo2, label=None, color='tab:orange', linewidth=2)
    plt.plot(x_values, y_values_promoterAI, label=None, color='tab:green', linewidth=2)

    plt.ylabel("Variant Effects", fontsize=12)
    plt.xlabel("Normalized Scores", fontsize=12)
    plt.title(f"PCC Plot for {motif}", fontsize=14)
    plt.legend(loc='upper right')
    plot_path = f"{output_dir}/{save_name}"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=400)
    plt.close()
    df_save_pca = df_pca[["normalized_scores", "variant_effects"]].copy()
    df_save_pca["model"] = "PCA"
    df_save_evo2 = df_evo2[["normalized_scores", "variant_effects"]].copy()
    df_save_evo2["model"] = "Evo2"
    df_save_promoterAI = df_promoterAI[["normalized_scores", "variant_effects"]].copy()
    df_save_promoterAI["model"] = "promoterAI"
    df_save = pd.concat([df_save_pca, df_save_evo2, df_save_promoterAI], ignore_index=True)
    csv_path = plot_path.replace(".pdf", ".csv")
    # df_save.to_csv(csv_path, index=False)
    return plot_path

''' Quantile-based Discover Rate & PCC '''

metric = "mahalanobis"
dataset = "MPRABase"
motif_list = ["TERT", "HBG1", "LDLR", "F9", "GP1BA", "IRF4", "IRF6", "PKLR", "ZFAND3", "SORT1", 
              "HBB", "UC88", "MYC_rs6983267", "RET", "TCF7L2"] 

output_dir = f"./Supps/S10_variant_perform_cagi5/"
print(f"Processing dataset: {dataset}")
df_dict = {"PCA": {}, "Evo2": {}, "promoterAI": {}, "phyloP100way": {}, "phyloP470way": {}, "phastCons100way": {}, "phastCons470way": {}, "gpnmsa": {}}

for motif in motif_list:
    df_pca = pd.read_csv(f"./Preds/D05_mprabase/analysis_{metric}/pca50_variant_scores_{motif}.csv")
    df_evo2 = pd.read_csv(f"./Preds/D05_mprabase/analysis_evo2/evo2_variant_scores_{motif}.csv")
    df_promoterAI = pd.read_csv(f"./Preds/D05_mprabase/analysis_promoterai/promoterAI_variant_scores_{motif}.csv")
    df_phyloP100way = pd.read_csv(f"./Preds/D05_mprabase/analysis_cons/phyloP100way_variant_scores_{motif}.csv")
    df_phyloP470way = pd.read_csv(f"./Preds/D05_mprabase/analysis_cons/phyloP470way_variant_scores_{motif}.csv")
    df_phastCons100way = pd.read_csv(f"./Preds/D05_mprabase/analysis_cons/phastCons100way_variant_scores_{motif}.csv")
    df_phastCons470way = pd.read_csv(f"./Preds/D05_mprabase/analysis_cons/phastCons470way_variant_scores_{motif}.csv")
    df_gpnmsa = pd.read_csv(f"./Preds/D05_mprabase/analysis_gpnmsa/gpnmsa_variant_scores_{motif}.csv")

    df_pca = df_pca[["scores", "variant_effects"]]
    df_pca["scores"] = -df_pca["scores"]
    df_evo2 = df_evo2[["scores", "variant_effects"]]
    df_promoterAI = df_promoterAI[["scores", "variant_effects"]]
    df_dict["PCA"][motif] = df_pca
    df_dict["Evo2"][motif] = df_evo2
    df_dict["promoterAI"][motif] = df_promoterAI
    df_dict["phyloP100way"][motif] = df_phyloP100way
    df_dict["phyloP470way"][motif] = df_phyloP470way
    df_dict["phastCons100way"][motif] = df_phastCons100way
    df_dict["phastCons470way"][motif] = df_phastCons470way
    df_dict["gpnmsa"][motif] = df_gpnmsa
            
    auc_dict = plot_quantile_discover_rate_per_motif(
        motif, df_dict, output_dir,
        quantile_grid=np.linspace(0.01, 0.99, 99),
        save_name=f"quantile_discover_rate_{motif}.pdf")
    plot_pcc_per_motif(motif, df_dict, output_dir, save_name=f"pcc_lineplot_{motif}.pdf")



