'''
/home/hyu/Digital_Platform/manuals/xfig2b_point_mutation_final.py

cp /home/hyu/Digital_Platform/manuals/xfig2b_point_mutation_final/* /home/hyu/DeepACE/Preds/D05_mprabase/analysis_united
cp /home/hyu/Figures/DeepACE/Fig2/Fig2e_AUC_heatmap.svg /home/hyu/DeepACE/Figs/F02_variant_effects/F02e_heatmap_quantile.svg
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


    
def calculate_auc_per_motif(motif, df_dict, quantile_grid=None):
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
    return auc_dict

def plot_auc_barplot(auc_records, output_dir, save_name="auc_barplot.pdf"):
    plot_data = []
    for motif, auc_dict in auc_records.items():
        for algo, auc in auc_dict.items():
            plot_data.append({"Motif": motif, "Algorithm": algo, "AUC": auc})
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(42, 8))
    ax = sns.barplot(data=df_plot, x="Motif", y="AUC",
                     hue="Algorithm", edgecolor="k")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
    plt.title("AUC Comparison Across Motifs")
    plt.xticks(rotation=0)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(df_plot["Algorithm"].unique()),
        frameon=False
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Saved AUC barplot → {save_path}")
    df_plot.to_csv(f"{output_dir}/{save_name[:-4] + '.csv'}")


def plot_pcc_barplot(df_dict, motif_list, output_dir, save_name="pcc_barplot.pdf"):
    plot_data = []
    algorithms = ["DeepACE-randaug", "DeepACE-random", "DeepACE-cold",
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    for motif in motif_list:
        for algo in algorithms:
            df = df_dict[algo][motif]
            df_valid = df.dropna(subset=["scores", "variant_effects"])
            if len(df_valid) < 3:
                pcc = np.nan
            else:
                pcc, _ = pearsonr(df_valid["scores"], df_valid["variant_effects"])
            plot_data.append({
                "Motif": motif,
                "Algorithm": algo,
                "PCC": pcc
            })
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(42, 8))
    ax = sns.barplot(
        data=df_plot,
        x="Motif",
        y="PCC",
        hue="Algorithm",
        edgecolor="k"
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
    plt.title("PCC Comparison Across Motifs")
    plt.xticks(rotation=0)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(df_plot["Algorithm"].unique()),
        frameon=False
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Saved PCC barplot → {save_path}")
    df_plot.to_csv(f"{output_dir}/{save_name[:-4] + '.csv'}")

def plot_scc_barplot(df_dict, motif_list, output_dir, save_name="scc_barplot.pdf"):
    plot_data = []
    algorithms = ["DeepACE-randaug", "DeepACE-random", "DeepACE-cold",
              "Evo2", "promoterAI",
              "phyloP100way", "phyloP470way",
              "phastCons100way", "phastCons470way", "gpnmsa"]
    for motif in motif_list:
        for algo in algorithms:
            df = df_dict[algo][motif]
            df_valid = df.dropna(subset=["scores", "variant_effects"])
            if len(df_valid) < 3:
                scc = np.nan
            else:
                scc, _ = spearmanr(df_valid["scores"], df_valid["variant_effects"])
            plot_data.append({
                "Motif": motif,
                "Algorithm": algo,
                "SCC": scc
            })
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(42, 8))
    ax = sns.barplot(
        data=df_plot,
        x="Motif",
        y="SCC",
        hue="Algorithm",
        edgecolor="k"
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
    plt.title("SCC Comparison Across Motifs")
    plt.xticks(rotation=0)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(df_plot["Algorithm"].unique()),
        frameon=False
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Saved SCC barplot → {save_path}")
    df_plot.to_csv(f"{output_dir}/{save_name[:-4] + '.csv'}")


def plot_auc_boxplot_summary(auc_records, output_dir, save_name="auc_boxplot_summary.pdf"):
    algorithms = ["DeepACE-randaug", "DeepACE-random", "DeepACE-cold",
              "Evo2", "promoterAI",
              "phyloP100way", "phyloP470way",
              "phastCons100way", "phastCons470way", "gpnmsa"]
    algo_palette = {"PCA": "#1f77b4", "Evo2": "#ff7f0e", "promoterAI": "#2ca02c"}

    plot_data = []
    motif_order = []
    for motif, auc_dict in auc_records.items():
        motif_order.append(motif)
        for algo in algorithms:
            plot_data.append({"Motif": motif, "Algorithm": algo, "AUC": auc_dict[algo]})
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(14, 5))
    ax = sns.boxplot(data=df_plot, x="Algorithm", y="AUC", width=0.45, palette="deep", # palette=algo_palette,
                     showfliers=False, whis=[0, 100])

    sns.stripplot(data=df_plot, x="Algorithm", y="AUC", color="black", alpha=0.45, jitter=True, dodge=False)
    ylocs = ax.get_yticks()
    ylabels = [f'{y:.2f}' for y in ylocs]
    ax.set_yticklabels(ylabels)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, save_name), dpi=400)
    plt.close()
    print(f"Saved AUC boxplot → {os.path.join(output_dir, save_name)}")


def plot_pcc_boxplot_summary(df_dict, motif_list, output_dir, save_name="pcc_boxplot_summary.pdf"):
    algorithms = ["DeepACE-randaug", "DeepACE-random", "DeepACE-cold",
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    algo_palette = {"PCA": "#1f77b4", "Evo2": "#ff7f0e", "promoterAI": "#2ca02c"}
    plot_data = []
    motif_order = []
    for motif in motif_list:
        motif_order.append(motif)
        for algo in algorithms:
            df = df_dict[algo][motif].dropna(subset=["scores", "variant_effects"])
            pcc = pearsonr(df["scores"], df["variant_effects"])[0] if len(df) >= 3 else np.nan
            plot_data.append({"Motif": motif, "Algorithm": algo, "PCC": pcc})

    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(14, 5))
    ax = sns.boxplot(data=df_plot, x="Algorithm", y="PCC", width=0.45, palette="deep", # palette=algo_palette,
                     showfliers=False, whis=[0, 100])
    sns.stripplot(data=df_plot, x="Algorithm", y="PCC", color="black", alpha=0.45, jitter=True, dodge=False)
    ylocs = ax.get_yticks()
    ylabels = [f'{y:.2f}' for y in ylocs]
    ax.set_yticklabels(ylabels)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, save_name), dpi=400)
    plt.close()
    print(f"Saved PCC boxplot → {os.path.join(output_dir, save_name)}")


def plot_scc_boxplot_summary(df_dict, motif_list, output_dir, save_name="scc_boxplot_summary.pdf"):
    algorithms = ["DeepACE-randaug", "DeepACE-random", "DeepACE-cold",
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    algo_palette = {"PCA": "#1f77b4", "Evo2": "#ff7f0e", "promoterAI": "#2ca02c"}
    plot_data = []
    motif_order = []
    for motif in motif_list:
        motif_order.append(motif)
        for algo in algorithms:
            df = df_dict[algo][motif].dropna(subset=["scores", "variant_effects"])
            scc = spearmanr(df["scores"], df["variant_effects"])[0] if len(df) >= 3 else np.nan
            plot_data.append({"Motif": motif, "Algorithm": algo, "SCC": scc})

    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(14, 5))
    ax = sns.boxplot(data=df_plot, x="Algorithm", y="SCC", width=0.45, palette="deep", # palette=algo_palette,
                     showfliers=False, whis=[0, 100])
    sns.stripplot(data=df_plot, x="Algorithm", y="SCC", color="black", alpha=0.45, jitter=True, dodge=False)
    ylocs = ax.get_yticks()
    ylabels = [f'{y:.2f}' for y in ylocs]
    ax.set_yticklabels(ylabels)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, save_name), dpi=400)
    plt.close()
    print(f"Saved SCC boxplot → {os.path.join(output_dir, save_name)}")

'''
Tables Combined
'''

dataset = "MPRABase"
motif_map = {
    "MPRABase": ["TERT", "HBG1", "LDLR", "F9", "GP1BA", "IRF4", "IRF6", "PKLR", "ZFAND3", "SORT1",
                 "HBB", "UC88", "MYC_rs6983267", "RET", "TCF7L2"] # 
    }

output_dir = f"./Preds/D05_mprabase/analysis_united"
os.makedirs(output_dir, exist_ok=True)

print(f"Processing dataset: {dataset}")
motif_list = motif_map[dataset]
df_dict = {"DeepACE-randaug": {}, "DeepACE-random": {}, "DeepACE-cold": {},
           "Evo2": {}, "promoterAI": {},
           "phyloP100way": {}, "phyloP470way": {},
           "phastCons100way": {}, "phastCons470way": {}, "gpnmsa": {}}
auc_records = {}

for motif in motif_list:
    df_deepace_randaug = pd.read_csv(f"./Preds/D05_mprabase/analysis_cosine/pca50_variant_scores_{motif}.csv")
    df_deepace_random = pd.read_csv(f"./Preds/D05_mprabase/analysis_mahalanobis/pca50_variant_scores_{motif}.csv")
    df_deepace_cold = pd.read_csv(f"./Preds/D05_mprabase/analysis_cold/pca50_variant_scores_{motif}.csv")
    df_evo2 = pd.read_csv(f"./Preds/D05_mprabase/analysis_evo2/evo2_variant_scores_{motif}.csv")
    df_promoterAI = pd.read_csv(f"./Preds/D05_mprabase/analysis_promoterai/promoterAI_variant_scores_{motif}.csv")
    df_phyloP100way = pd.read_csv(f"./Preds/D05_mprabase/analysis_cons/phyloP100way_variant_scores_{motif}.csv")
    df_phyloP470way = pd.read_csv(f"./Preds/D05_mprabase/analysis_cons/phyloP470way_variant_scores_{motif}.csv")
    df_phastCons100way = pd.read_csv(f"./Preds/D05_mprabase/analysis_cons/phastCons100way_variant_scores_{motif}.csv")
    df_phastCons470way = pd.read_csv(f"./Preds/D05_mprabase/analysis_cons/phastCons470way_variant_scores_{motif}.csv")
    df_gpnmsa = pd.read_csv(f"./Preds/D05_mprabase/analysis_gpnmsa/gpnmsa_variant_scores_{motif}.csv")
            
    df_deepace_randaug = df_deepace_randaug[["scores", "variant_effects"]]
    df_deepace_randaug["scores"] = -df_deepace_randaug["scores"]
    df_deepace_random = df_deepace_random[["scores", "variant_effects"]]
    df_deepace_random["scores"] = -df_deepace_random["scores"]
    df_deepace_cold = df_deepace_cold[["scores", "variant_effects"]]
    df_deepace_cold["scores"] = -df_deepace_cold["scores"]
    df_evo2 = df_evo2[["scores", "variant_effects"]]
    df_promoterAI = df_promoterAI[["scores", "variant_effects"]]
    
    df_dict["DeepACE-randaug"][motif] = df_deepace_randaug
    df_dict["DeepACE-random"][motif] = df_deepace_random
    df_dict["DeepACE-cold"][motif] = df_deepace_cold
    df_dict["Evo2"][motif] = df_evo2
    df_dict["promoterAI"][motif] = df_promoterAI
    df_dict["phyloP100way"][motif] = df_phyloP100way
    df_dict["phyloP470way"][motif] = df_phyloP470way
    df_dict["phastCons100way"][motif] = df_phastCons100way
    df_dict["phastCons470way"][motif] = df_phastCons470way
    df_dict["gpnmsa"][motif] = df_gpnmsa
    auc_dict = calculate_auc_per_motif(motif, df_dict)
    auc_records[motif] = auc_dict

plot_auc_barplot(auc_records, output_dir, save_name="all_motifs_auc_barplot.pdf")
plot_pcc_barplot(df_dict=df_dict, motif_list=motif_list, output_dir=output_dir, save_name="all_motifs_pcc_barplot.pdf")
plot_scc_barplot(df_dict=df_dict, motif_list=motif_list, output_dir=output_dir, save_name="all_motifs_scc_barplot.pdf")
plot_auc_boxplot_summary(auc_records, output_dir, save_name="all_motifs_auc_boxplot_summary.pdf")
plot_pcc_boxplot_summary(df_dict, motif_list, output_dir, save_name="all_motifs_pcc_boxplot_summary.pdf")
plot_scc_boxplot_summary(df_dict, motif_list, output_dir, save_name="all_motifs_scc_boxplot_summary.pdf")


'''
Ploting Figures
'''
file_path = "./Preds/D05_mprabase/analysis_united/all_motifs_auc_barplot.csv"
save_path = "./Figs/F02_variant_effects/F02e_heatmap_quantile.svg"
df = pd.read_csv(file_path)
if df.columns[0].lower().startswith("unnamed"):
    df = df.drop(columns=df.columns[0])
df["Motif"] = df["Motif"].replace({"MYC_rs6983267": "MYC"})
data = df.pivot(index="Motif", columns="Algorithm", values="AUC")
    
mean_row = data.mean(axis=0)
data = pd.concat([data, mean_row.to_frame().T])
data.index = list(data.index[:-1]) + ["Mean"]
    
data = data.rename(columns={
    "phyloP100way": "phyloP 100",
    "phyloP470way": "phyloP 470",
    "phastCons100way": "phastCons 100",
    "phastCons470way": "phastCons 470",
    "gpnmsa": "GPN-MSA"
    })
sns.set_theme(style="ticks", context="paper")
g = sns.clustermap(
    data,
    row_cluster=False,
    col_cluster=True,
    cmap="YlGnBu",
    annot=True,
    fmt=".2f",
    annot_kws={"size": 18},
    linewidths=.5,
    linecolor="white",
    figsize=(10, 8),
    tree_kws={'linewidths': 2, 'colors': '#2d3436'},
    dendrogram_ratio=(0, 0.12),
    cbar_pos=(0.92, 0.92, 0.015, 0.12),
    cbar_kws={"label": "AUC"}
)
g.ax_cbar.set_ylabel("AUC", fontsize=24, labelpad=15)
g.ax_cbar.tick_params(labelsize=24)
ax = g.ax_heatmap
ax.set_title("AUC performance comparison\n15 saturated mutagenesis datasets", fontsize=24, pad=60)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=24)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=24)
for label in ax.get_yticklabels():
    if label.get_text() == "Mean":
        label.set_fontweight("bold")
        label.set_color("black")
        label.set_fontsize(24)
ax.set_xlabel("")
ax.set_ylabel("")
g.ax_heatmap.invert_xaxis()
g.ax_col_dendrogram.invert_xaxis()
g.savefig(save_path, bbox_inches="tight")