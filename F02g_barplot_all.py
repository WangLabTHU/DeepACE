'''
/home/hyu/Digital_Platform/manuals/xfig2e_motif_substration_final.py

cp /home/hyu/Digital_Platform/manuals/fig2d_motif_substration/Epigenetics_3TF_cosine/randaug/pca50_variant_scores_* /home/hyu/DeepACE/Preds/D04_deeptfbu/analysis_cosine/
cp /home/hyu/Digital_Platform/manuals/fig2d_motif_substration/Epigenetics_3TF_mahalanobis/randaug/pca50_variant_scores_* /home/hyu/DeepACE/Preds/D04_deeptfbu/analysis_cosine/
cp /home/hyu/Digital_Platform/manuals/fig2g_motif_substration_cold/Epigenetics_3TF_mahalanobis/randaug/pca50_variant_scores_* /home/hyu/DeepACE/Preds/D04_deeptfbu/analysis_cold/
cp /home/hyu/Digital_Platform/manuals/fig2f_motif_substration_promoterAI/Epigenetics_promoterAI/promoterAI_variant_scores_* /home/hyu/DeepACE/Preds/D04_deeptfbu/analysis_promoterai/
cp /home/hyu/Digital_Platform/manuals/fig2f_motif_substration_evo2/Epigenetics_evo2/evo2_variant_scores_* /home/hyu/DeepACE/Preds/D04_deeptfbu/analysis_evo2/
cp /home/hyu/Digital_Platform/manuals/fig2f_motif_substration_phastcons/* /home/hyu/DeepACE/Preds/D04_deeptfbu/analysis_cons/
mv /home/hyu/DeepACE/Preds/D04_deeptfbu/analysis_cons/gpnmsa_* /home/hyu/DeepACE/Preds/D04_deeptfbu/analysis_gpnmsa/


cp /home/hyu/Digital_Platform/manuals/xfig2e_motif_substration_final/* /home/hyu/DeepACE/Preds/D04_deeptfbu/analysis_united

cp /home/hyu/Figures/DeepACE/Fig2/Fig2g_ELF1_auc_barplot.svg /home/hyu/DeepACE/Figs/F02_variant_effects/F02g_barplot_all_1.svg
cp /home/hyu/Figures/DeepACE/Fig2/Fig2g_HNF1A_auc_barplot.svg /home/hyu/DeepACE/Figs/F02_variant_effects/F02g_barplot_all_2.svg
cp /home/hyu/Figures/DeepACE/Fig2/Fig2g_HNF4A_auc_barplot.svg /home/hyu/DeepACE/Figs/F02_variant_effects/F02g_barplot_all_3.svg

cp -r /home/hyu/Digital_Platform/manuals/fig_dataset/motif_Epigenetics_ELF1* /home/hyu/DeepACE/Preds/D04_deeptfbu
cp -r /home/hyu/Digital_Platform/manuals/fig_dataset/motif_Epigenetics_HNF1A* /home/hyu/DeepACE/Preds/D04_deeptfbu
cp -r /home/hyu/Digital_Platform/manuals/fig_dataset/motif_Epigenetics_HNF4A* /home/hyu/DeepACE/Preds/D04_deeptfbu
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

def plot_pcc_barplot_randaug(df_plot, output_dir, save_name="pcc_barplot.pdf", mode="PCC"):
    
    plt.figure(figsize=( len(df_plot) // 3 + 4, 8))
    ax = sns.barplot(
        data=df_plot,
        x="Group",
        y=mode,
        hue="Algorithm",
        edgecolor="k"
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)

    plt.title(f"{mode} Comparison Across Groups")
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
    print(f"Saved {mode} barplot → {save_path}")
    df_plot.to_csv(f"{output_dir}/{save_name[:-4] + '.csv'}")
    
def plot_pcc_boxplot_randaug(df_plot, output_dir, save_name="pcc_boxplot_summary.pdf", mode="PCC"):

    algo_palette = {"PCA": "#1f77b4", "Evo2": "#ff7f0e", "promoterAI": "#2ca02c"}
    plt.figure(figsize=(15, 5))

    ax = sns.boxplot(
        data=df_plot,
        x="Algorithm", 
        y=mode, 
        width=0.45, 
        showfliers=False, 
        whis=[0, 100],
        palette = "deep"
        # palette=algo_palette
    )

    sns.stripplot(
        data=df_plot, 
        x="Algorithm", 
        y=mode, 
        color="black", 
        alpha=0.45, 
        jitter=True, 
        dodge=False
    )

    # plt.title(f"{mode} Summary Across Algorithms")
    ylocs = ax.get_yticks()
    ylabels = [f'{y:.2f}' for y in ylocs]
    ax.set_yticklabels(ylabels)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=400)
    plt.close()
    
    print(f"Saved {mode} barplot → {save_path}")
    df_plot.to_csv(f"{output_dir}/{save_name[:-4] + '.csv'}")

def calculate_auc_per_motif(scores, effects, quantile_grid=None):

    if quantile_grid is None:
        quantile_grid = np.linspace(0.01, 0.99, 99)

    discover_rates = []

    nan_mask = ~np.isnan(scores)
    scores = scores[nan_mask]
    effects = effects[nan_mask]
    
    if len(scores) < 2:
        return np.nan
    
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

    return auc

def calculate_nonnan_coeff(scores, effects, mode="pearsonr"):
    
    nan_mask = ~np.isnan(scores)
    scores = scores[nan_mask]
    effects = effects[nan_mask]
    
    if len(scores) < 2:
        return np.nan
    elif mode == "pearsonr":
        return pearsonr(scores[nan_mask], effects[nan_mask])[0]
    elif mode == "spearmanr":
        return spearmanr(scores[nan_mask], effects[nan_mask])[0]
    else:
        print("Error Mode Loading!")
        return

'''
Tables Combined
'''

dataset = "Epigenetics"
motif_map = {
    "Epigenetics": ["ELF1", "HNF1A", "HNF4A"] # 
}

output_dir = f"./Preds/D04_deeptfbu/analysis_united"
os.makedirs(output_dir, exist_ok=True)

print(f"Processing dataset: {dataset}")
motif_list = motif_map[dataset]
auc_records = {}

total_pcc_data, total_scc_data, total_auc_data = [], [], []

for motif in motif_list:
    df_deepace_randaug = pd.read_csv(f"./Preds/D04_deeptfbu/analysis_cosine/pca50_variant_scores_{motif}.csv")
    df_deepace_random = pd.read_csv(f"./Preds/D04_deeptfbu/analysis_mahalanobis/pca50_variant_scores_{motif}.csv")
    df_deepace_cold = pd.read_csv(f"./Preds/D04_deeptfbu/analysis_cold/pca50_variant_scores_{motif}.csv")
    df_evo2 = pd.read_csv(f"./Preds/D04_deeptfbu/analysis_evo2/evo2_variant_scores_{motif}.csv")
    df_promoterAI = pd.read_csv(f"./Preds/D04_deeptfbu/analysis_promoterai/promoterAI_variant_scores_{motif}.csv")
    df_phyloP100way = pd.read_csv(f"./Preds/D04_deeptfbu/analysis_cons/phyloP100way_variant_scores_{motif}.csv")
    df_phyloP470way = pd.read_csv(f"./Preds/D04_deeptfbu/analysis_cons/phyloP470way_variant_scores_{motif}.csv")
    df_phastCons100way = pd.read_csv(f"./Preds/D04_deeptfbu/analysis_cons/phastCons100way_variant_scores_{motif}.csv")
    df_phastCons470way = pd.read_csv(f"./Preds/D04_deeptfbu/analysis_cons/phastCons470way_variant_scores_{motif}.csv")
    df_gpnmsa = pd.read_csv(f"./Preds/D04_deeptfbu/analysis_gpnmsa/gpnmsa_variant_scores_{motif}.csv")
    
    df_deepace_randaug = df_deepace_randaug[["scores", "variant_effects"]]
    df_deepace_random = df_deepace_random[["scores", "variant_effects"]]
    df_deepace_cold = df_deepace_cold[["scores", "variant_effects"]]
    df_evo2 = df_evo2[["scores", "variant_effects"]]
    df_promoterAI = df_promoterAI[["scores", "variant_effects"]]
    variant_effects = df_evo2["variant_effects"].to_numpy()
            
    valid_mask = np.isfinite(variant_effects)
    df_evo2 = df_evo2[valid_mask].reset_index(drop=True)
    df_promoterAI = df_promoterAI[valid_mask].reset_index(drop=True)
    df_phyloP100way = df_phyloP100way[valid_mask].reset_index(drop=True)
    df_phyloP470way = df_phyloP470way[valid_mask].reset_index(drop=True)
    df_phastCons100way = df_phastCons100way[valid_mask].reset_index(drop=True)
    df_phastCons470way = df_phastCons470way[valid_mask].reset_index(drop=True)
    df_gpnmsa = df_gpnmsa[valid_mask].reset_index(drop=True)
    variant_effects = variant_effects[valid_mask]
    
    sim_deepace_randaug = df_deepace_randaug["scores"].to_numpy()
    sim_deepace_random = df_deepace_random["scores"].to_numpy()
    sim_deepace_cold = df_deepace_cold["scores"].to_numpy()
    sim_evo2 = df_evo2["scores"].to_numpy()
    sim_promoterAI = df_promoterAI["scores"].to_numpy()
    sim_phyloP100way = df_phyloP100way["scores"].to_numpy()
    sim_phyloP470way = df_phyloP470way["scores"].to_numpy()
    sim_phastCons100way = df_phastCons100way["scores"].to_numpy()
    sim_phastCons470way = df_phastCons470way["scores"].to_numpy()
    sim_gpnmsa = df_gpnmsa["scores"].to_numpy()

    df_alt = pd.read_csv(f"./fig_dataset/motif_Epigenetics_{motif}_alt.csv")
    df_alt = df_alt[valid_mask].reset_index(drop=True)
    df_alt["group_name"] = df_alt["sequence_name"].str.split(r'_-_|_\+_', expand=True)[0]
    group_names_in_order = df_alt["group_name"].drop_duplicates(keep='first')

    pcc_data, scc_data, auc_data = [], [], []
    geo_cnt, aim_cnt = 0, 0

    for group_name in group_names_in_order:
        print(f"group_name: {group_name}")
        
        alt_indices = df_alt[df_alt["group_name"] == group_name].index.tolist()
        sim_deepace_randaug_tmp = sim_deepace_randaug[alt_indices]
        sim_deepace_random_tmp = sim_deepace_random[alt_indices]
        sim_deepace_cold_tmp = sim_deepace_cold[alt_indices]
        sim_evo2_tmp = sim_evo2[alt_indices]
        sim_promoterAI_tmp = sim_promoterAI[alt_indices]
        sim_phyloP100way_tmp = sim_phyloP100way[alt_indices]
        sim_phyloP470way_tmp = sim_phyloP470way[alt_indices]
        sim_phastCons100way_tmp = sim_phastCons100way[alt_indices]
        sim_phastCons470way_tmp = sim_phastCons470way[alt_indices]
        sim_gpnmsa_tmp = sim_gpnmsa[alt_indices]
        variant_effects_tmp = variant_effects[alt_indices]
                
        if len(alt_indices) < 10:
            continue
        if np.max(variant_effects_tmp) - np.min(variant_effects_tmp) < 0.5:
            continue
        
        pcc_deepace_randaug = pearsonr(-sim_deepace_randaug_tmp, variant_effects_tmp)[0]
        pcc_deepace_random = pearsonr(-sim_deepace_random_tmp, variant_effects_tmp)[0]
        pcc_deepace_cold = pearsonr(-sim_deepace_cold_tmp, variant_effects_tmp)[0]
        pcc_evo2 = pearsonr(sim_evo2_tmp, variant_effects_tmp)[0]
        pcc_promoterAI = pearsonr(sim_promoterAI_tmp, variant_effects_tmp)[0]
        pcc_phyloP100way = pearsonr(sim_phyloP100way_tmp, variant_effects_tmp)[0]
        pcc_phyloP470way = calculate_nonnan_coeff(sim_phyloP470way_tmp, variant_effects_tmp, mode="pearsonr")
        pcc_phastCons100way = pearsonr(sim_phastCons100way_tmp, variant_effects_tmp)[0]
        pcc_phastCons470way = calculate_nonnan_coeff(sim_phastCons470way_tmp, variant_effects_tmp, mode="pearsonr")
        pcc_gpnmsa = calculate_nonnan_coeff(sim_gpnmsa_tmp, variant_effects_tmp, mode="pearsonr")
        
        scc_deepace_randaug = spearmanr(-sim_deepace_randaug_tmp, variant_effects_tmp)[0]
        scc_deepace_random = spearmanr(-sim_deepace_random_tmp, variant_effects_tmp)[0]
        scc_deepace_cold = spearmanr(-sim_deepace_cold_tmp, variant_effects_tmp)[0]
        scc_evo2 = spearmanr(sim_evo2_tmp, variant_effects_tmp)[0]
        scc_promoterAI = spearmanr(sim_promoterAI_tmp, variant_effects_tmp)[0]
        scc_phyloP100way = spearmanr(sim_phyloP100way_tmp, variant_effects_tmp)[0]
        scc_phyloP470way = calculate_nonnan_coeff(sim_phyloP470way_tmp, variant_effects_tmp, mode="spearmanr")
        scc_phastCons100way = spearmanr(sim_phastCons100way_tmp, variant_effects_tmp)[0]
        scc_phastCons470way = calculate_nonnan_coeff(sim_phastCons470way_tmp, variant_effects_tmp, mode="spearmanr")
        scc_gpnmsa = calculate_nonnan_coeff(sim_gpnmsa_tmp, variant_effects_tmp, mode="spearmanr")

        auc_deepace_randaug = calculate_auc_per_motif(-sim_deepace_randaug_tmp, variant_effects_tmp)
        auc_deepace_random = calculate_auc_per_motif(-sim_deepace_random_tmp, variant_effects_tmp)
        auc_deepace_cold = calculate_auc_per_motif(-sim_deepace_cold_tmp, variant_effects_tmp)
        auc_evo2 = calculate_auc_per_motif(sim_evo2_tmp, variant_effects_tmp)
        auc_promoterAI = calculate_auc_per_motif(sim_promoterAI_tmp, variant_effects_tmp)
        auc_phyloP100way = calculate_auc_per_motif(sim_phyloP100way_tmp, variant_effects_tmp)
        auc_phyloP470way = calculate_auc_per_motif(sim_phyloP470way_tmp, variant_effects_tmp)
        auc_phastCons100way = calculate_auc_per_motif(sim_phastCons100way_tmp, variant_effects_tmp)
        auc_phastCons470way = calculate_auc_per_motif(sim_phastCons470way_tmp, variant_effects_tmp)
        auc_gpnmsa = calculate_auc_per_motif(sim_gpnmsa_tmp, variant_effects_tmp)
                                
        tmp = group_name.split('_pos_')[1]
        tmp = "_".join(tmp.split("_")[:-2])
        if 'aim' in tmp:
            aim_cnt += 1
            tmp = f"{motif}_aim{aim_cnt}"
        else:
            geo_cnt += 1
            tmp = f"{motif}_geo{geo_cnt}"

        pcc_data.append({"Group": tmp, "Algorithm": "DeepACE-randaug", "PCC": pcc_deepace_randaug, "Name": group_name})
        pcc_data.append({"Group": tmp, "Algorithm": "DeepACE-random", "PCC": pcc_deepace_random, "Name": group_name})
        pcc_data.append({"Group": tmp, "Algorithm": "DeepACE-cold", "PCC": pcc_deepace_cold, "Name": group_name})
        pcc_data.append({"Group": tmp, "Algorithm": "Evo2", "PCC": pcc_evo2, "Name": group_name})
        pcc_data.append({"Group": tmp, "Algorithm": "promoterAI", "PCC": pcc_promoterAI, "Name": group_name})
        pcc_data.append({"Group": tmp, "Algorithm": "phyloP100way", "PCC": pcc_phyloP100way, "Name": group_name})
        pcc_data.append({"Group": tmp, "Algorithm": "phyloP470way", "PCC": pcc_phyloP470way, "Name": group_name})
        pcc_data.append({"Group": tmp, "Algorithm": "phastCons100way", "PCC": pcc_phastCons100way, "Name": group_name})
        pcc_data.append({"Group": tmp, "Algorithm": "phastCons470way", "PCC": pcc_phastCons470way, "Name": group_name})
        pcc_data.append({"Group": tmp, "Algorithm": "gpnmsa", "PCC": pcc_gpnmsa, "Name": group_name})

        scc_data.append({"Group": tmp, "Algorithm": "DeepACE-randaug", "SCC": scc_deepace_randaug, "Name": group_name})
        scc_data.append({"Group": tmp, "Algorithm": "DeepACE-random", "SCC": scc_deepace_random, "Name": group_name})
        scc_data.append({"Group": tmp, "Algorithm": "DeepACE-cold", "SCC": scc_deepace_cold, "Name": group_name})
        scc_data.append({"Group": tmp, "Algorithm": "Evo2", "SCC": scc_evo2, "Name": group_name})
        scc_data.append({"Group": tmp, "Algorithm": "promoterAI", "SCC": scc_promoterAI, "Name": group_name})
        scc_data.append({"Group": tmp, "Algorithm": "phyloP100way", "SCC": scc_phyloP100way, "Name": group_name})
        scc_data.append({"Group": tmp, "Algorithm": "phyloP470way", "SCC": scc_phyloP470way, "Name": group_name})
        scc_data.append({"Group": tmp, "Algorithm": "phastCons100way", "SCC": scc_phastCons100way, "Name": group_name})
        scc_data.append({"Group": tmp, "Algorithm": "phastCons470way", "SCC": scc_phastCons470way, "Name": group_name})
        scc_data.append({"Group": tmp, "Algorithm": "gpnmsa", "SCC": scc_gpnmsa, "Name": group_name})

        auc_data.append({"Group": tmp, "Algorithm": "DeepACE-randaug", "AUC": auc_deepace_randaug, "Name": group_name})
        auc_data.append({"Group": tmp, "Algorithm": "DeepACE-random", "AUC": auc_deepace_random, "Name": group_name})
        auc_data.append({"Group": tmp, "Algorithm": "DeepACE-cold", "AUC": auc_deepace_cold, "Name": group_name})
        auc_data.append({"Group": tmp, "Algorithm": "Evo2", "AUC": auc_evo2, "Name": group_name})
        auc_data.append({"Group": tmp, "Algorithm": "promoterAI", "AUC": auc_promoterAI, "Name": group_name})
        auc_data.append({"Group": tmp, "Algorithm": "phyloP100way", "AUC": auc_phyloP100way, "Name": group_name})
        auc_data.append({"Group": tmp, "Algorithm": "phyloP470way", "AUC": auc_phyloP470way, "Name": group_name})
        auc_data.append({"Group": tmp, "Algorithm": "phastCons100way", "AUC": auc_phastCons100way, "Name": group_name})
        auc_data.append({"Group": tmp, "Algorithm": "phastCons470way", "AUC": auc_phastCons470way, "Name": group_name})
        auc_data.append({"Group": tmp, "Algorithm": "gpnmsa", "AUC": auc_gpnmsa, "Name": group_name})
        
        total_pcc_data.append({"Group": tmp, "Algorithm": "DeepACE-randaug", "PCC": pcc_deepace_randaug, "Name": group_name})
        total_pcc_data.append({"Group": tmp, "Algorithm": "DeepACE-random", "PCC": pcc_deepace_random, "Name": group_name})
        total_pcc_data.append({"Group": tmp, "Algorithm": "DeepACE-cold", "PCC": pcc_deepace_cold, "Name": group_name})
        total_pcc_data.append({"Group": tmp, "Algorithm": "Evo2", "PCC": pcc_evo2, "Name": group_name})
        total_pcc_data.append({"Group": tmp, "Algorithm": "promoterAI", "PCC": pcc_promoterAI, "Name": group_name})
        total_pcc_data.append({"Group": tmp, "Algorithm": "phyloP100way", "PCC": pcc_phyloP100way, "Name": group_name})
        total_pcc_data.append({"Group": tmp, "Algorithm": "phyloP470way", "PCC": pcc_phyloP470way, "Name": group_name})
        total_pcc_data.append({"Group": tmp, "Algorithm": "phastCons100way", "PCC": pcc_phastCons100way, "Name": group_name})
        total_pcc_data.append({"Group": tmp, "Algorithm": "phastCons470way", "PCC": pcc_phastCons470way, "Name": group_name})
        total_pcc_data.append({"Group": tmp, "Algorithm": "gpnmsa", "PCC": pcc_gpnmsa, "Name": group_name})
        
        total_scc_data.append({"Group": tmp, "Algorithm": "DeepACE-randaug", "SCC": scc_deepace_randaug, "Name": group_name})
        total_scc_data.append({"Group": tmp, "Algorithm": "DeepACE-random", "SCC": scc_deepace_random, "Name": group_name})
        total_scc_data.append({"Group": tmp, "Algorithm": "DeepACE-cold", "SCC": scc_deepace_cold, "Name": group_name})
        total_scc_data.append({"Group": tmp, "Algorithm": "Evo2", "SCC": scc_evo2, "Name": group_name})
        total_scc_data.append({"Group": tmp, "Algorithm": "promoterAI", "SCC": scc_promoterAI, "Name": group_name})
        total_scc_data.append({"Group": tmp, "Algorithm": "phyloP100way", "SCC": scc_phyloP100way, "Name": group_name})
        total_scc_data.append({"Group": tmp, "Algorithm": "phyloP470way", "SCC": scc_phyloP470way, "Name": group_name})
        total_scc_data.append({"Group": tmp, "Algorithm": "phastCons100way", "SCC": scc_phastCons100way, "Name": group_name})
        total_scc_data.append({"Group": tmp, "Algorithm": "phastCons470way", "SCC": scc_phastCons470way, "Name": group_name})
        total_scc_data.append({"Group": tmp, "Algorithm": "gpnmsa", "SCC": scc_gpnmsa, "Name": group_name})

        total_auc_data.append({"Group": tmp, "Algorithm": "DeepACE-randaug", "AUC": auc_deepace_randaug, "Name": group_name})
        total_auc_data.append({"Group": tmp, "Algorithm": "DeepACE-random", "AUC": auc_deepace_random, "Name": group_name})
        total_auc_data.append({"Group": tmp, "Algorithm": "DeepACE-cold", "AUC": auc_deepace_cold, "Name": group_name})
        total_auc_data.append({"Group": tmp, "Algorithm": "Evo2", "AUC": auc_evo2, "Name": group_name})
        total_auc_data.append({"Group": tmp, "Algorithm": "promoterAI", "AUC": auc_promoterAI, "Name": group_name})
        total_auc_data.append({"Group": tmp, "Algorithm": "phyloP100way", "AUC": auc_phyloP100way, "Name": group_name})
        total_auc_data.append({"Group": tmp, "Algorithm": "phyloP470way", "AUC": auc_phyloP470way, "Name": group_name})
        total_auc_data.append({"Group": tmp, "Algorithm": "phastCons100way", "AUC": auc_phastCons100way, "Name": group_name})
        total_auc_data.append({"Group": tmp, "Algorithm": "phastCons470way", "AUC": auc_phastCons470way, "Name": group_name})
        total_auc_data.append({"Group": tmp, "Algorithm": "gpnmsa", "AUC": auc_gpnmsa, "Name": group_name})    
        
    df_pcc = pd.DataFrame(pcc_data)
    plot_pcc_barplot_randaug(df_pcc, output_dir, save_name=f"{motif}_pcc_barplot.pdf", mode="PCC")
    plot_pcc_boxplot_randaug(df_pcc, output_dir, save_name=f"{motif}_pcc_boxplot_summary.pdf", mode="PCC")

    df_scc = pd.DataFrame(scc_data)
    plot_pcc_barplot_randaug(df_scc, output_dir, save_name=f"{motif}_scc_barplot.pdf", mode="SCC")
    plot_pcc_boxplot_randaug(df_scc, output_dir, save_name=f"{motif}_scc_boxplot_summary.pdf", mode="SCC")

    df_auc = pd.DataFrame(auc_data)
    plot_pcc_barplot_randaug(df_auc, output_dir, save_name=f"{motif}_auc_barplot.pdf", mode="AUC")
    plot_pcc_boxplot_randaug(df_auc, output_dir, save_name=f"{motif}_auc_boxplot_summary.pdf", mode="AUC")
    
df_total_pcc = pd.DataFrame(total_pcc_data)
df_total_scc = pd.DataFrame(total_scc_data)
df_total_auc = pd.DataFrame(total_auc_data)
plot_pcc_boxplot_randaug(df_total_pcc, output_dir, save_name=f"total_pcc_boxplot_summary.pdf", mode="PCC")
plot_pcc_boxplot_randaug(df_total_scc, output_dir, save_name=f"total_scc_boxplot_summary.pdf", mode="SCC")
plot_pcc_boxplot_randaug(df_total_auc, output_dir, save_name=f"total_auc_boxplot_summary.pdf", mode="AUC")


'''
Ploting Figures
'''

file_path = "./Preds/D04_deeptfbu/analysis_united/total_auc_boxplot_summary.csv"
save_dir = "./Figs/F02_variant_effects"
if not os.path.exists(save_dir): os.makedirs(save_dir)
df = pd.read_csv(file_path)
name_map = {"phyloP100way": "phyloP 100", "phyloP470way": "phyloP 470", "phastCons100way": "phastCons 100", "phastCons470way": "phastCons 470", "gpnmsa": "GPN-MSA"}
df['Algorithm'] = df['Algorithm'].replace(name_map)
df[['TF_Category', 'Sample_ID']] = df['Group'].str.split('_', n=1, expand=True)
palette = {
    "DeepACE-randaug": "#2A4B7C",
    "DeepACE-random": "#4C72B0",
    "DeepACE-cold": "#7694C1",
    "Evo2": "#DD8452",
    "promoterAI": "#55A868",
    "phyloP 100": "#5A3E8C",
    "phyloP 470": "#B39DDB",
    "phastCons 100": "#00A6D6",
    "phastCons 470": "#7FDBFF",
    "GPN-MSA": "#636363"
}
sns.set_theme(style="ticks")
for i, tf in enumerate(['ELF1', 'HNF1A', 'HNF4A']):
    subset = df[df['TF_Category'] == tf].copy()
    if subset.empty: continue
    plt.figure(figsize=(5, 7))
    sns.barplot(data=subset, y='Algorithm', x='AUC', palette=palette, edgecolor='black', errorbar=None, alpha=0.9)
    sns.stripplot(data=subset, y='Algorithm', x='AUC', color='black', size=4, jitter=0.15, linewidth=0.5)
    plt.title(f'{tf}', fontsize=20, pad=8, loc='left', x=-0.2)
    plt.xlabel('AUC Score', fontsize=20)
    plt.ylabel('', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0.4, 1.0)
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"F02g_barplot_all_{i}.svg"), bbox_inches='tight')