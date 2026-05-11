'''
/home/hyu/Digital_Platform/manuals/xfig2d_classification_final.py

cp /home/hyu/Digital_Platform/manuals/figs8_classification_pca/pca50_variant_scores_* /home/hyu/DeepACE/Preds/D13_clinvar/analysis_mahalanobis/
cp /home/hyu/Digital_Platform/manuals/fig2g_classification_cold/pca50_variant_scores_* /home/hyu/DeepACE/Preds/D13_clinvar/analysis_cold/
cp /home/hyu/Digital_Platform/manuals/figs8_classification_promoterAI/promoterAI_variant_scores_* /home/hyu/DeepACE/Preds/D13_clinvar/analysis_promoterai/
cp /home/hyu/Digital_Platform/manuals/figs8_classification_evo2/evo2_variant_scores_* /home/hyu/DeepACE/Preds/D13_clinvar/analysis_evo2/
cp /home/hyu/Digital_Platform/manuals/figs8_classification_phastcons/* /home/hyu/DeepACE/Preds/D13_clinvar/analysis_cons/
mv /home/hyu/DeepACE/Preds/D13_clinvar/analysis_cons/gpnmsa_* /home/hyu/DeepACE/Preds/D13_clinvar/analysis_gpnmsa/

cp /home/hyu/Digital_Platform/manuals/xfig2d_classification_final/* /home/hyu/DeepACE/Preds/D13_clinvar/analysis_united
cp /home/hyu/Figures/DeepACE/Fig2/Fig2f_cagi5_auc_barplot.svg /home/hyu/DeepACE/Figs/F02_variant_effects/F02f_barplot_all_1.svg
cp /home/hyu/Figures/DeepACE/Fig2/Fig2f_gelrna_auc_barplot.svg /home/hyu/DeepACE/Figs/F02_variant_effects/F02f_barplot_all_2.svg
cp /home/hyu/Figures/DeepACE/Fig2/Fig2f_mprasat_auc_barplot.svg /home/hyu/DeepACE/Figs/F02_variant_effects/F02f_barplot_all_3.svg
cp /home/hyu/Figures/DeepACE/Fig2/Fig2f_clinvar_auc_barplot.svg /home/hyu/DeepACE/Figs/F02_variant_effects/F02f_barplot_all_4.svg

cp -r /home/hyu/Digital_Platform/manuals/fig_dataset/point_classification/promoterAI_* /home/hyu/DeepACE/Preds/D13_clinvar
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

deep_palette = sns.color_palette("deep")
new_palette = deep_palette[1:]

def plot_accuracy_barplot(df_dict, motif_list, output_dir, save_name="accuracy_barplot.pdf"):
    metrics = ["Accuracy"]
    algorithms = ["DeepACE-random", "DeepACE-cold", 
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    plot_data = []
    for motif in motif_list:
        for algo in algorithms:
            df = df_dict[algo]
            if motif in df:
                df_motif = df[motif]
                if "variant_effects" in df_motif.columns and "pred_effects" in df_motif.columns:
                    y_true = df_motif["variant_effects"].values.astype(int)
                    y_pred = df_motif["pred_effects"].values.astype(int)
                    accuracy = accuracy_score(y_true, y_pred)
                    plot_data.append({"Motif": motif, "Algorithm": algo, "Metric": "Accuracy", "Score": accuracy})
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(16, 8))
    ax = sns.barplot(data=df_plot, x="Motif", y="Score", hue="Algorithm", dodge=True, edgecolor="k", palette=new_palette)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
    # plt.title("Accuracy Comparison Across Algorithms")
    plt.xticks(rotation=0, ha='right')
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Saved accuracy barplot → {save_path}")
    df_plot.to_csv(f"{output_dir}/{save_name[:-4] + '.csv'}")


def plot_precision_barplot(df_dict, motif_list, output_dir, save_name="precision_barplot.pdf"):
    metrics = ["Precision"]
    algorithms = ["DeepACE-random", "DeepACE-cold", 
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    plot_data = []
    for motif in motif_list:
        for algo in algorithms:
            df = df_dict[algo]
            if motif in df:
                df_motif = df[motif]
                if "variant_effects" in df_motif.columns and "pred_effects" in df_motif.columns:
                    y_true = df_motif["variant_effects"].values.astype(int)
                    y_pred = df_motif["pred_effects"].values.astype(int)
                    precision = precision_score(y_true, y_pred)
                    plot_data.append({"Motif": motif, "Algorithm": algo, "Metric": "Precision", "Score": precision})
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(16, 8))
    ax = sns.barplot(data=df_plot, x="Motif", y="Score", hue="Algorithm", dodge=True, edgecolor="k", palette=new_palette)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
    # plt.title("Precision Comparison Across Algorithms")
    plt.xticks(rotation=0, ha='right')
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Saved precision barplot → {save_path}")
    df_plot.to_csv(f"{output_dir}/{save_name[:-4] + '.csv'}")

def plot_recall_barplot(df_dict, motif_list, output_dir, save_name="recall_barplot.pdf"):
    metrics = ["Recall"]
    algorithms = ["DeepACE-random", "DeepACE-cold", 
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    plot_data = []
    for motif in motif_list:
        for algo in algorithms:
            df = df_dict[algo]
            if motif in df:
                df_motif = df[motif]
                if "variant_effects" in df_motif.columns and "pred_effects" in df_motif.columns:
                    y_true = df_motif["variant_effects"].values.astype(int)
                    y_pred = df_motif["pred_effects"].values.astype(int)
                    recall = recall_score(y_true, y_pred)
                    plot_data.append({"Motif": motif, "Algorithm": algo, "Metric": "Recall", "Score": recall})
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(16, 8))
    ax = sns.barplot(data=df_plot, x="Motif", y="Score", hue="Algorithm", dodge=True, edgecolor="k", palette=new_palette)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
    # plt.title("Recall Comparison Across Algorithms")
    plt.xticks(rotation=0, ha='right')
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Saved recall barplot → {save_path}")
    df_plot.to_csv(f"{output_dir}/{save_name[:-4] + '.csv'}")
    
def plot_f1_barplot(df_dict, motif_list, output_dir, save_name="f1_barplot.pdf"):
    metrics = ["F1"]
    algorithms = ["DeepACE-random", "DeepACE-cold", 
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    plot_data = []
    for motif in motif_list:
        for algo in algorithms:
            df = df_dict[algo]
            if motif in df:
                df_motif = df[motif]
                if "variant_effects" in df_motif.columns and "pred_effects" in df_motif.columns:
                    y_true = df_motif["variant_effects"].values.astype(int)
                    y_pred = df_motif["pred_effects"].values.astype(int)
                    f1 = f1_score(y_true, y_pred)
                    plot_data.append({"Motif": motif, "Algorithm": algo, "Metric": "F1", "Score": f1})
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(16, 8))
    ax = sns.barplot(data=df_plot, x="Motif", y="Score", hue="Algorithm", dodge=True, edgecolor="k", palette=new_palette)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
    # plt.title("F1 Score Comparison Across Algorithms")
    plt.xticks(rotation=0, ha='right')
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Saved F1 barplot → {save_path}")
    df_plot.to_csv(f"{output_dir}/{save_name[:-4] + '.csv'}")
    
def plot_accuracy_boxplot_summary(df_dict, motif_list, output_dir, save_name="accuracy_boxplot_summary.pdf"):
    metrics = ["Accuracy"]
    algorithms = ["DeepACE-random", "DeepACE-cold", 
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    algo_palette = {"PCA": "#1f77b4", "Evo2": "#ff7f0e", "promoterAI": "#2ca02c"}
    plot_data = []
    for motif in motif_list:
        for algo in algorithms:
            df = df_dict[algo].get(motif, pd.DataFrame()).dropna(subset=["pred_effects", "variant_effects"])
            if not df.empty:
                y_true = df["variant_effects"].values.astype(int)
                y_pred = df["pred_effects"].values.astype(int)
                accuracy = accuracy_score(y_true, y_pred)
                plot_data.append({"Motif": motif, "Algorithm": algo, "Accuracy": accuracy})
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(12, 5))
    ax = sns.boxplot(data=df_plot, x="Algorithm", y="Accuracy", width=0.45, palette=new_palette, showfliers=False, whis=[0, 100])
    sns.stripplot(data=df_plot, x="Algorithm", y="Accuracy", color="black", alpha=0.45, jitter=True, dodge=False)

    # plt.title("Accuracy Boxplot Summary Across Algorithms")
    plt.xlabel("")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Saved accuracy boxplot summary → {save_path}")

def plot_precision_boxplot_summary(df_dict, motif_list, output_dir, save_name="precision_boxplot_summary.pdf"):
    metrics = ["Precision"]
    algorithms = ["DeepACE-random", "DeepACE-cold", 
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    algo_palette = {"PCA": "#1f77b4", "Evo2": "#ff7f0e", "promoterAI": "#2ca02c"}
    plot_data = []
    for motif in motif_list:
        for algo in algorithms:
            df = df_dict[algo].get(motif, pd.DataFrame()).dropna(subset=["pred_effects", "variant_effects"])
            if not df.empty:
                y_true = df["variant_effects"].values.astype(int)
                y_pred = df["pred_effects"].values.astype(int)
                precision = precision_score(y_true, y_pred)
                plot_data.append({"Motif": motif, "Algorithm": algo, "Precision": precision})
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(12, 5))
    ax = sns.boxplot(data=df_plot, x="Algorithm", y="Precision", width=0.45, palette=new_palette, showfliers=False, whis=[0, 100])
    sns.stripplot(data=df_plot, x="Algorithm", y="Precision", color="black", alpha=0.45, jitter=True, dodge=False)
    # plt.title("Precision Boxplot Summary Across Algorithms")
    plt.xlabel("")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Saved precision boxplot summary → {save_path}")

def plot_recall_boxplot_summary(df_dict, motif_list, output_dir, save_name="recall_boxplot_summary.pdf"):
    metrics = ["Recall"]
    algorithms = ["DeepACE-random", "DeepACE-cold", 
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    algo_palette = {"PCA": "#1f77b4", "Evo2": "#ff7f0e", "promoterAI": "#2ca02c"}
    plot_data = []
    for motif in motif_list:
        for algo in algorithms:
            df = df_dict[algo].get(motif, pd.DataFrame()).dropna(subset=["pred_effects", "variant_effects"])
            if not df.empty:
                y_true = df["variant_effects"].values.astype(int)
                y_pred = df["pred_effects"].values.astype(int)
                recall = recall_score(y_true, y_pred)
                plot_data.append({"Motif": motif, "Algorithm": algo, "Recall": recall})
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(12, 5))
    ax = sns.boxplot(data=df_plot, x="Algorithm", y="Recall", width=0.45, palette=new_palette, showfliers=False, whis=[0, 100])
    sns.stripplot(data=df_plot, x="Algorithm", y="Recall", color="black", alpha=0.45, jitter=True, dodge=False)
    # plt.title("Recall Boxplot Summary Across Algorithms")
    plt.xlabel("")
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Saved recall boxplot summary → {save_path}")

def plot_f1_boxplot_summary(df_dict, motif_list, output_dir, save_name="f1_boxplot_summary.pdf"):
    metrics = ["F1"]
    algorithms = ["DeepACE-random", "DeepACE-cold", 
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    algo_palette = {"PCA": "#1f77b4", "Evo2": "#ff7f0e", "promoterAI": "#2ca02c"}
    plot_data = []
    for motif in motif_list:
        for algo in algorithms:
            df = df_dict[algo].get(motif, pd.DataFrame()).dropna(subset=["pred_effects", "variant_effects"])
            if not df.empty:
                y_true = df["variant_effects"].values.astype(int)
                y_pred = df["pred_effects"].values.astype(int)
                f1 = f1_score(y_true, y_pred)
                plot_data.append({"Motif": motif, "Algorithm": algo, "F1": f1})
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(12, 5))
    ax = sns.boxplot(data=df_plot, x="Algorithm", y="F1", width=0.45, palette=new_palette, showfliers=False, whis=[0, 100])
    sns.stripplot(data=df_plot, x="Algorithm", y="F1", color="black", alpha=0.45, jitter=True, dodge=False)
    # plt.title("F1 Boxplot Summary Across Algorithms")
    plt.xlabel("")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Saved F1 boxplot summary → {save_path}")


def plot_prauc_barplot(df_dict, motif_list, output_dir, save_name="prauc_barplot.pdf"):
    metrics = ["PRAUC"]
    algorithms = ["DeepACE-random", "DeepACE-cold", 
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    plot_data = []
    for motif in motif_list:
        for algo in algorithms:
            df = df_dict[algo]
            if motif in df:
                df_motif = df[motif]
                if "variant_effects" in df_motif.columns and "scores" in df_motif.columns:
                    y_true = df_motif["variant_effects"].values.astype(int)
                    y_scores = df_motif["scores"].values
                    nan_mask = ~np.isnan(y_scores)
                    pr_auc = average_precision_score(y_true[nan_mask], y_scores[nan_mask])
                    plot_data.append({"Motif": motif, "Algorithm": algo, "Metric": "PRAUC", "Score": pr_auc})
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(16, 8))
    ax = sns.barplot(data=df_plot, x="Motif", y="Score", hue="Algorithm", dodge=True, edgecolor="k", palette=new_palette)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
    # plt.title("PRAUC Comparison Across Algorithms")
    plt.xticks(rotation=0, ha='right')
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Saved PRAUC barplot → {save_path}")
    df_plot.to_csv(f"{output_dir}/{save_name[:-4] + '.csv'}")

def plot_prauc_boxplot_summary(df_dict, motif_list, output_dir, save_name="prauc_boxplot_summary.pdf"):
    algorithms = ["DeepACE-random", "DeepACE-cold", 
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    algo_palette = {"PCA": "#1f77b4", "Evo2": "#ff7f0e", "promoterAI": "#2ca02c"}
    plot_data = []
    for motif in motif_list:
        for algo in algorithms:
            df = df_dict[algo].get(motif, pd.DataFrame()).dropna(subset=["scores", "variant_effects"])
            if not df.empty:
                y_true = df["variant_effects"].values.astype(int)
                y_scores = df["scores"].values
                nan_mask = ~np.isnan(y_scores)
                pr_auc = average_precision_score(y_true[nan_mask], y_scores[nan_mask])
                plot_data.append({"Motif": motif, "Algorithm": algo, "PRAUC": pr_auc})
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(12, 5))
    ax = sns.boxplot(data=df_plot, x="Algorithm", y="PRAUC", width=0.45, palette=new_palette, showfliers=False, whis=[0, 100])
    sns.stripplot(data=df_plot, x="Algorithm", y="PRAUC", color="black", alpha=0.45, jitter=True, dodge=False)
    # plt.title("PRAUC Boxplot Summary Across Algorithms")
    plt.xlabel("")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Saved PRAUC boxplot summary → {save_path}")
    df_plot.to_csv(f"{output_dir}/{save_name[:-4] + '.csv'}")


def plot_rocauc_barplot(df_dict, motif_list, output_dir, save_name="rocauc_barplot.pdf"):
    metrics = ["ROCAUC"]
    algorithms = ["DeepACE-random", "DeepACE-cold", 
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    plot_data = []
    for motif in motif_list:
        for algo in algorithms:
            df = df_dict[algo]
            if motif in df:
                df_motif = df[motif]
                if "variant_effects" in df_motif.columns and "scores" in df_motif.columns:
                    y_true = df_motif["variant_effects"].values.astype(int)
                    y_scores = df_motif["scores"].values
                    nan_mask = ~np.isnan(y_scores)
                    roc_auc = roc_auc_score(y_true[nan_mask], y_scores[nan_mask])
                    plot_data.append({"Motif": motif, "Algorithm": algo, "Metric": "ROCAUC", "Score": roc_auc})
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(16, 8))
    ax = sns.barplot(data=df_plot, x="Motif", y="Score", hue="Algorithm", dodge=True, edgecolor="k", palette=new_palette)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
    # plt.title("ROCAUC Comparison Across Algorithms")
    plt.xticks(rotation=0, ha='right')
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Saved ROCAUC barplot → {save_path}")
    df_plot.to_csv(f"{output_dir}/{save_name[:-4] + '.csv'}")


def plot_rocauc_boxplot_summary(df_dict, motif_list, output_dir, save_name="rocauc_boxplot_summary.pdf"):
    algorithms = ["DeepACE-random", "DeepACE-cold", 
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    algo_palette = {"PCA": "#1f77b4", "Evo2": "#ff7f0e", "promoterAI": "#2ca02c"}
    plot_data = []
    for motif in motif_list:
        for algo in algorithms:
            df = df_dict[algo].get(motif, pd.DataFrame()).dropna(subset=["scores", "variant_effects"])
            if not df.empty:
                y_true = df["variant_effects"].values.astype(int)
                y_scores = df["scores"].values
                nan_mask = ~np.isnan(y_scores)
                roc_auc = roc_auc_score(y_true[nan_mask], y_scores[nan_mask])
                plot_data.append({"Motif": motif, "Algorithm": algo, "ROCAUC": roc_auc})
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(12, 5))
    ax = sns.boxplot(data=df_plot, x="Algorithm", y="ROCAUC", width=0.45, palette=new_palette, showfliers=False, whis=[0, 100])
    sns.stripplot(data=df_plot, x="Algorithm", y="ROCAUC", color="black", alpha=0.45, jitter=True, dodge=False)
    # plt.title("ROCAUC Boxplot Summary Across Algorithms")
    plt.xlabel("")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"Saved ROCAUC boxplot summary → {save_path}")
    df_plot.to_csv(f"{output_dir}/{save_name[:-4] + '.csv'}")


'''
Tables Combined
'''

datasets = ["clinvar", "cagi5", "gelrna", "mprasat"]
output_dir = "./Preds/D13_clinvar/analysis_united"
os.makedirs(output_dir, exist_ok=True)

df_dict = {"DeepACE-random": {}, "DeepACE-cold": {},
           "Evo2": {}, "promoterAI": {},
           "phyloP100way": {}, "phyloP470way": {},
           "phastCons100way": {}, "phastCons470way": {},
           "gpnmsa": {}}

for dataset in datasets:
    print(f"Processing dataset: {dataset}")
    
    df_deepace_random = pd.read_csv(f"./Preds/D13_clinvar/analysis_mahalanobis/pca50_variant_scores_{dataset}.csv")
    df_deepace_cold = pd.read_csv(f"./Preds/D13_clinvar/analysis_cold/pca50_variant_scores_{dataset}.csv")
    df_promoterAI = pd.read_csv(f"./Preds/D13_clinvar/analysis_promoterai/promoterAI_variant_scores_{dataset}.csv")
    df_evo2 = pd.read_csv(f"./Preds/D13_clinvar/analysis_evo2/evo2_variant_scores_{dataset}.csv")
    df_phyloP100way = pd.read_csv(f"./Preds/D13_clinvar/analysis_cons/phyloP100way_variant_scores_{dataset}.csv")
    df_phyloP470way = pd.read_csv(f"./Preds/D13_clinvar/analysis_cons/phyloP470way_variant_scores_{dataset}.csv")
    df_phastCons100way = pd.read_csv(f"./Preds/D13_clinvar/analysis_cons/phastCons100way_variant_scores_{dataset}.csv")
    df_phastCons470way = pd.read_csv(f"./Preds/D13_clinvar/analysis_cons/phastCons470way_variant_scores_{dataset}.csv")
    df_gpnmsa = pd.read_csv(f"./Preds/D13_clinvar/analysis_gpnmsa/gpnmsa_variant_scores_{dataset}.csv")
    
    df_deepace_random = df_deepace_random[["scores", "variant_effects"]]
    df_deepace_cold = df_deepace_cold[["scores", "variant_effects"]]
    df_evo2 = df_evo2[["scores", "variant_effects"]]
    df_promoterAI = df_promoterAI[["scores", "variant_effects"]]

    df_deepace_random["pred_effects"] = (df_deepace_random["scores"] >= 0).astype(int)
    df_deepace_cold["pred_effects"] = (df_deepace_cold["scores"] >= 0).astype(int)    
    df_evo2["pred_effects"] = (df_evo2["scores"] >= 0).astype(int)
    df_promoterAI["pred_effects"] = (df_promoterAI["scores"] >= 0).astype(int)
    df_phyloP100way["pred_effects"] = (df_phyloP100way["scores"] >= -0.5).astype(int)
    df_phyloP470way["pred_effects"] = (df_phyloP470way["scores"] >= -0.5).astype(int)
    df_phastCons100way["pred_effects"] = (df_phastCons100way["scores"] >= -0.5).astype(int)
    df_phastCons470way["pred_effects"] = (df_phastCons470way["scores"] >= -0.5).astype(int)
    df_gpnmsa["pred_effects"] = (df_gpnmsa["scores"] >= -2).astype(int)
    
    
    motif = dataset
    df_dict["DeepACE-random"][motif] = df_deepace_random
    df_dict["DeepACE-cold"][motif] = df_deepace_cold
    df_dict["Evo2"][motif] = df_evo2
    df_dict["promoterAI"][motif] = df_promoterAI
    df_dict["phyloP100way"][motif] = df_phyloP100way
    df_dict["phyloP470way"][motif] = df_phyloP470way
    df_dict["phastCons100way"][motif] = df_phastCons100way
    df_dict["phastCons470way"][motif] = df_phastCons470way
    df_dict["gpnmsa"][motif] = df_gpnmsa
    
plot_prauc_barplot(df_dict, datasets, output_dir, save_name="prauc_barplot.pdf")
plot_prauc_boxplot_summary(df_dict, datasets, output_dir, save_name="prauc_boxplot_summary.pdf")
plot_rocauc_barplot(df_dict, datasets, output_dir, save_name="rocauc_barplot.pdf")
plot_rocauc_boxplot_summary(df_dict, datasets, output_dir, save_name="rocauc_boxplot_summary.pdf")
                            
plot_accuracy_barplot(df_dict, datasets, output_dir, save_name=f"accuracy_barplot.pdf")
plot_precision_barplot(df_dict, datasets, output_dir, save_name=f"precision_barplot.pdf")
plot_recall_barplot(df_dict, datasets, output_dir, save_name=f"recall_barplot.pdf")
plot_f1_barplot(df_dict, datasets, output_dir, save_name=f"f1_barplot.pdf")

plot_accuracy_boxplot_summary(df_dict, datasets, output_dir, save_name=f"accuracy_boxplot_summary.pdf")
plot_precision_boxplot_summary(df_dict, datasets, output_dir, save_name=f"precision_boxplot_summary.pdf")
plot_recall_boxplot_summary(df_dict, datasets, output_dir, save_name=f"recall_boxplot_summary.pdf")
plot_f1_boxplot_summary(df_dict, datasets, output_dir, save_name=f"f1_boxplot_summary.pdf")


'''
Ploting Figures
'''


file_path = "./Preds/D13_clinvar/analysis_united/rocauc_barplot.csv"
save_dir = "./Figs/F02_variant_effects/"
df = pd.read_csv(file_path)
name_map = {"phyloP100way": "phyloP 100", "phyloP470way": "phyloP 470", "phastCons100way": "phastCons 100", "phastCons470way": "phastCons 470", "gpnmsa": "GPN-MSA"}
df['Algorithm'] = df['Algorithm'].replace(name_map)
motif_name_map = {
    "clinvar": "ClinVar",
    "gelrna": "GEL RNA-seq",
    "cagi5": "CAGI5",
    "mprasat": "MPRASat"
}
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
datasets = df['Motif'].unique()
for i, ds in enumerate(datasets):
    display_name = motif_name_map.get(ds, ds)
    subset = df[df['Motif'] == ds].copy()
    plt.figure(figsize=(5, 7))
    sns.barplot(data=subset, y='Algorithm', x='Score', palette=palette, edgecolor='black', alpha=0.9)
    plt.title(f'{display_name}', fontsize=20, pad=8, loc='left', x=-0.2)
    plt.xlabel('AUROC Score', fontsize=20)
    plt.ylabel('', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0.4, 1.0)
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"F02f_barplot_all_{i}.svg"), bbox_inches='tight')