
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import numpy as np
import pandas as pd
import os, sys
import random

from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib


''' ---------------- Configuration ---------------- '''

SUPP_ROOT = "./Supps/S13_barplot_robust_randomsample"
METRICS_DIR = f"{SUPP_ROOT}/metrics"  
SCORES_DIR = f"{SUPP_ROOT}/scores"     
RANDOM_SAMPLE_LIST = ["random_sample_1", "random_sample_2", "random_sample_3", "random_sample_4"]
RANDOM_TMPL = "./Preds/D10_random/{sample}/uni_pred.npy"
CLF_PRED_TMPL = "./Preds/D13_clinvar/promoterAI_{dataset}/uni_pred.npy"
CLF_LABEL_TMPL = "./Preds/D13_clinvar/promoterAI_{dataset}_summary.tsv"
COLD_PCA_PATH = "./Preds/D01_screens/pca_model.pkl"
DATASETS = ["clinvar", "cagi5", "gelrna", "mprasat"]
BASELINE_SPECS = {
    "promoterAI":      ("./Preds/D13_clinvar/analysis_promoterai", "promoterAI_variant_scores_{dataset}.csv"),
    "Evo2":            ("./Preds/D13_clinvar/analysis_evo2",       "evo2_variant_scores_{dataset}.csv"),
    "phyloP100way":    ("./Preds/D13_clinvar/analysis_cons",       "phyloP100way_variant_scores_{dataset}.csv"),
    "phyloP470way":    ("./Preds/D13_clinvar/analysis_cons",       "phyloP470way_variant_scores_{dataset}.csv"),
    "phastCons100way": ("./Preds/D13_clinvar/analysis_cons",       "phastCons100way_variant_scores_{dataset}.csv"),
    "phastCons470way": ("./Preds/D13_clinvar/analysis_cons",       "phastCons470way_variant_scores_{dataset}.csv"),
    "gpnmsa":          ("./Preds/D13_clinvar/analysis_gpnmsa",     "gpnmsa_variant_scores_{dataset}.csv"),
}
DEEPACE_ALGOS = ["DeepACE-random", "DeepACE-cold"]
SAVE_FORMATS = ["pdf", "svg", "png"]
deep_palette = sns.color_palette("deep")
new_palette = deep_palette[1:]


def compute_abs_mahalanobis_dists(pred_alt, pred_ref):
    cov = np.cov(pred_ref, rowvar=False)
    cov_inv = np.linalg.pinv(cov)
    total_dists = []
    for x in pred_alt:
        dists = [mahalanobis(x, y, cov_inv) for y in pred_ref]
        dists = np.array(dists)
        total_dists.append(dists.mean())
    total_dists = np.array(total_dists)
    return total_dists

def indicator_function(x, y):
    return [0 if (xi - yi) > 0 else 1 for xi, yi in zip(x, y)]

def mutate_sequence(seq, mutation_rate=0.05):
    bases = ["A", "T", "C", "G"]
    seq = list(seq)
    length = len(seq)
    n_mut = int(length * mutation_rate)
    mutate_indices = random.sample(range(length), n_mut)

    for idx in mutate_indices:
        original_base = seq[idx]
        possible_bases = [b for b in bases if b != original_base]
        seq[idx] = random.choice(possible_bases)
    return "".join(seq)


def load_dataset(dataset):
    combined = np.load(CLF_PRED_TMPL.format(dataset=dataset))
    if dataset == "clinvar":
        df_target = pd.read_csv(CLF_LABEL_TMPL.format(dataset=dataset), sep='\t')
        consequence = df_target['is_pathogenic'].values
        chrom = df_target['chrom'].values
        y = np.where(consequence == True, 0, 1)
        unique_labels, counts = np.unique(y, return_counts=True)
        valid_mask = (y == 0) | (y == 1)

    elif dataset == "cagi5":
        df_target = pd.read_csv(CLF_LABEL_TMPL.format(dataset=dataset), sep='\t')
        consequence = df_target['consequence'].values
        chrom = df_target['chrom'].values
        y = np.where(consequence == 'over', 1, np.where(consequence == 'under', 0, 2))
        unique_labels, counts = np.unique(y, return_counts=True)
        valid_mask = (y == 0) | (y == 1)

    elif dataset == "gelrna":
        df_target = pd.read_csv(CLF_LABEL_TMPL.format(dataset=dataset), sep='\t')
        consequence = df_target['consequence'].values
        chrom = df_target['chrom'].values
        y = np.where(consequence == 'over', 1, np.where(consequence == 'under', 0, 2))
        unique_labels, counts = np.unique(y, return_counts=True)
        valid_mask = (y == 0) | (y == 1)

    elif dataset == "mprasat":
        df_target = pd.read_csv(CLF_LABEL_TMPL.format(dataset=dataset), sep='\t')
        consequence = df_target['consequence'].values
        chrom = df_target['chrom'].values
        y = np.where(consequence == 'over', 1, np.where(consequence == 'under', 0, 2))
        unique_labels, counts = np.unique(y, return_counts=True)
        valid_mask = (y == 0) | (y == 1)

    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    return combined, y, valid_mask


def generate_warm(dataset, combined, y, valid_mask, rand_path, output_dir):
    nonan_mask = np.isfinite(combined).any(axis=0)
    pred_alt = combined[1::2, nonan_mask]
    pred_ref = combined[::2, nonan_mask]
    pred_rand = np.load(rand_path)
    pred_rand = pred_rand[:, nonan_mask]

    combined = np.vstack([pred_alt, pred_ref, pred_rand])
    combined_pca = PCA(n_components=50, random_state=42).fit_transform(combined)
    pred_alt_pca = combined_pca[:len(pred_alt)]
    pred_ref_pca = combined_pca[len(pred_alt):-len(pred_rand)]
    pred_rand_pca = combined_pca[-len(pred_rand):]

    dists_alt = compute_abs_mahalanobis_dists(pred_alt_pca, pred_rand_pca)
    dists_ref = compute_abs_mahalanobis_dists(pred_ref_pca, pred_rand_pca)
    dists_lfc = np.log2( dists_alt / dists_ref )
    indicator_rand = np.array(dists_lfc)[valid_mask]
    y = y[valid_mask]

    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/pca50_variant_scores_{dataset}.csv"
    pd.DataFrame({"scores": indicator_rand, "variant_effects": y}).to_csv(output_path, index=False)
    print(f"Saved scores to {output_path}")


def generate_cold(dataset, combined, y, valid_mask, rand_path, output_dir):
    nonan_mask = np.isfinite(combined).any(axis=0)
    filled_array = np.zeros((combined.shape[0], combined.shape[1]))
    filled_array[:, nonan_mask] = combined[:, nonan_mask]

    pred_alt = filled_array[1::2, :]
    pred_ref = filled_array[::2, :]
    pred_rand = np.load(rand_path)
    filled_rand = np.zeros((pred_rand.shape[0], pred_rand.shape[1]))
    filled_rand[:, nonan_mask] = pred_rand[:, nonan_mask]
    pred_rand = filled_rand

    combined = np.vstack([pred_alt, pred_ref, pred_rand])

    pca = joblib.load(COLD_PCA_PATH)
    combined_pca = pca.transform(combined)
    pred_alt_pca = combined_pca[:len(pred_alt)]
    pred_ref_pca = combined_pca[len(pred_alt):-len(pred_rand)]
    pred_rand_pca = combined_pca[-len(pred_rand):]

    dists_alt = compute_abs_mahalanobis_dists(pred_alt_pca, pred_rand_pca)
    dists_ref = compute_abs_mahalanobis_dists(pred_ref_pca, pred_rand_pca)
    dists_lfc = np.log2( dists_alt / dists_ref )
    indicator_rand = np.array(dists_lfc)[valid_mask]
    y = y[valid_mask]

    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/pca50_variant_scores_{dataset}.csv"
    pd.DataFrame({"scores": indicator_rand, "variant_effects": y}).to_csv(output_path, index=False)
    print(f"Saved scores to {output_path}")

def save_all_formats(output_dir, save_name, bbox_inches=None, dpi=400):
    os.makedirs(output_dir, exist_ok=True)
    for ext in SAVE_FORMATS:
        plt.savefig(os.path.join(output_dir, f"{save_name}.{ext}"), dpi=dpi, bbox_inches=bbox_inches)


def _collect(df_dict_by_draw, motif_list, algorithms, extract, key_name):
    plot_data = []
    for draw, df_dict in df_dict_by_draw.items():
        for motif in motif_list:
            for algo in algorithms:
                val = extract(df_dict, motif, algo)
                if val is None:
                    continue
                plot_data.append({"Motif": motif, "Algorithm": algo, key_name: val, "Draw": draw})
    return pd.DataFrame(plot_data)


''' ---------------- The following 12 plotting functions ----------------------'''

def plot_accuracy_barplot(df_dict_by_draw, motif_list, output_dir, save_name="accuracy_barplot"):
    metrics = ["Accuracy"]
    algorithms = ["DeepACE-random", "DeepACE-cold",
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    def extract(df_dict, motif, algo):
        df = df_dict[algo]
        if motif not in df:
            return None
        df_motif = df[motif]
        if "variant_effects" in df_motif.columns and "pred_effects" in df_motif.columns:
            y_true = df_motif["variant_effects"].values.astype(int)
            y_pred = df_motif["pred_effects"].values.astype(int)
            return accuracy_score(y_true, y_pred)
        return None
    df_plot = _collect(df_dict_by_draw, motif_list, algorithms, extract, "Score")
    plt.figure(figsize=(16, 8))
    ax = sns.barplot(data=df_plot, x="Motif", y="Score", hue="Algorithm", dodge=True, edgecolor="k",
                     palette=new_palette, errorbar=None)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
    # plt.title("Accuracy Comparison Across Algorithms")
    plt.xticks(rotation=0, ha='right')
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    save_all_formats(output_dir, save_name)
    plt.close()
    print(f"Saved accuracy barplot → {output_dir}/{save_name}.")
    df_plot.to_csv(f"{output_dir}/{save_name}.csv", index=False)


def plot_precision_barplot(df_dict_by_draw, motif_list, output_dir, save_name="precision_barplot"):
    metrics = ["Precision"]
    algorithms = ["DeepACE-random", "DeepACE-cold",
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    def extract(df_dict, motif, algo):
        df = df_dict[algo]
        if motif not in df:
            return None
        df_motif = df[motif]
        if "variant_effects" in df_motif.columns and "pred_effects" in df_motif.columns:
            y_true = df_motif["variant_effects"].values.astype(int)
            y_pred = df_motif["pred_effects"].values.astype(int)
            return precision_score(y_true, y_pred)
        return None
    df_plot = _collect(df_dict_by_draw, motif_list, algorithms, extract, "Score")
    plt.figure(figsize=(16, 8))
    ax = sns.barplot(data=df_plot, x="Motif", y="Score", hue="Algorithm", dodge=True, edgecolor="k",
                     palette=new_palette, errorbar=None)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
    # plt.title("Precision Comparison Across Algorithms")
    plt.xticks(rotation=0, ha='right')
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    save_all_formats(output_dir, save_name)
    plt.close()
    print(f"Saved precision barplot → {output_dir}/{save_name}.")
    df_plot.to_csv(f"{output_dir}/{save_name}.csv", index=False)


def plot_recall_barplot(df_dict_by_draw, motif_list, output_dir, save_name="recall_barplot"):
    metrics = ["Recall"]
    algorithms = ["DeepACE-random", "DeepACE-cold",
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    def extract(df_dict, motif, algo):
        df = df_dict[algo]
        if motif not in df:
            return None
        df_motif = df[motif]
        if "variant_effects" in df_motif.columns and "pred_effects" in df_motif.columns:
            y_true = df_motif["variant_effects"].values.astype(int)
            y_pred = df_motif["pred_effects"].values.astype(int)
            return recall_score(y_true, y_pred)
        return None
    df_plot = _collect(df_dict_by_draw, motif_list, algorithms, extract, "Score")
    plt.figure(figsize=(16, 8))
    ax = sns.barplot(data=df_plot, x="Motif", y="Score", hue="Algorithm", dodge=True, edgecolor="k",
                     palette=new_palette, errorbar=None)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
    # plt.title("Recall Comparison Across Algorithms")
    plt.xticks(rotation=0, ha='right')
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    save_all_formats(output_dir, save_name)
    plt.close()
    print(f"Saved recall barplot → {output_dir}/{save_name}.")
    df_plot.to_csv(f"{output_dir}/{save_name}.csv", index=False)

def plot_f1_barplot(df_dict_by_draw, motif_list, output_dir, save_name="f1_barplot"):
    metrics = ["F1"]
    algorithms = ["DeepACE-random", "DeepACE-cold",
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    def extract(df_dict, motif, algo):
        df = df_dict[algo]
        if motif not in df:
            return None
        df_motif = df[motif]
        if "variant_effects" in df_motif.columns and "pred_effects" in df_motif.columns:
            y_true = df_motif["variant_effects"].values.astype(int)
            y_pred = df_motif["pred_effects"].values.astype(int)
            return f1_score(y_true, y_pred)
        return None
    df_plot = _collect(df_dict_by_draw, motif_list, algorithms, extract, "Score")
    plt.figure(figsize=(16, 8))
    ax = sns.barplot(data=df_plot, x="Motif", y="Score", hue="Algorithm", dodge=True, edgecolor="k",
                     palette=new_palette, errorbar=None)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
    # plt.title("F1 Score Comparison Across Algorithms")
    plt.xticks(rotation=0, ha='right')
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    save_all_formats(output_dir, save_name)
    plt.close()
    print(f"Saved F1 barplot → {output_dir}/{save_name}.")
    df_plot.to_csv(f"{output_dir}/{save_name}.csv", index=False)

def plot_accuracy_boxplot_summary(df_dict_by_draw, motif_list, output_dir, save_name="accuracy_boxplot_summary"):
    metrics = ["Accuracy"]
    algorithms = ["DeepACE-random", "DeepACE-cold",
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    algo_palette = {"PCA": "#1f77b4", "Evo2": "#ff7f0e", "promoterAI": "#2ca02c"}
    def extract(df_dict, motif, algo):
        df = df_dict[algo].get(motif, pd.DataFrame()).dropna(subset=["pred_effects", "variant_effects"])
        if not df.empty:
            y_true = df["variant_effects"].values.astype(int)
            y_pred = df["pred_effects"].values.astype(int)
            return accuracy_score(y_true, y_pred)
        return None
    df_plot = _collect(df_dict_by_draw, motif_list, algorithms, extract, "Accuracy")
    plt.figure(figsize=(12, 5))
    ax = sns.boxplot(data=df_plot, x="Algorithm", y="Accuracy", width=0.45, palette=new_palette, showfliers=False, whis=[0, 100])
    sns.stripplot(data=df_plot, x="Algorithm", y="Accuracy", color="black", alpha=0.45, jitter=True, dodge=False)

    # plt.title("Accuracy Boxplot Summary Across Algorithms")
    plt.xlabel("")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_all_formats(output_dir, save_name)
    plt.close()
    print(f"Saved accuracy boxplot summary → {output_dir}/{save_name}.")
    df_plot.to_csv(f"{output_dir}/{save_name}.csv", index=False)

def plot_precision_boxplot_summary(df_dict_by_draw, motif_list, output_dir, save_name="precision_boxplot_summary"):
    metrics = ["Precision"]
    algorithms = ["DeepACE-random", "DeepACE-cold",
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    algo_palette = {"PCA": "#1f77b4", "Evo2": "#ff7f0e", "promoterAI": "#2ca02c"}
    def extract(df_dict, motif, algo):
        df = df_dict[algo].get(motif, pd.DataFrame()).dropna(subset=["pred_effects", "variant_effects"])
        if not df.empty:
            y_true = df["variant_effects"].values.astype(int)
            y_pred = df["pred_effects"].values.astype(int)
            return precision_score(y_true, y_pred)
        return None
    df_plot = _collect(df_dict_by_draw, motif_list, algorithms, extract, "Precision")
    plt.figure(figsize=(12, 5))
    ax = sns.boxplot(data=df_plot, x="Algorithm", y="Precision", width=0.45, palette=new_palette, showfliers=False, whis=[0, 100])
    sns.stripplot(data=df_plot, x="Algorithm", y="Precision", color="black", alpha=0.45, jitter=True, dodge=False)
    # plt.title("Precision Boxplot Summary Across Algorithms")
    plt.xlabel("")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_all_formats(output_dir, save_name)
    plt.close()
    print(f"Saved precision boxplot summary → {output_dir}/{save_name}.")
    df_plot.to_csv(f"{output_dir}/{save_name}.csv", index=False)

def plot_recall_boxplot_summary(df_dict_by_draw, motif_list, output_dir, save_name="recall_boxplot_summary"):
    metrics = ["Recall"]
    algorithms = ["DeepACE-random", "DeepACE-cold",
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    algo_palette = {"PCA": "#1f77b4", "Evo2": "#ff7f0e", "promoterAI": "#2ca02c"}
    def extract(df_dict, motif, algo):
        df = df_dict[algo].get(motif, pd.DataFrame()).dropna(subset=["pred_effects", "variant_effects"])
        if not df.empty:
            y_true = df["variant_effects"].values.astype(int)
            y_pred = df["pred_effects"].values.astype(int)
            return recall_score(y_true, y_pred)
        return None
    df_plot = _collect(df_dict_by_draw, motif_list, algorithms, extract, "Recall")
    plt.figure(figsize=(12, 5))
    ax = sns.boxplot(data=df_plot, x="Algorithm", y="Recall", width=0.45, palette=new_palette, showfliers=False, whis=[0, 100])
    sns.stripplot(data=df_plot, x="Algorithm", y="Recall", color="black", alpha=0.45, jitter=True, dodge=False)
    # plt.title("Recall Boxplot Summary Across Algorithms")
    plt.xlabel("")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_all_formats(output_dir, save_name)
    plt.close()
    print(f"Saved recall boxplot summary → {output_dir}/{save_name}.")
    df_plot.to_csv(f"{output_dir}/{save_name}.csv", index=False)

def plot_f1_boxplot_summary(df_dict_by_draw, motif_list, output_dir, save_name="f1_boxplot_summary"):
    metrics = ["F1"]
    algorithms = ["DeepACE-random", "DeepACE-cold",
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    algo_palette = {"PCA": "#1f77b4", "Evo2": "#ff7f0e", "promoterAI": "#2ca02c"}
    def extract(df_dict, motif, algo):
        df = df_dict[algo].get(motif, pd.DataFrame()).dropna(subset=["pred_effects", "variant_effects"])
        if not df.empty:
            y_true = df["variant_effects"].values.astype(int)
            y_pred = df["pred_effects"].values.astype(int)
            return f1_score(y_true, y_pred)
        return None
    df_plot = _collect(df_dict_by_draw, motif_list, algorithms, extract, "F1")
    plt.figure(figsize=(12, 5))
    ax = sns.boxplot(data=df_plot, x="Algorithm", y="F1", width=0.45, palette=new_palette, showfliers=False, whis=[0, 100])
    sns.stripplot(data=df_plot, x="Algorithm", y="F1", color="black", alpha=0.45, jitter=True, dodge=False)
    # plt.title("F1 Boxplot Summary Across Algorithms")
    plt.xlabel("")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_all_formats(output_dir, save_name)
    plt.close()
    print(f"Saved F1 boxplot summary → {output_dir}/{save_name}.")
    df_plot.to_csv(f"{output_dir}/{save_name}.csv", index=False)


def plot_prauc_barplot(df_dict_by_draw, motif_list, output_dir, save_name="prauc_barplot"):
    metrics = ["PRAUC"]
    algorithms = ["DeepACE-random", "DeepACE-cold",
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    def extract(df_dict, motif, algo):
        df = df_dict[algo]
        if motif not in df:
            return None
        df_motif = df[motif]
        if "variant_effects" in df_motif.columns and "scores" in df_motif.columns:
            y_true = df_motif["variant_effects"].values.astype(int)
            y_scores = df_motif["scores"].values
            nan_mask = ~np.isnan(y_scores)
            return average_precision_score(y_true[nan_mask], y_scores[nan_mask])
        return None
    df_plot = _collect(df_dict_by_draw, motif_list, algorithms, extract, "Score")
    plt.figure(figsize=(16, 8))
    ax = sns.barplot(data=df_plot, x="Motif", y="Score", hue="Algorithm", dodge=True, edgecolor="k",
                     palette=new_palette, errorbar=None)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
    # plt.title("PRAUC Comparison Across Algorithms")
    plt.xticks(rotation=0, ha='right')
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    save_all_formats(output_dir, save_name)
    plt.close()
    print(f"Saved PRAUC barplot → {output_dir}/{save_name}.")
    df_plot.to_csv(f"{output_dir}/{save_name}.csv", index=False)

def plot_prauc_boxplot_summary(df_dict_by_draw, motif_list, output_dir, save_name="prauc_boxplot_summary"):
    algorithms = ["DeepACE-random", "DeepACE-cold",
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    algo_palette = {"PCA": "#1f77b4", "Evo2": "#ff7f0e", "promoterAI": "#2ca02c"}
    def extract(df_dict, motif, algo):
        df = df_dict[algo].get(motif, pd.DataFrame()).dropna(subset=["scores", "variant_effects"])
        if not df.empty:
            y_true = df["variant_effects"].values.astype(int)
            y_scores = df["scores"].values
            nan_mask = ~np.isnan(y_scores)
            return average_precision_score(y_true[nan_mask], y_scores[nan_mask])
        return None
    df_plot = _collect(df_dict_by_draw, motif_list, algorithms, extract, "PRAUC")
    plt.figure(figsize=(12, 5))
    ax = sns.boxplot(data=df_plot, x="Algorithm", y="PRAUC", width=0.45, palette=new_palette, showfliers=False, whis=[0, 100])
    sns.stripplot(data=df_plot, x="Algorithm", y="PRAUC", color="black", alpha=0.45, jitter=True, dodge=False)
    # plt.title("PRAUC Boxplot Summary Across Algorithms")
    plt.xlabel("")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_all_formats(output_dir, save_name)
    plt.close()
    print(f"Saved PRAUC boxplot summary → {output_dir}/{save_name}.")
    df_plot.to_csv(f"{output_dir}/{save_name}.csv", index=False)


def plot_rocauc_barplot(df_dict_by_draw, motif_list, output_dir, save_name="rocauc_barplot"):
    metrics = ["ROCAUC"]
    algorithms = ["DeepACE-random", "DeepACE-cold",
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    def extract(df_dict, motif, algo):
        df = df_dict[algo]
        if motif not in df:
            return None
        df_motif = df[motif]
        if "variant_effects" in df_motif.columns and "scores" in df_motif.columns:
            y_true = df_motif["variant_effects"].values.astype(int)
            y_scores = df_motif["scores"].values
            nan_mask = ~np.isnan(y_scores)
            return roc_auc_score(y_true[nan_mask], y_scores[nan_mask])
        return None
    df_plot = _collect(df_dict_by_draw, motif_list, algorithms, extract, "Score")
    plt.figure(figsize=(16, 8))
    ax = sns.barplot(data=df_plot, x="Motif", y="Score", hue="Algorithm", dodge=True, edgecolor="k",
                     palette=new_palette, errorbar=None)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
    # plt.title("ROCAUC Comparison Across Algorithms")
    plt.xticks(rotation=0, ha='right')
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    save_all_formats(output_dir, save_name)
    plt.close()
    print(f"Saved ROCAUC barplot → {output_dir}/{save_name}.")
    df_plot.to_csv(f"{output_dir}/{save_name}.csv", index=False)


def plot_rocauc_boxplot_summary(df_dict_by_draw, motif_list, output_dir, save_name="rocauc_boxplot_summary"):
    algorithms = ["DeepACE-random", "DeepACE-cold",
                  "Evo2", "promoterAI",
                  "phyloP100way", "phyloP470way",
                  "phastCons100way", "phastCons470way", "gpnmsa"]
    algo_palette = {"PCA": "#1f77b4", "Evo2": "#ff7f0e", "promoterAI": "#2ca02c"}
    def extract(df_dict, motif, algo):
        df = df_dict[algo].get(motif, pd.DataFrame()).dropna(subset=["scores", "variant_effects"])
        if not df.empty:
            y_true = df["variant_effects"].values.astype(int)
            y_scores = df["scores"].values
            nan_mask = ~np.isnan(y_scores)
            return roc_auc_score(y_true[nan_mask], y_scores[nan_mask])
        return None
    df_plot = _collect(df_dict_by_draw, motif_list, algorithms, extract, "ROCAUC")
    plt.figure(figsize=(12, 5))
    ax = sns.boxplot(data=df_plot, x="Algorithm", y="ROCAUC", width=0.45, palette=new_palette, showfliers=False, whis=[0, 100])
    sns.stripplot(data=df_plot, x="Algorithm", y="ROCAUC", color="black", alpha=0.45, jitter=True, dodge=False)
    # plt.title("ROCAUC Boxplot Summary Across Algorithms")
    plt.xlabel("")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_all_formats(output_dir, save_name)
    plt.close()
    print(f"Saved ROCAUC boxplot summary → {output_dir}/{save_name}.")
    df_plot.to_csv(f"{output_dir}/{save_name}.csv", index=False)


''' ================= Part A: The two DeepACE columns for each draw ================= '''

for d in (SUPP_ROOT, METRICS_DIR, SCORES_DIR):
    os.makedirs(d, exist_ok=True)

GENERATORS = [(generate_warm, "analysis_mahalanobis"), (generate_cold, "analysis_cold")]

for draw in RANDOM_SAMPLE_LIST:
    rand_path = RANDOM_TMPL.format(sample=draw)
    print(f"\n{'='*70}\n[draw] Random Sample Dataset = {draw}\n{'='*70}", flush=True)
    for dataset in DATASETS:
        pending = [(gen, sub) for gen, sub in GENERATORS
                   if not os.path.exists(f"{SCORES_DIR}/{draw}/{sub}/pca50_variant_scores_{dataset}.csv")]
        if not pending:
            print(f"[reuse] {draw}/{dataset}: score tables already on disk, skipped", flush=True)
            continue
        print(f"Processing dataset: {dataset}", flush=True)
        combined, y, valid_mask = load_dataset(dataset)
        for gen, sub in pending:
            gen(dataset, combined, y, valid_mask, rand_path, output_dir=f"{SCORES_DIR}/{draw}/{sub}")


''' ================= Part B: For Table Combined, one df_dict per draw ================= '''

print(f"\n{'='*70}\nTables Combined\n{'='*70}", flush=True)

df_dict_by_draw = {}

for draw in RANDOM_SAMPLE_LIST:
    draw_dir = f"{SCORES_DIR}/{draw}"
    output_dir = draw_dir

    df_dict = {"DeepACE-random": {}, "DeepACE-cold": {},
               "Evo2": {}, "promoterAI": {},
               "phyloP100way": {}, "phyloP470way": {},
               "phastCons100way": {}, "phastCons470way": {},
               "gpnmsa": {}}

    for dataset in DATASETS:
        df_deepace_random = pd.read_csv(f"{draw_dir}/analysis_mahalanobis/pca50_variant_scores_{dataset}.csv")
        df_deepace_cold = pd.read_csv(f"{draw_dir}/analysis_cold/pca50_variant_scores_{dataset}.csv")
        df_promoterAI = pd.read_csv(f"{BASELINE_SPECS['promoterAI'][0]}/{BASELINE_SPECS['promoterAI'][1].format(dataset=dataset)}")
        df_evo2 = pd.read_csv(f"{BASELINE_SPECS['Evo2'][0]}/{BASELINE_SPECS['Evo2'][1].format(dataset=dataset)}")
        df_phyloP100way = pd.read_csv(f"{BASELINE_SPECS['phyloP100way'][0]}/{BASELINE_SPECS['phyloP100way'][1].format(dataset=dataset)}")
        df_phyloP470way = pd.read_csv(f"{BASELINE_SPECS['phyloP470way'][0]}/{BASELINE_SPECS['phyloP470way'][1].format(dataset=dataset)}")
        df_phastCons100way = pd.read_csv(f"{BASELINE_SPECS['phastCons100way'][0]}/{BASELINE_SPECS['phastCons100way'][1].format(dataset=dataset)}")
        df_phastCons470way = pd.read_csv(f"{BASELINE_SPECS['phastCons470way'][0]}/{BASELINE_SPECS['phastCons470way'][1].format(dataset=dataset)}")
        df_gpnmsa = pd.read_csv(f"{BASELINE_SPECS['gpnmsa'][0]}/{BASELINE_SPECS['gpnmsa'][1].format(dataset=dataset)}")

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

    df_dict_by_draw[draw] = df_dict
    print(f"[{draw}] tables combined", flush=True)


''' ================= Part C: F02f local figures, mean only ================= '''

plot_prauc_barplot(df_dict_by_draw, DATASETS, METRICS_DIR, save_name="prauc_barplot")
plot_prauc_boxplot_summary(df_dict_by_draw, DATASETS, METRICS_DIR, save_name="prauc_boxplot_summary")
plot_rocauc_barplot(df_dict_by_draw, DATASETS, METRICS_DIR, save_name="rocauc_barplot")
plot_rocauc_boxplot_summary(df_dict_by_draw, DATASETS, METRICS_DIR, save_name="rocauc_boxplot_summary")

plot_accuracy_barplot(df_dict_by_draw, DATASETS, METRICS_DIR, save_name=f"accuracy_barplot")
plot_precision_barplot(df_dict_by_draw, DATASETS, METRICS_DIR, save_name=f"precision_barplot")
plot_recall_barplot(df_dict_by_draw, DATASETS, METRICS_DIR, save_name=f"recall_barplot")
plot_f1_barplot(df_dict_by_draw, DATASETS, METRICS_DIR, save_name=f"f1_barplot")

plot_accuracy_boxplot_summary(df_dict_by_draw, DATASETS, METRICS_DIR, save_name=f"accuracy_boxplot_summary")
plot_precision_boxplot_summary(df_dict_by_draw, DATASETS, METRICS_DIR, save_name=f"precision_boxplot_summary")
plot_recall_boxplot_summary(df_dict_by_draw, DATASETS, METRICS_DIR, save_name=f"recall_boxplot_summary")
plot_f1_boxplot_summary(df_dict_by_draw, DATASETS, METRICS_DIR, save_name=f"f1_boxplot_summary")


''' ================= Part D: F02f AUROC panels, mean only ================= '''

file_path = f"{METRICS_DIR}/rocauc_barplot.csv"
save_dir = f"{SUPP_ROOT}/"
df = pd.read_csv(file_path)
name_map = {"phyloP100way": "phyloP 100", "phyloP470way": "phyloP 470", "phastCons100way": "phastCons 100", "phastCons470way": "phastCons 470", "gpnmsa": "GPN-MSA"}
df['Algorithm'] = df['Algorithm'].replace(name_map)

os.makedirs(SUPP_ROOT, exist_ok=True)
df.to_csv(f"{SUPP_ROOT}/S13_barplot_all.csv", index=False)
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
    sns.barplot(data=subset, y='Algorithm', x='Score', palette=palette, edgecolor='black', alpha=0.9,
                errorbar=None)
    plt.title(f'{display_name}', fontsize=20, pad=8, loc='left', x=-0.2)
    plt.xlabel('AUROC Score', fontsize=20)
    plt.ylabel('', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0.4, 1.0)
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    for ext in SAVE_FORMATS:
        plt.savefig(os.path.join(save_dir, f"S13_barplot_all_{i}_{ds}.{ext}"), bbox_inches='tight')
    plt.close()
    print(f"Saved S13_barplot_all_{i}_{ds}.[{'|'.join(SAVE_FORMATS)}]")


''' ================= Final Statistical Evaluation  ================= '''

rocauc = pd.read_csv(f"{METRICS_DIR}/rocauc_barplot.csv")
piv = rocauc.pivot_table(index=["Motif", "Algorithm"], columns="Draw", values="Score")
piv["spread"] = piv.max(axis=1) - piv.min(axis=1)
mean_spread = piv["spread"].groupby(level="Algorithm").mean().sort_values(ascending=False)
summary_path = f"{SCORES_DIR}/S13_rocauc_spread_summary.csv"
mean_spread.to_frame("mean_spread_across_datasets").to_csv(summary_path)

print("\n" + "=" * 70)
print(f"ROCAUC range across {len(RANDOM_SAMPLE_LIST)} random sample groups (averaged by algorithm)")
print("=" * 70)
print(mean_spread.round(5).to_string())
print(f"\n-> {summary_path}")
base_spread = mean_spread.drop(index=DEEPACE_ALGOS, errors="ignore")
ds_spread = mean_spread.reindex(DEEPACE_ALGOS)
if base_spread.max() > 1e-12:
    print("\nNote: The range for baseline algorithms is not 0 - they should be independent of the random background, indicating other random sources have been mixed in, which needs investigation.")
else:
    print("\nBaseline algorithm ranges are all 0 (as expected: they do not depend on random background samples).")
print(f"DeepACE two columns: " + ", ".join(f"{a}={v:.5f}" for a, v in ds_spread.items()))
print("=" * 70)



''' ================= Final Statistical Evaluation: combined ================= '''

records = []
for ds in DATASETS:
    for algo in DEEPACE_ALGOS:
        vals = [rocauc[(rocauc["Motif"] == ds) & (rocauc["Algorithm"] == algo) &
                       (rocauc["Draw"] == draw)]["Score"].mean()
                for draw in RANDOM_SAMPLE_LIST]
        records.append({"Motif": ds, "Algorithm": algo,
                        "Mean_AUROC": float(np.mean(vals)),
                        **{f"{draw}_AUROC": v for draw, v in zip(RANDOM_SAMPLE_LIST, vals)}})
auroc_table = pd.DataFrame(records)

print("\n" + "=" * 78)
print(f"DeepACE mean AUROC (averaged over {len(RANDOM_SAMPLE_LIST)} random background sample groups)")
print("=" * 78)
last = None
for _, row in auroc_table.iterrows():
    if row["Motif"] != last:
        print(f"\n[{row['Motif']}]")
        last = row["Motif"]
    draws = ", ".join(f"{v:.4f}" for v in row[[f"{d}_AUROC" for d in RANDOM_SAMPLE_LIST]])
    print(f"    {row['Algorithm']:<18s} mean = {row['Mean_AUROC']:.4f}    (draws: {draws})")

print("\n" + "-" * 78)
print("Overall mean across all 4 datasets:")
for algo in DEEPACE_ALGOS:
    print(f"    {algo:<18s} {auroc_table[auroc_table['Algorithm'] == algo]['Mean_AUROC'].mean():.4f}")
print("=" * 78)

auroc_path = f"{SCORES_DIR}/S13_deepace_mean_auroc.csv"
auroc_table.round(4).to_csv(auroc_path, index=False)
print(f"-> {auroc_path}")


''' ================= DeepACE mean across the 4 draws ================= '''

draw_cols = [f"{draw}_AUROC" for draw in RANDOM_SAMPLE_LIST]
per_draw = auroc_table.groupby("Algorithm")[draw_cols].mean().loc[DEEPACE_ALGOS]
per_draw.columns = RANDOM_SAMPLE_LIST
mean_row_stats = pd.DataFrame({"Mean": per_draw.mean(axis=1),
                               "SD": per_draw.std(axis=1, ddof=1)}).join(per_draw)

mean_row_path = f"{SCORES_DIR}/S13_deepace_mean_across_draws.csv"
mean_row_stats.to_csv(mean_row_path)

print("\n" + "=" * 70)
print(f"DeepACE Mean AUROC across the {len(RANDOM_SAMPLE_LIST)} random background sample groups")
print("=" * 70)
print(mean_row_stats.round(4).to_string())
print(f"\n-> {mean_row_path}")
print("=" * 70)
