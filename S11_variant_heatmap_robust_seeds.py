
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import os, sys
import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA

import joblib
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


''' ---------------- 配置 ---------------- '''

SUPP_ROOT = "./Supps/S11_variant_heatmap_robust_seeds"
SAVE_FORMATS = ["pdf", "png", "svg"]


def _strip_img_ext(path):
    for ext in SAVE_FORMATS + ["eps", "jpg", "jpeg", "tif", "tiff"]:
        if path.lower().endswith("." + ext):
            return path[: -(len(ext) + 1)]
    return path


def save_all_formats(target, path, **kwargs):
    stem = _strip_img_ext(str(path))
    d = os.path.dirname(stem)
    if d:
        os.makedirs(d, exist_ok=True)
    for ext in SAVE_FORMATS:
        target.savefig(f"{stem}.{ext}", **kwargs)
    return f"{stem}.{SAVE_FORMATS[0]}"

REPLICATES = [
    ("random_sample_2", "random_sample_2", 0),
    ("random_sample_3", "random_sample_3", 1),
    ("random_sample_4", "random_sample_4", 2),
]
N_DRAWS = len(REPLICATES)
D10_DIR_TMPL = "./Preds/D10_random/{sample}/uni_pred.npy"
RANDAUG_ROOT = "/home/hyu/DeepACE/Preds/D14_extra_point_randaug"
RANDAUG_DIR_TMPL = RANDAUG_ROOT + "/point_MPRABase_{motif}_randaug_seed43-44-45"
RANDAUG_BLOCKS_CSV = RANDAUG_ROOT + "/S11_randaug_fasta_blocks.csv"
COLD_PCA_PATH = "./Preds/D01_screens/pca_model.pkl"
PCA_RANDOM_STATE = 42
MOTIF_LIST = ["TERT", "HBG1", "LDLR", "F9", "GP1BA", "IRF4", "IRF6", "PKLR", "ZFAND3", "SORT1",
              "HBB", "UC88", "MYC_rs6983267", "RET", "TCF7L2"]
BASELINE_SPECS = {
    "Evo2":            ("./Preds/D05_mprabase/analysis_evo2",        "evo2_variant_scores_{motif}.csv"),
    "promoterAI":      ("./Preds/D05_mprabase/analysis_promoterai",  "promoterAI_variant_scores_{motif}.csv"),
    "phyloP100way":    ("./Preds/D05_mprabase/analysis_cons",        "phyloP100way_variant_scores_{motif}.csv"),
    "phyloP470way":    ("./Preds/D05_mprabase/analysis_cons",        "phyloP470way_variant_scores_{motif}.csv"),
    "phastCons100way": ("./Preds/D05_mprabase/analysis_cons",        "phastCons100way_variant_scores_{motif}.csv"),
    "phastCons470way": ("./Preds/D05_mprabase/analysis_cons",        "phastCons470way_variant_scores_{motif}.csv"),
    "gpnmsa":          ("./Preds/D05_mprabase/analysis_gpnmsa",      "gpnmsa_variant_scores_{motif}.csv"),
}
DEEPACE_ALGOS = ["DeepACE-randaug", "DeepACE-random", "DeepACE-cold"]
ALGORITHMS = ["DeepACE-randaug", "DeepACE-random", "DeepACE-cold",
              "Evo2", "promoterAI",
              "phyloP100way", "phyloP470way",
              "phastCons100way", "phastCons470way", "gpnmsa"]
DEEPACE_SPECS = {
    "DeepACE-randaug": ("./Preds/D05_mprabase/analysis_cosine",      "pca50_variant_scores_{motif}.csv"),
    "DeepACE-random":  ("./Preds/D05_mprabase/analysis_mahalanobis", "pca50_variant_scores_{motif}.csv"),
    "DeepACE-cold":    ("./Preds/D05_mprabase/analysis_cold",        "pca50_variant_scores_{motif}.csv"),
}
DEEPACE_COL_WIDTH = 1.8     # "0.69±0.01" needs ~1.8x the width of "0.69", the padding then matches
BASELINE_ALGOS = [algo for algo in ALGORITHMS if algo not in DEEPACE_ALGOS]
FIG_SIZE = (10 * (len(BASELINE_ALGOS) + len(DEEPACE_ALGOS) * DEEPACE_COL_WIDTH) / len(ALGORITHMS), 8)


''' ---------------- Similarity Calculation ---------------- '''

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


''' ---------------- Quantile-based AUC calculation ---------------- '''

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
    save_path = save_all_formats(plt, os.path.join(output_dir, save_name), dpi=400)
    plt.close()
    print(f"Saved AUC barplot → {save_path}  [{','.join(SAVE_FORMATS)}]")
    df_plot.to_csv(f"{output_dir}/{os.path.splitext(save_name)[0] + '.csv'}")


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
    save_path = save_all_formats(plt, os.path.join(output_dir, save_name), dpi=400)
    plt.close()
    print(f"Saved PCC barplot → {save_path}  [{','.join(SAVE_FORMATS)}]")
    df_plot.to_csv(f"{output_dir}/{os.path.splitext(save_name)[0] + '.csv'}")

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
    save_path = save_all_formats(plt, os.path.join(output_dir, save_name), dpi=400)
    plt.close()
    print(f"Saved SCC barplot → {save_path}  [{','.join(SAVE_FORMATS)}]")
    df_plot.to_csv(f"{output_dir}/{os.path.splitext(save_name)[0] + '.csv'}")


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
    save_path = save_all_formats(plt, os.path.join(output_dir, save_name), dpi=400)
    plt.close()
    print(f"Saved AUC boxplot → {save_path}  [{','.join(SAVE_FORMATS)}]")


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
    save_path = save_all_formats(plt, os.path.join(output_dir, save_name), dpi=400)
    plt.close()
    print(f"Saved PCC boxplot → {save_path}  [{','.join(SAVE_FORMATS)}]")


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
    save_path = save_all_formats(plt, os.path.join(output_dir, save_name), dpi=400)
    plt.close()
    print(f"Saved SCC boxplot → {save_path}  [{','.join(SAVE_FORMATS)}]")


''' ================= Deciphering of paths ================= '''

os.makedirs(SUPP_ROOT, exist_ok=True)
_randaug_cache = {}
_blocks_df = None

def randaug_pred(motif):
    if motif not in _randaug_cache:
        path = f"{RANDAUG_DIR_TMPL.format(motif=motif)}/uni_pred.npy"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Lack of randaug prediction: {path}")
        _randaug_cache[motif] = np.load(path)[:-1]
    return _randaug_cache[motif]


def randaug_block_range(motif, block_idx):
    global _blocks_df
    if _blocks_df is None and os.path.exists(RANDAUG_BLOCKS_CSV):
        _blocks_df = pd.read_csv(RANDAUG_BLOCKS_CSV)
    if _blocks_df is not None:
        sub = _blocks_df[(_blocks_df["motif"] == motif) & (_blocks_df["block"] == block_idx)]
        if len(sub) == 1:
            return int(sub["row_start"].iloc[0]), int(sub["row_end"].iloc[0])
    n_rows = len(randaug_pred(motif))
    if n_rows % N_DRAWS != 0:
        raise ValueError(f"{motif}: randaug N_ROWS={n_rows} can not be devided by N_DRAWS={N_DRAWS}")
    size = n_rows // N_DRAWS
    return block_idx * size, (block_idx + 1) * size

_probe_motif = MOTIF_LIST[0]
_probe_rows = [randaug_block_range(_probe_motif, i) for i in range(N_DRAWS)]
if len(set(_probe_rows)) != N_DRAWS:
    raise ValueError(f"randaug blocks have overlapping row ranges: {_probe_rows}")
_n_rows = len(randaug_pred(_probe_motif))
if _probe_rows[-1][1] != _n_rows:
    raise ValueError(f"randaug blocks do not cover all {_n_rows} rows: {_probe_rows}")
print(f"randaug independent augmentation probe passed (motif={_probe_motif}): {_n_rows} rows -> {N_DRAWS} blocks {_probe_rows}")
print(f"  n={_probe_rows[0][1] - _probe_rows[0][0]} per block (same as original image), three blocks are i.i.d. samples")



''' ================= Reading Baselines for calculation ================= '''

baseline_dict = {algo: {} for algo in BASELINE_SPECS}
for motif in MOTIF_LIST:
    for algo, (algo_dir, fname_tmpl) in BASELINE_SPECS.items():
        df = pd.read_csv(f"{algo_dir}/{fname_tmpl.format(motif=motif)}")
        baseline_dict[algo][motif] = df[["scores", "variant_effects"]]
    print(f"[baseline] loaded {motif} x {len(BASELINE_SPECS)} algos", flush=True)


''' ================= Random Sampling ================= '''

all_auc = {}

for draw_idx, (label, random_sample, randaug_block) in enumerate(REPLICATES):
    draw_dir = f"{SUPP_ROOT}/{label}"
    os.makedirs(draw_dir, exist_ok=True)
    rand_path = D10_DIR_TMPL.format(sample=random_sample)
    ra_start, ra_end = randaug_block_range(MOTIF_LIST[0], randaug_block)

    print(f"\n{'='*70}\n[draw {draw_idx}] label={label}  random_sample={random_sample}  "
          f"randaug_block={randaug_block} (rows {ra_start}:{ra_end})\n{'='*70}", flush=True)

    ''' ---- Part 1: (PCA50 cosine / mahalanobis) ---- '''

    metrics = ["cosine", "mahalanobis"]
    datasets = ["MPRABase"]

    for dataset in datasets:
        for metric in metrics:
            print(f"Processing dataset: {dataset}, metric: {metric}, draw: {random_sample}", flush=True)
            output_dir = f"{draw_dir}/analysis_{metric}"
            os.makedirs(output_dir, exist_ok=True)
            all_similarities, all_groups, all_sources = [], [], []
            motif_list = MOTIF_LIST
            for motif in motif_list:
                out_path = f"{output_dir}/pca50_variant_scores_{motif}.csv"
                if os.path.exists(out_path):
                    print(f"  [reuse] {out_path}", flush=True)
                    continue
                print(f"  Processing motif/background: {motif}", flush=True)
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
                    _s, _e = randaug_block_range(motif, randaug_block)
                    pred_rand = randaug_pred(motif)[_s:_e]
                    pred_rand = pred_rand[:, valid_mask]
                elif metric == "mahalanobis":
                    pred_rand = np.load(rand_path)
                    pred_rand = pred_rand[:, valid_mask]

                combined = np.vstack([pred_alt, pred_ref, pred_rand])
                combined_pca = PCA(n_components=50, random_state=PCA_RANDOM_STATE).fit_transform(combined)
                pred_alt_pca = combined_pca[:len(pred_alt)]
                pred_ref_pca = combined_pca[len(pred_alt):-len(pred_rand)]
                pred_rand_pca = combined_pca[-len(pred_rand):]

                sample_data = np.vstack([pred_ref_pca, pred_alt_pca, pred_rand_pca])
                sample_labels = ['Pseudo'] * len(pred_ref_pca) + ['Real'] * len(pred_alt_pca) + ['Rand'] * len(pred_rand_pca)
                similarity_all = compute_sample_similarity(pred_alt_pca, pred_rand_pca, metric=metric, mode="batch")
                pd.DataFrame({"scores": similarity_all, "variant_effects": variant_effects}).to_csv(f"{output_dir}/pca50_variant_scores_{motif}.csv")

    ''' ---- Part 1b: (PCA50 cold) ---- '''

    metrics = ["mahalanobis"]

    for dataset in datasets:
        for metric in metrics:
            print(f"Processing dataset: {dataset}, metric: cold, draw: {random_sample}", flush=True)
            output_dir = f"{draw_dir}/analysis_cold"
            os.makedirs(output_dir, exist_ok=True)
            all_similarities, all_groups, all_sources = [], [], []
            motif_list = MOTIF_LIST
            for motif in motif_list:
                out_path = f"{output_dir}/pca50_variant_scores_{motif}.csv"
                if os.path.exists(out_path):
                    print(f"  [reuse] {out_path}", flush=True)
                    continue
                print(f"  Processing motif/background: {motif} (cold)", flush=True)
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
                    _s, _e = randaug_block_range(motif, randaug_block)
                    pred_rand = randaug_pred(motif)[_s:_e]
                    pred_rand = pred_rand[:, valid_mask]
                elif metric == "mahalanobis":
                    pred_rand = np.load(rand_path)
                    pred_rand = pred_rand[:, valid_mask]

                combined = np.vstack([pred_alt, pred_ref, pred_rand])
                pca = joblib.load(COLD_PCA_PATH)
                combined_pca = pca.transform(combined)
                pred_alt_pca = combined_pca[:len(pred_alt)]
                pred_ref_pca = combined_pca[len(pred_alt):-len(pred_rand)]
                pred_rand_pca = combined_pca[-len(pred_rand):]

                sample_data = np.vstack([pred_ref_pca, pred_alt_pca, pred_rand_pca])
                sample_labels = ['Pseudo'] * len(pred_ref_pca) + ['Real'] * len(pred_alt_pca) + ['Rand'] * len(pred_rand_pca)
                similarity_all = compute_sample_similarity(pred_alt_pca, pred_rand_pca, metric=metric, mode="batch")
                pd.DataFrame({"scores": similarity_all, "variant_effects": variant_effects}).to_csv(f"{output_dir}/pca50_variant_scores_{motif}.csv")

    ''' ---- Part 2: Ploting local Figures ---- '''

    print(f"[{random_sample}] combining tables and plotting", flush=True)

    for motif in MOTIF_LIST:
        df_deepace_randaug = pd.read_csv(f"{draw_dir}/analysis_cosine/pca50_variant_scores_{motif}.csv")
        df_deepace_random = pd.read_csv(f"{draw_dir}/analysis_mahalanobis/pca50_variant_scores_{motif}.csv")
        df_deepace_cold = pd.read_csv(f"{draw_dir}/analysis_cold/pca50_variant_scores_{motif}.csv")
        df_evo2 = baseline_dict["Evo2"][motif]
        df_promoterAI = baseline_dict["promoterAI"][motif]
        df_phyloP100way = baseline_dict["phyloP100way"][motif]
        df_phyloP470way = baseline_dict["phyloP470way"][motif]
        df_phastCons100way = baseline_dict["phastCons100way"][motif]
        df_phastCons470way = baseline_dict["phastCons470way"][motif]
        df_gpnmsa = baseline_dict["gpnmsa"][motif]

        df_deepace_randaug = df_deepace_randaug[["scores", "variant_effects"]]
        df_deepace_randaug["scores"] = -df_deepace_randaug["scores"]
        df_deepace_random = df_deepace_random[["scores", "variant_effects"]]
        df_deepace_random["scores"] = -df_deepace_random["scores"]
        df_deepace_cold = df_deepace_cold[["scores", "variant_effects"]]
        df_deepace_cold["scores"] = -df_deepace_cold["scores"]

        df_evo2 = df_evo2[["scores", "variant_effects"]]
        df_promoterAI = df_promoterAI[["scores", "variant_effects"]]

        df_dict = {"DeepACE-randaug": {}, "DeepACE-random": {}, "DeepACE-cold": {},
                   "Evo2": {}, "promoterAI": {},
                   "phyloP100way": {}, "phyloP470way": {},
                   "phastCons100way": {}, "phastCons470way": {}, "gpnmsa": {}}
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

        if motif == MOTIF_LIST[0]:
            auc_records = {}
        auc_dict = calculate_auc_per_motif(motif, df_dict)
        auc_records[motif] = auc_dict

        if motif == MOTIF_LIST[0]:
            df_dict_all = {algo: {} for algo in ALGORITHMS}
        for algo in ALGORITHMS:
            df_dict_all[algo][motif] = df_dict[algo][motif]

    all_auc[label] = auc_records

    plot_auc_barplot(auc_records, draw_dir, save_name="all_motifs_auc_barplot.pdf")
    plot_pcc_barplot(df_dict=df_dict_all, motif_list=MOTIF_LIST, output_dir=draw_dir, save_name="all_motifs_pcc_barplot.pdf")
    plot_scc_barplot(df_dict=df_dict_all, motif_list=MOTIF_LIST, output_dir=draw_dir, save_name="all_motifs_scc_barplot.pdf")
    plot_auc_boxplot_summary(auc_records, draw_dir, save_name="all_motifs_auc_boxplot_summary.pdf")
    plot_pcc_boxplot_summary(df_dict_all, MOTIF_LIST, draw_dir, save_name="all_motifs_pcc_boxplot_summary.pdf")
    plot_scc_boxplot_summary(df_dict_all, MOTIF_LIST, draw_dir, save_name="all_motifs_scc_boxplot_summary.pdf")

    print(f"[{label}] done -> {draw_dir}  [{','.join(SAVE_FORMATS)}]", flush=True)


''' ================= Final Statistical Evaluation ================= '''

summary = {tag: pd.DataFrame(rec).mean(axis=1) for tag, rec in all_auc.items()}
summary_df = pd.DataFrame(summary)
summary_df["spread"] = summary_df.max(axis=1) - summary_df.min(axis=1)
summary_df = summary_df.sort_values("spread", ascending=False)
summary_path = f"{SUPP_ROOT}/S11_auc_summary_across_draws.csv"
summary_df.to_csv(summary_path)

print("\n" + "=" * 70)
print("Mean AUC across independent random samples (spread = range of three columns)")
print("=" * 70)
print(summary_df.round(4).to_string())
print(f"\n-> {summary_path}")
ds = summary_df.loc[DEEPACE_ALGOS, "spread"]
if ds.max() < 1e-6:
    print("\nNote: The range for all three DeepACE versions is 0 — the three plots are still identical.")
    print("      In this case, honestly report 'the pipeline is insensitive to background sampling', do not artificially create differences.")
else:
    print(f"\nCross-sampling range for the three DeepACE versions: "
          + ", ".join(f"{a}={v:.4f}" for a, v in ds.items()))
print("=" * 70)


''' ================= Merged Clustermap ================= '''

original_auc = {}
for motif in MOTIF_LIST:
    df_dict = {}
    for algo, (algo_dir, fname_tmpl) in DEEPACE_SPECS.items():
        df = pd.read_csv(f"{algo_dir}/{fname_tmpl.format(motif=motif)}")
        df = df[["scores", "variant_effects"]]
        df["scores"] = -df["scores"]
        df_dict[algo] = {motif: df}
    original_auc[motif] = calculate_auc_per_motif(motif, df_dict)
    print(f"[original] F02e6 DeepACE aucs for {motif}", flush=True)

run_labels = ["F02e6_original"] + [label for label, _, _ in REPLICATES]
run_records = [original_auc] + [all_auc[label] for label, _, _ in REPLICATES]
stack = np.stack([pd.DataFrame(rec).loc[DEEPACE_ALGOS, MOTIF_LIST].to_numpy() for rec in run_records])
row_names = [motif.replace("MYC_rs6983267", "MYC") for motif in MOTIF_LIST]
deepace_mean = pd.DataFrame(stack.mean(axis=0), index=DEEPACE_ALGOS, columns=row_names)
deepace_sd = pd.DataFrame(stack.std(axis=0, ddof=1), index=DEEPACE_ALGOS, columns=row_names)

per_draw_mean = stack.mean(axis=2)
mean_row_stats = pd.DataFrame(per_draw_mean, index=run_labels, columns=DEEPACE_ALGOS).T
mean_row_stats["Mean"] = mean_row_stats.mean(axis=1)
mean_row_stats["SD"] = mean_row_stats[run_labels].std(axis=1, ddof=1)
mean_row_stats = mean_row_stats[["Mean", "SD"] + run_labels]


def relabel_cells(g, labels, algos):
    cols = list(g.data2d.columns)
    rows = list(g.data2d.index)
    for t in g.ax_heatmap.texts:
        x, y = t.get_position()
        col, row = int(round(x - .5)), int(round(y - .5))
        if cols[col] in algos:
            t.set_text(labels.loc[rows[row], cols[col]])


def widen_columns(g, algos, width):
    ax = g.ax_heatmap
    cols = list(g.data2d.columns)
    edges = np.concatenate([[0.], np.cumsum([width if c in algos else 1. for c in cols])])
    centers = (edges[:-1] + edges[1:]) / 2
    remap = lambda x: np.interp(x, np.arange(len(cols)) + .5, centers)

    mesh = ax.collections[0]    # stays in place, the colorbar is bound to it
    mesh.set_visible(False)
    ax.pcolormesh(edges, np.arange(len(g.data2d) + 1), g.data2d.to_numpy(),
                  cmap=mesh.cmap, norm=mesh.norm, linewidths=.5, edgecolor="white")

    for t in ax.texts:
        t.set_x(remap(t.get_position()[0]))

    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    ax.set_xticks(centers)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(0, edges[-1])

    for coll in g.ax_col_dendrogram.collections:
        coll.set_segments([np.column_stack([remap(seg[:, 0] / 10.) * 10., seg[:, 1]])
                           for seg in coll.get_segments()])
    g.ax_col_dendrogram.set_xlim(0, edges[-1] * 10.)


file_path = f"{SUPP_ROOT}/{REPLICATES[-1][0]}/all_motifs_auc_barplot.csv"
df = pd.read_csv(file_path)
if df.columns[0].lower().startswith("unnamed"):
    df = df.drop(columns=df.columns[0])
df["Motif"] = df["Motif"].replace({"MYC_rs6983267": "MYC"})
data = df.pivot(index="Motif", columns="Algorithm", values="AUC")
data[DEEPACE_ALGOS] = deepace_mean.loc[DEEPACE_ALGOS].T   # baselines do not depend on the draw, DeepACE becomes the four-draw mean
if data[DEEPACE_ALGOS].isna().any().any():
    raise ValueError(f"DeepACE columns do not align with the motifs in {file_path}")

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

annot = pd.DataFrame(index=data.index, columns=DEEPACE_ALGOS, dtype=object)
for algo in DEEPACE_ALGOS:
    annot[algo] = pd.Series([f"{m:.2f}±{s:.2f}" for m, s in zip(deepace_mean.loc[algo], deepace_sd.loc[algo])],
                            index=row_names).reindex(data.index[:-1])
    annot.loc["Mean", algo] = f"{mean_row_stats.loc[algo, 'Mean']:.2f}±{mean_row_stats.loc[algo, 'SD']:.2f}"

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
    figsize=FIG_SIZE,
    tree_kws={'linewidths': 2, 'colors': '#2d3436'},
    dendrogram_ratio=(0, 0.12),
    cbar_pos=(0.92, 0.92, 0.015, 0.12),
    cbar_kws={"label": "AUC"}
)
relabel_cells(g, annot, DEEPACE_ALGOS)
widen_columns(g, DEEPACE_ALGOS, DEEPACE_COL_WIDTH)
g.ax_cbar.set_ylabel("AUC", fontsize=24, labelpad=15)
g.ax_cbar.tick_params(labelsize=24)
ax = g.ax_heatmap
ax.set_title("AUC performance comparison\n15 saturated mutagenesis datasets", fontsize=24, pad=60)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=24)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=24)
for tick_label in ax.get_yticklabels():
    if tick_label.get_text() == "Mean":
        tick_label.set_fontweight("bold")
        tick_label.set_color("black")
        tick_label.set_fontsize(24)
ax.set_xlabel("")
ax.set_ylabel("")
g.ax_heatmap.invert_xaxis()
g.ax_col_dendrogram.invert_xaxis()
heatmap_path = save_all_formats(g, f"{SUPP_ROOT}/S11_merged_heatmap_quantile", bbox_inches="tight")

print(f"[merged] done -> {heatmap_path}  [{','.join(SAVE_FORMATS)}]", flush=True)


''' ================= AUC Tables Across the 4 Draws ================= '''

per_motif_path = f"{SUPP_ROOT}/S11_deepace_auc_mean_sd_per_motif.csv"
pd.concat([deepace_mean.T.add_suffix("_mean"), deepace_sd.T.add_suffix("_sd")], axis=1).to_csv(per_motif_path)

mean_row_path = f"{SUPP_ROOT}/S11_deepace_mean_row_across_draws.csv"
mean_row_stats.to_csv(mean_row_path)

print("\n" + "=" * 70)
print(f"DeepACE Mean-row AUC across the {len(run_labels)} draws "
      f"({len(REPLICATES)} extra seeds + the original F02e6 result)")
print("=" * 70)
print(mean_row_stats.round(4).to_string())
print(f"\n-> {mean_row_path}")
print(f"-> {per_motif_path}")
print("=" * 70)
