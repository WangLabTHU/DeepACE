'''
    HepG2      DNase 109 / ATAC  56   both ATAC/DNase    -> F9, LDLR, SORT1
    K562       DNase 118 / ATAC  60   both ATAC/DNase   -> PKLR
    HEL 92.1.7 / SK-MEL-28 / HaCaT / SF7996 / MIN6 / Neuro-2a / Hek293T   neither ATAC/DNase
'''


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import seaborn as sns

import os
import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr
from scipy.cluster.hierarchy import linkage as _hier_linkage  
import joblib      


''' ---------------- Configuration ---------------- '''

SUPP_ROOT = "./Supps/S12_afg_variant_heatmap"
AFG_DIR = f"{SUPP_ROOT}/afg"
SAVE_FORMATS = ["pdf", "png", "svg"]
AFG_SCORE_COLUMN = "scores"
AUC_VMIN, AUC_VMAX = 0.4, 0.9
AUC_CBAR_TICKS = [0.4, 0.6, 0.8]
MOTIF_LIST = ["TERT", "HBG1", "LDLR", "F9", "GP1BA", "IRF4", "IRF6", "PKLR", "ZFAND3", "SORT1",
              "HBB", "UC88", "MYC_rs6983267", "RET", "TCF7L2"]

DEEPACE_SPECS = {
    "DeepACE-randaug": ("./Preds/D05_mprabase/analysis_cosine",       "pca50_variant_scores_{motif}.csv"),
    "DeepACE-random":  ("./Preds/D05_mprabase/analysis_mahalanobis", "pca50_variant_scores_{motif}.csv"),
    "DeepACE-cold":    ("./Preds/D05_mprabase/analysis_cold",        "pca50_variant_scores_{motif}.csv"),
}
AFG_SPECS = {
    "AlphaGenome-DNase": "afg_dnase_variant_scores_{motif}.csv",
    "AlphaGenome-ATAC":  "afg_atac_variant_scores_{motif}.csv",
}

ALGORITHMS = list(DEEPACE_SPECS) + list(AFG_SPECS)


''' ---------------- saving figures ---------------- '''

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


''' ---------------- Metrics ---------------- '''

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


def safe_auc(motif, algo, df):
    '''
    AlphaGenome might not have identical cell line for ATAC/DNase, thus we add a function to deal with NaN
    '''
    if df is None:
        return np.nan, 0
    v = np.asarray(df["scores"], dtype=float)
    n = int(np.isfinite(v).sum())
    if n == 0:
        return np.nan, 0
    auc = calculate_auc_per_motif(motif, {algo: {motif: df}})[algo]
    return auc, n


''' ---------------- reading table ---------------- '''

os.makedirs(SUPP_ROOT, exist_ok=True)


def afg_path(algo, motif):
    return os.path.join(AFG_DIR, AFG_SPECS[algo].format(motif=motif))


def load_pair(df, what):
    for col in ("scores", "variant_effects"):
        if col not in df.columns:
            raise ValueError(f"{what}: lack of {col} (current cols: {list(df.columns)})")
    return df[["scores", "variant_effects"]].copy()


motif_list = [m for m in MOTIF_LIST
              if os.path.exists(afg_path("AlphaGenome-DNase", m))
              or os.path.exists(afg_path("AlphaGenome-ATAC", m))]
if not motif_list:
    raise FileNotFoundError(
        f"No afg_*_variant_scores_*.csv found under {AFG_DIR}\n"
        f"  Run revisions/S12_afg_predict.py locally first (env model_afg); it writes the csvs to\n"
        f"  ./Supps/S12_afg_variant_heatmap/afg/. Copy that directory to the same path under this project root.\n"
        f"  (Run with --list-only first to see which datasets have an AlphaGenome channel)"
    )
print(f"[rows] {len(motif_list)} / {len(MOTIF_LIST)} datasets have AlphaGenome predictions -> {motif_list}")
_coverage = os.path.join(AFG_DIR, "S12_afg_coverage.csv")
if os.path.exists(_coverage):
    _cov = pd.read_csv(_coverage)
    _drop = _cov[_cov["status"] == "skip"]
    if len(_drop):
        print(f"[rows] skipped {len(_drop)} datasets (no matching DNase/ATAC cell-line channel in AlphaGenome): "
              + ", ".join(f"{r.dataset}({r.cell_line})" for r in _drop.itertuples()))


df_dict, metrics_rows = {algo: {} for algo in ALGORITHMS}, []
for motif in motif_list:
    for algo in ALGORITHMS:
        if algo in DEEPACE_SPECS:
            d, tmpl = DEEPACE_SPECS[algo]
            path = f"{d}/{tmpl.format(motif=motif)}"
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing DeepACE column table: {path}\n"
                    f"  This column reads the published F02e1 output (background = random_sample_1); "
                    f"this script does not rerun the model.\n"
                    f"  Make sure ./Preds/D05_mprabase/analysis_*/ exists under the project root "
                    f"(same batch as F02e6)."
                )
            sub = load_pair(pd.read_csv(path), path)
            sub["scores"] = -sub["scores"]
        else:
            if not os.path.exists(afg_path(algo, motif)):
                df_dict[algo][motif] = None            # no channel for this dataset -> leave empty
                continue
            path = afg_path(algo, motif)
            raw = pd.read_csv(path)
            if AFG_SCORE_COLUMN not in raw.columns:
                raise ValueError(f"{path}: no '{AFG_SCORE_COLUMN}' column (available: {list(raw.columns)})")
            raw = raw.drop(columns=[c for c in ["scores"]
                                    if c in raw.columns and c != AFG_SCORE_COLUMN])
            raw = raw.rename(columns={AFG_SCORE_COLUMN: "scores"})
            sub = load_pair(raw, path)
        df_dict[algo][motif] = sub
    print(f"[load] {motif} ok", flush=True)



''' ---------------- AUC calculation ---------------- '''

for motif in motif_list:
    for algo in ALGORITHMS:
        df = df_dict[algo][motif]
        auc, n_valid = safe_auc(motif, algo, df)
        if df is None or n_valid < 3:
            pcc = scc = np.nan
        else:
            dv = df.dropna(subset=["scores", "variant_effects"])
            pcc, _ = pearsonr(dv["scores"], dv["variant_effects"])
            scc, _ = spearmanr(dv["scores"], dv["variant_effects"])
        metrics_rows.append(dict(Motif=motif, Algorithm=algo, AUC=auc, PCC=pcc, SCC=scc,
                                 N_valid=n_valid))
plt.close("all")

metrics = pd.DataFrame(metrics_rows)
metrics.to_csv(f"{SUPP_ROOT}/S12_afg_metrics.csv", index=False)
auc_table = metrics[["Motif", "Algorithm", "AUC"]].copy()
auc_table.to_csv(f"{SUPP_ROOT}/all_motifs_auc_barplot.csv")

print("\nPer-cell AUC / PCC (NaN when n_valid < 3 or channel missing):")
print(metrics.pivot(index="Motif", columns="Algorithm", values="AUC").round(4).to_string())
_ag = metrics[metrics["Algorithm"].str.startswith("AlphaGenome")]
_bad = _ag[_ag["PCC"] <= 0]
if len(_bad):
    print(f"\nNote: PCC <= 0 in {len(_bad)} AlphaGenome cells "
          f"({', '.join(f'{r.Motif}/{r.Algorithm}={r.PCC:.3f}' for r in _bad.itertuples())}).\n"
          f"      If this happens broadly, check the sign self-test in the local S12_afg_predict_log.csv ")


''' ---------------- Plotting Figures ---------------- '''

file_path = f"{SUPP_ROOT}/all_motifs_auc_barplot.csv"
df = pd.read_csv(file_path)
if df.columns[0].lower().startswith("unnamed"):
    df = df.drop(columns=df.columns[0])
df["Motif"] = df["Motif"].replace({"MYC_rs6983267": "MYC"})
data = df.pivot(index="Motif", columns="Algorithm", values="AUC")
n_datasets = len(data)
mean_row = data.mean(axis=0)
data = pd.concat([data, mean_row.to_frame().T])
data.index = list(data.index[:-1]) + ["Mean"]
# placeholder
data = data.rename(columns={
    "phyloP100way": "phyloP 100",
    "phyloP470way": "phyloP 470",
    "phastCons100way": "phastCons 100",
    "phastCons470way": "phastCons 470",
    "gpnmsa": "GPN-MSA"
    })
sns.set_theme(style="ticks", context="paper")

_has_nan = bool(np.isnan(np.asarray(data, dtype=float)).any())
_col_linkage = None
if _has_nan:
    _filled = data.fillna(data.mean(axis=0))
    _col_linkage = _hier_linkage(np.asarray(_filled, dtype=float).T,
                                 method="average", metric="euclidean")

g = sns.clustermap(
    data,
    row_cluster=False,
    col_cluster=True,
    col_linkage=_col_linkage,      
    cmap="YlGnBu",
    annot=True,
    fmt=".2f",
    annot_kws={"size": 22},
    linewidths=.5,
    linecolor="white",
    figsize=(9, 6),
    tree_kws={'linewidths': 2, 'colors': '#2d3436'},
    dendrogram_ratio=(0, 0.12),
    cbar_pos=(0.92, 0.92, 0.015, 0.12),
    cbar_kws={"label": "AUC", "ticks": AUC_CBAR_TICKS},
    vmin=AUC_VMIN,
    vmax=AUC_VMAX,
)
g.ax_cbar.set_ylabel("AUC", fontsize=24, labelpad=15)
g.ax_cbar.tick_params(labelsize=22)
ax = g.ax_heatmap
ax.set_title(f"AUC performance comparison\n{n_datasets} saturated mutagenesis datasets",
             fontsize=24, pad=60)
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

_vals = np.asarray(data, dtype=float)
_lo = int(np.sum(_vals < AUC_VMIN))
_hi = int(np.sum(_vals > AUC_VMAX))
print(f"[cbar] data {np.nanmin(_vals):.3f} ~ {np.nanmax(_vals):.3f}, fixed colorbar limits "
      f"{AUC_VMIN} ~ {AUC_VMAX} "
      + (f" -- note: {_lo} cells below vmin, {_hi} above vmax; these will be clipped to the endpoint colors"
         if (_lo or _hi) else " -- all cells within range"))
heatmap_path = save_all_formats(g, f"{SUPP_ROOT}/S12_afg_variant_heatmap", bbox_inches="tight")
print(f"\n-> {heatmap_path}  [{','.join(SAVE_FORMATS)}]")
print(f"-> {SUPP_ROOT}/S12_afg_metrics.csv")
print(f"-> {SUPP_ROOT}/all_motifs_auc_barplot.csv")
