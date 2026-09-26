
import random
import os, sys
import time
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
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import joblib

random.seed(42)
np.random.seed(42)
from math import pi
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'


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
    pseudo_data = np.load("./Preds/D10_random/random_sample_1/uni_pred.npy")
    return primary_data, pseudo_data, labels

def filter_data_by_models(data, selected_models, anno_df):
    model_indices = {}
    for model in model_list:
        model_indices[model] = anno_df[anno_df['model'] == model].index.tolist()
    keep_indices = []
    for model in selected_models:
        keep_indices.extend(model_indices[model])
    keep_indices = sorted(keep_indices)
    return data[:, keep_indices] if keep_indices else data

def preprocess_data(primary_data, pseudo_data, labels):
    """Scale data and categorize into positive, negative, and mid groups."""
    combined_data = np.vstack((primary_data, pseudo_data)) if len(pseudo_data) > 0 else primary_data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)
    # Separate scaled data
    scaled_primary = scaled_data[:-len(pseudo_data)] if len(pseudo_data) > 0 else scaled_data
    scaled_pseudo = scaled_data[-len(pseudo_data):] if len(pseudo_data) > 0 else np.array([])
    # Categorize labels
    n_total = len(labels)
    n_top = int(n_total * 0.2)
    indices = np.argsort(labels)
    neg_data = scaled_primary[indices[:n_top]]
    pos_data = scaled_primary[indices[-n_top:]]
    mid_data = scaled_primary[indices[n_top:-n_top]]
    sorted_labels = np.concatenate([labels[indices[:n_top]], labels[indices[-n_top:]], labels[indices[n_top:-n_top]]])
    # Combine samples and create labels
    sample_data = np.vstack((neg_data, pos_data, mid_data, scaled_pseudo)) if len(pseudo_data) > 0 else np.vstack((neg_data, pos_data, mid_data))
    sample_labels = (['Negative'] * len(neg_data) + ['Positive'] * len(pos_data) + ['Mid'] * len(mid_data) + ['Pseudo'] * len(scaled_pseudo))
    return sample_data, sample_labels, sorted_labels


def compute_fold_change(sample_data, sample_labels, sorted_labels, n_neighbors=500):

    pseudo_mask = np.array([g == 'Pseudo' for g in sample_labels])
    real_idx = np.where(~pseudo_mask)[0]
    pseudo_idx = np.where(pseudo_mask)[0]
    real_vectors = sample_data[real_idx]
    pseudo_vectors = sample_data[pseudo_idx]
    n_neighbors = min(n_neighbors, len(pseudo_idx))

    var = np.var(pseudo_vectors, axis=0)
    inv_std = 1.0 / np.sqrt(var + 1e-8)
    def diag_mahalanobis(x, y, inv_std=inv_std):
        diff = (x - y) * inv_std
        return np.sqrt(np.dot(diff, diff))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=diag_mahalanobis).fit(pseudo_vectors)
    distances, _ = nbrs.kneighbors(real_vectors)
    pseudo_similarity = 1 - np.mean(distances, axis=1)

    order = np.argsort(-pseudo_similarity)
    expr_sorted = sorted_labels[order]
    n = len(expr_sorted)

    group_size = 100
    groups = [expr_sorted[i:i + group_size] for i in range(0, n, group_size)]
    means   = [np.mean(g)   for g in groups]
    medians = [np.median(g) for g in groups]
    fc = 2 ** (means[-1] - np.mean(means))
    return fc if len(medians) >= 2 else np.nan


# ============================================================================================
# Configuration
# ============================================================================================
N_RANDOM_COMBOS = 10        
RANDOM_SEED = 42            
COMBO_SIZES = None          
RANDOM_COMBOS = None        
GREEDY_CSV = "./Figs/F04_interpret_robust/F04d_robust_heatmap/greedy_search.csv"
PUBLISHED_COMB_CSV = "./Figs/F04_interpret_robust/F04d_robust_heatmap/comb_model_validation.csv"
MAX_DRAW_ATTEMPTS = 5000    

# ---- DeepACE reference col ----
DEEPACE_FC_SOURCE = "auto"  # "auto" | "compute" | "csv" 
REFERENCE_CSV = "./Figs/F04_interpret_robust/F04d_robust_heatmap/single_model_validation.csv"
REFERENCE_COL = "DeepACE"
REFERENCE_TOL = 1e-6        

# ---- Existing Foldchange table ----
REUSE_EXISTING_FC = False  


'''
basic settings 
'''

print("\n" + "="*60)
print("S24: RANDOM MODEL-COMBINATION ROBUSTNESS HEATMAP")
print("="*60)
model_list = [
    "Malinois", "Basset", "DanQ", "MPRALegNet", "SahuCNN", "APARENT2",
    "DeepDNAshape", "CLIPNET", "Puffin", "Enformer", "Basenji2",
    "Expecto", "Sei", "SpliceAI", "Borzoi", "SegmentNT"
]
scenarios_list = [
    ("MPRA", "HepG2", None, "HepG2"),
    ("MPRA", "K562", None, "K562"),
    ("MPRA", "SKNSH", None, "SKNSH"),
    ("epigenetics", "HepG2", "ELF1_1_aim", "ELF1"),
    ("epigenetics", "HepG2", "HNF1A_1_aim", "HNF1A"),
    ("epigenetics", "HepG2", "HNF4A_1_aim", "HNF4A")
]
output_dir = "./Supps/S24_random_combo_heatmap"
os.makedirs(output_dir, exist_ok=True)
anno_df = pd.read_csv(f"./total_features.csv")
scenario_names = [f"{dataset}_{plot_tag}" for dataset, _, _, plot_tag in scenarios_list]
n_scen = len(scenario_names)


'''
random combination
'''
print("\n" + "-"*60)
print("STEP 1: DRAW RANDOM MODEL COMBINATIONS")
print("-"*60)


def greedy_combo_sizes(path):
    """ setting the number of models per combination as the same as F04d -- greedy_search.csv """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    cols = [c for c in df.columns if str(c).strip().lower() == "selected_models"]
    if not cols:
        raise KeyError(f"Column 'selected_models' not found in {path} (existing columns: {list(df.columns)})")
    col = cols[0]
    sizes, detail = [], []
    for i, raw in enumerate(df[col].tolist()):
        models = [m.strip() for m in str(raw).split("→") if m.strip()]
        bad = [m for m in models if m not in model_list]
        if bad:
            raise KeyError(f"Row {i} in {path} contains models outside model_list: {bad}; "
                           f"the model tables on the two sides do not match. Please reconcile them before running.")
        sizes.append(len(models))
        detail.append(models)
    empty = [j for j, k in enumerate(sizes) if k == 0]
    if empty:
        raise ValueError(
            f"{len(empty)} rows in {path} have empty greedy combinations (rows {empty}) -- "
            f"cannot determine the sizes of random combinations from this file.\n"
            f"  Please hard-code COMBO_SIZES at the top of this file (e.g. [2, 3, 3, 4])")
    print(f"  [greedy] {path}")
    for name, models in zip(df.iloc[:, 0].tolist(), detail):
        print(f"           {name}: {len(models)} models = {' + '.join(models)}")
    return sizes


def published_combo_sizes(path, announce=False):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0, index_col=0)
    cols = [c for c in header.columns
            if str(c).strip() != REFERENCE_COL and not str(c).startswith("Unnamed")]
    if not cols:
        raise KeyError(f"No combination columns found in {path} (existing columns: {list(header.columns)})")
    sizes, detail = [], []
    for c in cols:
        models = [m for m in str(c).split("_") if m]
        bad = [m for m in models if m not in model_list]
        if not models or bad:
            raise KeyError(f"Column name '{c}' in {path} cannot be parsed into a model combination "
                           f"(parsed {models}, of which {bad} are not in model_list); "
                           f"please hard-code COMBO_SIZES at the top of this file (e.g. [2, 3, 3, 4]).")
        sizes.append(len(models))
        detail.append(models)
    if announce:
        print(f"  [sizes] greedy_search.csv not found; falling back to the published F04d heatmap table: {path}")
        print(f"          (its column names are the deduplicated greedy combinations; if the two datasets "
              f"converged on the same combination, its size will be undercounted once)")
        for models in detail:
            print(f"          {len(models)} models = {' + '.join(models)}")
    return sizes


def combo_size_pool():
    if COMBO_SIZES is not None:
        print(f"  [combos] Using the hard-coded COMBO_SIZES from the top of this file: {list(COMBO_SIZES)}")
        return list(COMBO_SIZES), "COMBO_SIZES"
    try:
        pool = greedy_combo_sizes(GREEDY_CSV)
        source = f"greedy_csv: {GREEDY_CSV}"
        try:
            other = published_combo_sizes(PUBLISHED_COMB_CSV)
        except Exception:
            other = None
        if other is not None and sorted(other) != sorted(pool):
            print(f"  [sizes][note] The size multisets from the two sources disagree: "
                  f"greedy_search.csv {sorted(pool)} vs published table (deduplicated) {sorted(other)} "
                  f"-- this script uses the former.")
        return pool, source
    except FileNotFoundError:
        pass
    try:
        pool = published_combo_sizes(PUBLISHED_COMB_CSV, announce=True)
        return pool, f"published_comb_csv: {PUBLISHED_COMB_CSV}"
    except FileNotFoundError:
        pass
    raise FileNotFoundError(
        f"Cannot obtain greedy combination sizes: neither {GREEDY_CSV} nor {PUBLISHED_COMB_CSV} is available.\n"
        f"  Three options, pick one:\n"
        f"  (1) Hard-code COMBO_SIZES as a list at the top of this file, e.g. COMBO_SIZES = [2, 3, 3, 4]; "
        f"then neither csv is needed;\n"
        f"  (2) Or run the F04d greedy search step under {os.path.dirname(GREEDY_CSV)} "
        f"(it will write greedy_search.csv);\n"
        f"  (3) Or place the published comb_model_validation.csv used for the heatmap in the same directory.")


def sorted_models(models):
    picked = tuple(sorted(models, key=model_list.index))
    if len(set(picked)) != len(picked):
        raise ValueError(f"Duplicate models in combination: {models}")
    bad = [m for m in picked if m not in model_list]
    if bad:
        raise KeyError(f"Models outside model_list in combination: {bad} (available: {model_list})")
    return picked


def draw_random_combos():
    """ 
    Return (combos, sizes, source).
    combos = [(model, ...), ...], sizes = number of models in each combination,
    source = "RANDOM_COMBOS" / "COMBO_SIZES" / "greedy_csv: <path>" / "published_comb_csv: <path>".
    """
    if RANDOM_COMBOS is not None:
        combos = [sorted_models(m) for m in RANDOM_COMBOS]
        print(f"  [combos] Using the hard-coded RANDOM_COMBOS from the top of this file directly ({len(combos)} combos); "
              f"COMBO_SIZES / GREEDY_CSV / RANDOM_SEED are all ignored.")
        return combos, [len(m) for m in combos], "RANDOM_COMBOS"

    sizes_pool, source = combo_size_pool()
    if not all(1 <= k <= len(model_list) for k in sizes_pool):
        raise ValueError(f"Sizes must be integers between 1 and {len(model_list)}, got {sizes_pool}")

    rng = random.Random(RANDOM_SEED)
    order = list(range(len(sizes_pool)))
    rng.shuffle(order)
    sizes = [sizes_pool[order[i % len(sizes_pool)]] for i in range(N_RANDOM_COMBOS)]

    combos, seen = [], set()
    for i, k in enumerate(sizes):
        for _ in range(MAX_DRAW_ATTEMPTS):
            picked = sorted_models(rng.sample(model_list, k))
            if picked not in seen:
                break
        else:
            raise RuntimeError(f"While drawing combination {i+1}, hit an existing combination "
                               f"{MAX_DRAW_ATTEMPTS} times in a row; "
                               f"reduce N_RANDOM_COMBOS or use a different RANDOM_SEED.")
        seen.add(picked)
        combos.append(picked)
    print(f"  [combos] seed = {RANDOM_SEED}; sizes = {sizes} "
          f"(i.e. how many times each greedy size is reused; see S24_random_combos.csv for the one-to-one mapping)")
    return combos, sizes, source


'''
generation of model combination, following the settings (model number per comb) in Fig. 4
'''

random_combos, combo_sizes, size_source = draw_random_combos()
combo_names = ["_".join(models) for models in random_combos]
if len(set(combo_names)) != len(combo_names):
    raise RuntimeError("Duplicate combo names after drawing -- this should be impossible; "
                       "check the dedup logic in draw_random_combos.")
print(f"  [combos] {len(combo_names)} random combinations:")
for idx, (name, models) in enumerate(zip(combo_names, random_combos), 1):
    print(f"           {idx:>2}. {name}   ({len(models)} models)")

combos_df = pd.DataFrame({
    "column_index": np.arange(1, len(combo_names) + 1),      
    "combo_name": combo_names,
    "n_models": combo_sizes,
    "models": [" + ".join(m) for m in random_combos],
    "size_source": size_source,                             
    "random_seed": RANDOM_SEED,
})
combos_csv = os.path.join(output_dir, "S24_random_combos.csv")
combos_df.to_csv(combos_csv, index=False)
print(f"  [combos] Manifest saved: {combos_csv}")


'''
DeepACE reference col, already verified in Fig. 4
'''

def reference_from_csv(path, col, names):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:                                     
        print(f"  [ref] cannot read {path} ({exc}) -> computing the DeepACE column on the fly")
        return None
    if col not in df.columns:
        print(f"  [ref] no '{col}' column in {path} -> computing the DeepACE column on the fly")
        return None
    key_col = df.columns[0]
    got = {}
    for name in names:
        hit = df[df[key_col].astype(str) == name]
        if len(hit) != 1:
            print(f"  [ref] '{name}' matched {len(hit)} rows in {path} (exactly 1 required) "
                  f"-> computing the DeepACE column on the fly")
            return None
        got[name] = float(hit[col].iloc[0])
    return pd.Series([got[n] for n in names], index=names)


fc_csv = os.path.join(output_dir, "S24_random_comb_model_validation.csv")
fc_df = None
reused = False

if REUSE_EXISTING_FC and os.path.exists(fc_csv):
    cached = pd.read_csv(fc_csv, index_col=0)
    why = []
    if list(cached.columns) != combo_names + [REFERENCE_COL]:
        why.append(f"column names/order mismatch (previously saved: {list(cached.columns)[:3]}..., "
                   f"now {combo_names[:3]}...)")
    if list(cached.index) != scenario_names:
        why.append(f"row names/order mismatch (previously saved: {list(cached.index)}, now {scenario_names})")
    if not why:
        fc_df = cached.astype(float)
        reused = True
        print("\n" + "="*60)
        print(f"REUSE_EXISTING_FC = True: directly reusing {fc_csv}")
        print("="*60)
    else:
        print(f"\n  [reuse] cannot reuse the previously saved table -> recomputing as usual: {'; '.join(why)}")



'''
Calculation of fold-change
'''

if not reused:
    print("\n" + "="*60)
    print("STEP 2: FOLD-CHANGE FOR EVERY (COMBINATION x DATASET) CELL")
    print("="*60)
    print(f"  {n_scen} datasets x ({len(combo_names)} random combos + 1 DeepACE) = "
          f"{n_scen * (len(combo_names) + 1)} cells")

    fc_matrix = np.full((n_scen, len(combo_names)), np.nan)
    deepace_fc = np.full(n_scen, np.nan)
    t_all = time.time()

    for scen_idx, (dataset, cell, motif, plot_tag) in enumerate(scenarios_list):
        scenario_name = f"{dataset}_{plot_tag}"
        primary_data, pseudo_data, labels = load_data(cell, motif)
        print(f"\n  [{scen_idx + 1}/{n_scen}] {scenario_name}: primary={primary_data.shape}, "
              f"pseudo={pseudo_data.shape}, labels={np.shape(labels)}")

        # ---- random combination ----
        for comb_idx, comb_models in enumerate(random_combos):
            comb_name = combo_names[comb_idx]
            t0 = time.time()
            primary_data_temp = filter_data_by_models(primary_data, list(comb_models), anno_df)
            pseudo_data_temp = filter_data_by_models(pseudo_data, list(comb_models), anno_df)
            combined_data = np.vstack((primary_data_temp, pseudo_data_temp)) if len(pseudo_data_temp) > 0 else primary_data_temp
            n_features = combined_data.shape[1]
            if n_features <= 1:
                print(f"      [skip: <2 feats] {comb_name} has only {n_features} columns -> cell set to NaN")
                continue
            elif n_features < 50:
                uni_selected = combined_data
            else:
                uni_selected = PCA(n_components=50, random_state=42).fit_transform(combined_data)
            primary_data_temp = uni_selected[:-len(pseudo_data_temp)] if len(pseudo_data_temp) > 0 else uni_selected
            pseudo_data_temp = uni_selected[-len(pseudo_data_temp):] if len(pseudo_data_temp) > 0 else np.array([])
            sample_data, sample_labels, sorted_labels = preprocess_data(primary_data_temp, pseudo_data_temp, labels)
            fc = compute_fold_change(sample_data, sample_labels, sorted_labels)
            fc_matrix[scen_idx, comb_idx] = fc
            print(f"      {comb_idx + 1:>2}/{len(combo_names)} {comb_name}: "
                  f"{n_features} feats, FC={fc:.4f}   ({time.time() - t0:.1f}s)")

        # ---- DeepACE reference ----
        if DEEPACE_FC_SOURCE == "csv":
            continue
        t0 = time.time()
        filt_primary = filter_data_by_models(primary_data, model_list, anno_df)
        filt_pseudo = filter_data_by_models(pseudo_data, model_list, anno_df)
        combined = np.vstack((filt_primary, filt_pseudo)) if len(filt_pseudo) > 0 else filt_primary
        uni_selected = PCA(n_components=50, random_state=42).fit_transform(combined)
        primary_data = uni_selected[:-len(filt_pseudo)] if len(filt_pseudo) > 0 else uni_selected
        pseudo_data = uni_selected[-len(filt_pseudo):] if len(filt_pseudo) > 0 else np.array([])
        sample_data, sample_labels, sorted_labels = preprocess_data(primary_data, pseudo_data, labels)
        deepace_fc[scen_idx] = compute_fold_change(sample_data, sample_labels, sorted_labels)
        print(f"      -> DeepACE (all {len(model_list)} models): FC={deepace_fc[scen_idx]:.4f} "
              f"({combined.shape[1]} feats, {time.time() - t0:.1f}s)")

    print(f"\n  Per-cell computation done in {time.time() - t_all:.1f}s")

    # ---- checking to ensure whether deepace calculated the same as before (Fig. 4) ----
    ref_from_csv = reference_from_csv(REFERENCE_CSV, REFERENCE_COL, scenario_names) \
        if DEEPACE_FC_SOURCE in ("auto", "csv") else None
    if ref_from_csv is not None:
        ref_values = ref_from_csv.to_numpy(dtype=float)
        if DEEPACE_FC_SOURCE == "auto" and np.isfinite(deepace_fc).all():
            delta = np.abs(ref_values - deepace_fc)
            worst = float(np.nanmax(delta))
            check_df = pd.DataFrame({
                "scenario": scenario_names,
                "deepace_fc_from_csv": ref_values,
                "deepace_fc_recomputed": deepace_fc,
                "abs_delta": delta,
            })
            check_csv = os.path.join(output_dir, "S24_deepace_reference_check.csv")
            check_df.to_csv(check_csv, index=False, float_format="%.10f")
            print(f"  [ref] Reference column taken from '{REFERENCE_COL}' in {REFERENCE_CSV} "
                  f"(same source as the published figure);")
            print(f"        recomputed DeepACE vs csv: max|delta| = {worst:.3e} -> {check_csv}")
            if worst > REFERENCE_TOL:
                print("  [ref][warn] Reference column mismatch: the figure still uses the csv values "
                      "(consistent with the published figure). Check code/data/environment differences "
                      "before deciding which column to use.")
        else:
            print(f"  [ref] Reference column taken from '{REFERENCE_COL}' in {REFERENCE_CSV} "
                  f"(DEEPACE_FC_SOURCE='{DEEPACE_FC_SOURCE}': no cross-check)")
    else:
        print("  [ref] Reference column not available from csv -> using recomputed DeepACE values")
        ref_values = deepace_fc
    if not np.isfinite(ref_values).all():
        raise RuntimeError(f"NaN/Inf in the DeepACE reference column: {ref_values}; "
                           f"ratios cannot be computed without it.")


if reused:
    ref_values = fc_df[REFERENCE_COL].to_numpy(dtype=float)


'''
saving figures 
'''
comb_csv = fc_csv
fc_df_comb = pd.read_csv(comb_csv, index_col=0)
fc_df_comb = fc_df_comb.astype(float)
deepace_fc_comb = fc_df_comb["DeepACE"].values
other_cols_comb = [c for c in fc_df_comb.columns if c != "DeepACE"]
fc_values_comb = fc_df_comb[other_cols_comb].values

# normalized color strength: |fc| / |All| × sign(fc)
abs_ratio = np.abs(fc_values_comb) / (np.abs(deepace_fc_comb)[:, np.newaxis] + 1e-12)
signs = np.sign(fc_values_comb)
color_intensity = abs_ratio * signs
deepace_intensity = np.ones((len(fc_df_comb), 1))
intensity_matrix = np.hstack([color_intensity, deepace_intensity])

average_row = fc_df_comb.mean()
average_row.name = "Average"
fc_df_comb = pd.concat([fc_df_comb, average_row.to_frame().T], axis=0)
mean_row = intensity_matrix.mean(axis=0)
intensity_matrix = np.vstack([intensity_matrix, mean_row])

plot_df = fc_df_comb.copy()
fig, ax = plt.subplots(figsize=(len(plot_df.columns) * 1.5 + 4, 5))
custom_cmap = LinearSegmentedColormap.from_list(
    "GreenWhiteYellow",
    ["#74a892", "#ffffff", "#e5c185"]
)
cmap = custom_cmap
norm = TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=1.5)

custom_cmap = LinearSegmentedColormap.from_list(
    "GreenWhiteYellow",
    ["#74a892", "#ffffff", "#e5c185"])
cmap = custom_cmap

for i in range(len(plot_df)):
    for j in range(len(plot_df.columns)):
        fc_val = plot_df.iloc[i, j]
        intensity = intensity_matrix[i, j]

        if np.isnan(fc_val):
            color = 'lightgray'
            text = 'NaN'
        else:
            color = cmap(norm(intensity))
            text = f"{intensity:.3f}"
        ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                   facecolor=color, edgecolor='gray', linewidth=0.5))
        fontweight = 'normal'
        color_text = 'white' if abs(intensity) > 1.2 or abs(intensity) < 0.8 else 'black'
        ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                fontsize=16, fontweight=fontweight, color=color_text)

ax.set_xlim(0, len(plot_df.columns))
ax.set_ylim(0, len(plot_df))
ax.set_xticks(np.arange(len(plot_df.columns)) + 0.5)
ax.set_yticks(np.arange(len(plot_df)) + 0.5)
tmp_xlabels = plot_df.columns.str.replace('_', '\n')
ax.set_xticklabels([])
ax.set_yticklabels(plot_df.index, fontsize=18)
ax.invert_yaxis()

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cax = fig.add_axes([1.01, 0.72, 0.015, 0.15])
cbar = fig.colorbar(sm, cax=cax)
cbar.ax.text(
    3, -0.6,
    'Relative\nFC',
    transform=cbar.ax.transAxes,
    ha='center',
    va='center',
    fontsize=18
)
cbar.ax.tick_params(labelsize=16)
cbar.set_ticks([0.5, 1.0, 1.5])
cbar.set_ticklabels(['0.5×', '1.0×', '1.5×'])


plt.tight_layout()
fig_stem = os.path.join(output_dir, "S24_random_comb_heatmap_normalized")
for ext in ["png", "pdf", "svg"]:
    plt.savefig(f"{fig_stem}.{ext}", dpi=400, bbox_inches='tight')
    print(f"Random-combination heatmap saved: {fig_stem}.{ext}")
plt.close()

# saving ratio csv
ratio_df_comb = pd.DataFrame(intensity_matrix, index=fc_df_comb.index, columns=fc_df_comb.columns)
ratio_csv_comb = os.path.join(output_dir, "S24_random_comb_heatmap_normalized.csv")
ratio_df_comb.to_csv(ratio_csv_comb, float_format='%.6f')
print(f"Random-combination ratio matrix saved: {ratio_csv_comb}")


'''
Evaluation
'''
# Top 3 models
mean_ratio_comb = ratio_df_comb.iloc[:, :-1].mean(axis=0).sort_values(ascending=False)
print("\nTop 3 Best Combination Models (Avg FC / All_Models):")
for i, (model, ratio) in enumerate(mean_ratio_comb.head(3).items(), 1):
    print(f"  {i}. {model}: ×{ratio:.3f}")

combos_ratio = ratio_df_comb.iloc[:-1, :-1]                      
summary = pd.DataFrame({
    "combo_name": combo_names,
    "n_models": combo_sizes,
    "models": [" + ".join(m) for m in random_combos],
    "mean_fc": [float(np.nanmean(fc_df[c].values)) for c in combo_names],
    "mean_ratio_vs_deepace": combos_ratio.mean(axis=0).values,
    "min_ratio_vs_deepace": combos_ratio.min(axis=0).values,
    "max_ratio_vs_deepace": combos_ratio.max(axis=0).values,
    "n_datasets_better_than_deepace": (combos_ratio > 1.0).sum(axis=0).values,
})
summary_csv = os.path.join(output_dir, "S24_random_combo_summary.csv")
summary.to_csv(summary_csv, index=False, float_format="%.6f")
best = summary.loc[summary['mean_ratio_vs_deepace'].idxmax(), 'combo_name']
print(f"\nBest mean ratio among the {len(combo_names)} random combos: {best} "
      f"(x{summary['mean_ratio_vs_deepace'].max():.3f}); "
      f"max datasets beating DeepACE: {int(summary['n_datasets_better_than_deepace'].max())}/{n_scen}")
print(f"Random-combination summary saved: {summary_csv}")
print(f"\nDeepACE mean FC = {np.nanmean(ref_values):.4f} "
      f"(per dataset: {', '.join(f'{v:.4f}' for v in ref_values)})")
