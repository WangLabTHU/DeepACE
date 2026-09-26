
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns  
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import os, sys
import json      
import hashlib
import random
import time
import numpy as np
import pandas as pd

from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity 
import joblib 

random.seed(42) 


''' ---------------- Configuration ---------------- '''

SUPP_ROOT = "./Supps/S26_loo_model_ablation_auc"
SAVE_FORMATS = ["pdf", "png", "svg"]
RESUME = True


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


model_list = [
    "Malinois", "Basset", "DanQ", "MPRALegNet", "SahuCNN", "APARENT2",
    "DeepDNAshape", "CLIPNET", "Puffin", "Enformer", "Basenji2",
    "Expecto", "Sei", "SpliceAI", "Borzoi", "SegmentNT"
]


MOTIF_LIST = ["TERT", "HBG1", "LDLR", "F9", "GP1BA", "IRF4", "IRF6", "PKLR", "ZFAND3", "SORT1",
              "HBB", "UC88", "MYC_rs6983267", "RET", "TCF7L2"]
ANNO_CSV = "./total_features.csv"
SAT_NPY_TMPL = "./Preds/D05_mprabase/point_MPRABase_{motif}_saturation/uni_pred.npy"
SAT_TSV_TMPL = "./Preds/D05_mprabase/point_MPRABase_{motif}_saturation.tsv"
RAND_BG_PATH = "./Preds/D10_random/random_sample_1/uni_pred.npy"
RANDAUG_BG_TMPL = "./Preds/D05_mprabase/point_MPRABase_{motif}_randaug/uni_pred.npy"
ARMS_TO_RUN = ["randaug"]
ARMS = {
    "randaug": {
        "metric": "cosine",
        "bg_kind": "per_motif",
        "bg_tmpl": RANDAUG_BG_TMPL,
        "bg_desc": "each motif's own augmented set",
    },
}

RAND_BG_CACHE = True
N_COMPONENTS = 50
PCA_RANDOM_STATE = 42
USE_FAST_SIMILARITY = True
FAST_CHUNK = 2048          
VERIFY_FIRST_MOTIF = True  
VERIFY_N_ALT = 50          
AUC_CHANCE_LEVEL = 0.49
CHANCE_CORRECTED = True

CBAR_LABEL = 'AUC / DeepACE'
CBAR_TICKS = [0.9, 0.95, 1.0, 1.05, 1.1]
CBAR_TICKLABELS = ['0.9×', '0.95×', '1.0×', '1.05×', '1.1×']
HEATMAP_VMIN, HEATMAP_VMAX = 0.9, 1.1

HEATMAP_MODELS = ["Borzoi", "Enformer", "Sei", "Basenji2", "CLIPNET"]
MEAN_DELTA_NEG_COLOR = '#C44E52'
MEAN_DELTA_POS_COLOR = '#4C72B0'
COL_LABEL_MAP = {}

SINGLE_MODEL_COLS = ["Borzoi", "Enformer"]
SINGLE_BARPLOT = True          
SINGLE_BAR_COLORS = ['#74a892', '#d4a558']
SINGLE_YLABEL = 'Δ Normalized AUC\n(relative to DeepACE)'
SINGLE_XLABEL = 'Variant'      
SINGLE_LEGEND_TITLE = 'Model'
SINGLE_LABEL_TMPL = '{model} (only)'


''' ---------------- Loading Data ---------------- '''

def load_saturation_raw(motif):
    uni_pred = np.load(SAT_NPY_TMPL.format(motif=motif))
    df = pd.read_csv(SAT_TSV_TMPL.format(motif=motif), sep="\t")
    labels = df['VariantExpressionEffect (log2)'].to_numpy()

    alt_data = uni_pred[:-1]
    ref_data = uni_pred[-1:]
    valid_cols = np.isfinite(alt_data).any(axis=0) & np.isfinite(ref_data).any(axis=0)
    return alt_data, ref_data, labels, valid_cols


def load_arm_background(arm_name, motif, valid_cols, shared_bg=None):
    cfg = ARMS[arm_name]
    if cfg["bg_kind"] == "shared":
        raw = shared_bg if shared_bg is not None else np.load(cfg["bg_path"])
    else:
        path = cfg["bg_tmpl"].format(motif=motif)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[{arm_name}] missing augmented-set background for {motif}: {path}\n")
        raw = np.load(path)[:-1]
    return raw[:, valid_cols]


def filter_data_by_models(data, selected_models, anno_df):
    model_indices = {m: anno_df[anno_df['model'] == m].index.tolist() for m in model_list}
    keep_indices = []
    for m in selected_models:
        keep_indices.extend(model_indices[m])
    keep_indices = sorted(keep_indices)
    return data[:, keep_indices] if keep_indices else data


''' ---------------- Scoring ---------------- '''

def _cov_inv_safe(pred_ref):
    return np.linalg.pinv(np.atleast_2d(np.cov(pred_ref, rowvar=False)))


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
            cov_inv = _cov_inv_safe(pred_ref)
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
            cov_inv = _cov_inv_safe(pred_ref)
            sims = []
            for x in pred_alt:
                dists = [-mahalanobis(x, y, cov_inv) for y in pred_ref]
                dists = np.array(dists)
                sims.append(dists.mean())
            sims = np.array(sims)
            sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-12)
            return sims


def compute_sample_similarity_fast(pred_alt, pred_ref, metric="mahalanobis", mode="batch",
                                   chunk=None):
    if metric != "mahalanobis" or mode != "batch":
        raise ValueError("compute_sample_similarity_fast only implements metric='mahalanobis', mode='batch'")
    if chunk is None:
        chunk = FAST_CHUNK
    pred_alt = np.asarray(pred_alt)
    pred_ref = np.asarray(pred_ref)
    cov_inv = _cov_inv_safe(pred_ref)      

    A = pred_alt @ cov_inv
    x2 = np.einsum('ij,ij->i', pred_alt, A)
    B = pred_ref @ cov_inv
    y2 = np.einsum('ij,ij->i', pred_ref, B)

    sims = np.empty(len(pred_alt), dtype=float)
    for s in range(0, len(pred_alt), chunk):
        e = min(s + chunk, len(pred_alt))
        xy = A[s:e] @ pred_ref.T                       # (e-s, n_rand)
        d2 = x2[s:e, None] + y2[None, :] - 2.0 * xy
        np.clip(d2, 0.0, None, out=d2)                 
        sims[s:e] = -np.sqrt(d2).mean(axis=1)
    sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-12)
    return sims


def similarity_scores(pred_alt_pca, pred_rand_pca, metric="mahalanobis"):
    if USE_FAST_SIMILARITY and metric == "mahalanobis":
        return compute_sample_similarity_fast(pred_alt_pca, pred_rand_pca, metric=metric, mode="batch")
    return compute_sample_similarity(pred_alt_pca, pred_rand_pca, metric=metric, mode="batch")


''' ---------------- Metric quantile-based AUC ---------------- '''

def quantile_auc(scores, effects, quantile_grid=None):
    if quantile_grid is None:
        quantile_grid = np.linspace(0.01, 0.99, 99)
    scores = np.asarray(scores, dtype=float)
    effects = np.asarray(effects, dtype=float)
    nan_mask = ~np.isnan(scores)
    scores = scores[nan_mask]
    effects = effects[nan_mask]
    discover_rates = []
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
    return float(np.trapz(discover_rates, quantile_grid))


def check_fast_similarity(alt_pca, rand_pca, n_alt):
    n = min(n_alt, len(alt_pca))
    if n == 0:
        return
    t0 = time.time()
    slow = compute_sample_similarity(alt_pca[:n], rand_pca, metric="mahalanobis", mode="batch")
    t_slow = time.time() - t0
    t0 = time.time()
    fast = compute_sample_similarity_fast(alt_pca[:n], rand_pca, metric="mahalanobis", mode="batch")
    t_fast = time.time() - t0
    d_raw = np.max(np.abs(slow - fast))
    d_score = np.max(np.abs(-slow - (-fast)))
    print(f"  [self-check] verbatim vs fast version (first {n} alt x {len(rand_pca)} background): "
          f"max|Δ normalized score| = {d_raw:.3e}, max|Δ activity score| = {d_score:.3e}, "
          f"time {t_slow:.2f}s -> {t_fast:.3f}s (x{t_slow / max(t_fast, 1e-9):.0f})", flush=True)
    if d_raw > 1e-8:
        print("  [self-check][warn] deviation between the two paths exceeds 1e-8 - please set USE_FAST_SIMILARITY to False, "
              "and send me this log line.", flush=True)
    else:
        print("  [self-check] consistent (deviation at floating-point noise level).", flush=True)


def preflight_arms(arms_run, motifs=None):
    motifs = list(MOTIF_LIST if motifs is None else motifs)
    for arm_name in arms_run:
        cfg = ARMS[arm_name]
        if cfg["bg_kind"] == "shared":
            if not os.path.exists(cfg["bg_path"]):
                raise FileNotFoundError(f"[{arm_name}] background file does not exist: {cfg['bg_path']}")
            print(f"[preflight] {arm_name:8s} metric={cfg['metric']:12s} background = {cfg['bg_path']}  "
                  f"({cfg['bg_desc']})")
        else:
            missing = [m for m in motifs if not os.path.exists(cfg["bg_tmpl"].format(motif=m))]
            if missing:
                raise FileNotFoundError(
                    f"[{arm_name}] {len(missing)}/{len(motifs)} motifs lack augmented-set background, "
                    f"e.g. {cfg['bg_tmpl'].format(motif=missing[0])}\n"
                    f"  missing motifs: {missing}")
            print(f"[preflight] {arm_name:8s} metric={cfg['metric']:12s} background = "
                  f"{cfg['bg_tmpl'].format(motif='{motif}')}  ({cfg['bg_desc']}, "
                  f"{len(motifs)}/{len(MOTIF_LIST)} motifs ready)")


''' ================= Part 1: AUC matrix (15 motif × 17 cols) ================='''

os.makedirs(SUPP_ROOT, exist_ok=True)
anno_df = pd.read_csv(ANNO_CSV)
print(f"[anno] {ANNO_CSV}: {len(anno_df)} feature rows, {anno_df['model'].nunique()} models")
for m in model_list:
    n_col = int((anno_df['model'] == m).sum())
    if n_col == 0:
        print(f"  [warn] no columns for model {m} in {ANNO_CSV} - this column will equal DeepACE")
knockout_cols = list(model_list) + ["DeepACE"]
SINGLE_COLS = [f"{m}_only" for m in SINGLE_MODEL_COLS]
if SINGLE_BARPLOT:
    for _m in SINGLE_MODEL_COLS:
        if _m not in model_list:
            raise KeyError(f"'{_m}' in SINGLE_MODEL_COLS is not in model_list; available: {model_list}")
for _m in HEATMAP_MODELS:
    if _m not in model_list:
        raise KeyError(f"'{_m}' in HEATMAP_MODELS is not in model_list; available: {model_list}")
HEATMAP_COLS = list(HEATMAP_MODELS) + ["DeepACE"]


def build_column_specs():
    specs = [(c, list(model_list) if c == "DeepACE" else [m for m in model_list if m != c],
              "loo", knockout_cols.index(c)) for c in knockout_cols]
    if SINGLE_BARPLOT:
        specs += [(c, [m], "single", i) for i, (c, m) in enumerate(zip(SINGLE_COLS, SINGLE_MODEL_COLS))]
    return specs

column_specs = build_column_specs()


''' ----------------  (RESUME) ---------------- '''

RESUME_DIR = os.path.join(SUPP_ROOT, ".resume")
LOO_CSV_COLS = list(knockout_cols)
SINGLE_CSV_COLS = list(SINGLE_COLS) + ["DeepACE"]

def _fingerprint(arm_name, kind):
    cfg = ARMS[arm_name]
    payload = {
        "kind": kind,
        "arm": arm_name,
        "metric": cfg["metric"],
        "bg_kind": cfg["bg_kind"],
        "bg_path": cfg.get("bg_path"),
        "bg_tmpl": cfg.get("bg_tmpl"),
        "anno_csv": ANNO_CSV,
        "sat_npy": SAT_NPY_TMPL,
        "sat_tsv": SAT_TSV_TMPL,
        "motifs": list(MOTIF_LIST),
        "models": list(model_list),
        "columns": list(LOO_CSV_COLS if kind == "loo" else SINGLE_CSV_COLS),
        "n_components": N_COMPONENTS,
        "pca_random_state": PCA_RANDOM_STATE,
        "use_fast_similarity": bool(USE_FAST_SIMILARITY),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()

def _manifest_path(arm_name, kind):
    return os.path.join(RESUME_DIR, f"S26_resume_{arm_name}_{kind}.json")

def _read_manifest(arm_name, kind):
    p = _manifest_path(arm_name, kind)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("fingerprint")
    except Exception:
        return None

def _write_manifest(arm_name, kind):
    os.makedirs(RESUME_DIR, exist_ok=True)
    with open(_manifest_path(arm_name, kind), "w", encoding="utf-8") as f:
        json.dump({"arm": arm_name, "kind": kind,
                   "fingerprint": _fingerprint(arm_name, kind),
                   "written_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                  f, ensure_ascii=False, indent=2)


def try_load_matrix(path, columns, arm_name, kind, n_take=None):
    name = os.path.basename(path)
    if not RESUME:
        print(f"[resume] RESUME=False -> {name} will be recomputed as usual")
        return None
    if not os.path.exists(path):
        print(f"[resume] {name} not found -> needs computing")
        return None
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as e:
        print(f"[resume][warn] {name} exists but cannot be read ({e}) -> recompute")
        return None
    if df.index.tolist() != list(MOTIF_LIST):
        print(f"[resume][warn] row index of {name} does not match current MOTIF_LIST -> recompute")
        return None
    if df.columns.tolist() != list(columns):
        print(f"[resume][warn] columns of {name} {df.columns.tolist()} do not match current config {list(columns)} "
              f"-> recompute")
        return None
    fp_old = _read_manifest(arm_name, kind)
    if fp_old is None:
        print(f"[resume] reusing {name} ({df.shape[0]}x{df.shape[1]}) - but it has no fingerprint record "
              f"(probably written by an earlier version of the script); please confirm it was computed with the current config; "
              f"if unsure, set RESUME to False.")
    elif fp_old != _fingerprint(arm_name, kind):
        print(f"[resume][warn] fingerprint of {name} does not match current config (background/model table/MOTIF_LIST changed?) -> recompute")
        return None
    else:
        print(f"[resume] reusing {name} ({df.shape[0]}x{df.shape[1]}, fingerprint matches)")
    vals = df.values.astype(float)
    return vals if n_take is None else vals[:, :n_take]


arms_run = [a for a in ARMS_TO_RUN]
for a in arms_run:
    if a not in ARMS:
        raise KeyError(f"'{a}' in ARMS_TO_RUN is not defined in ARMS (available: {list(ARMS)})")
auc_matrices = {a: np.full((len(MOTIF_LIST), len(knockout_cols)), np.nan) for a in arms_run}
single_matrices = {a: np.full((len(MOTIF_LIST), len(SINGLE_COLS)), np.nan) for a in arms_run}
print("\n[resume] checking existing matrices (RESUME=%s):" % RESUME)

for a in arms_run:
    v = try_load_matrix(f"{SUPP_ROOT}/S26_loo_quantile_auc_{a}_cagi5.csv",
                        LOO_CSV_COLS, a, "loo")
    if v is not None:
        auc_matrices[a] = v
    if SINGLE_BARPLOT:
        v2 = try_load_matrix(f"{SUPP_ROOT}/S26_single_model_vs_deepace_{a}.csv",
                             SINGLE_CSV_COLS, a, "single", n_take=len(SINGLE_COLS))
        if v2 is not None:
            single_matrices[a] = v2


def pending_specs(arm_name, motif_idx):
    todo = []
    for spec in column_specs:
        mat = auc_matrices[arm_name] if spec[2] == "loo" else single_matrices[arm_name]
        if np.isnan(mat[motif_idx, spec[3]]):
            todo.append(spec)
    return todo


n_planned = len(arms_run) * len(MOTIF_LIST) * len(column_specs)
n_have = sum(int(np.isfinite(m).sum()) for m in auc_matrices.values()) \
    + sum(int(np.isfinite(m).sum()) for m in single_matrices.values())
n_todo = n_planned - n_have
pending_arms = [a for a in arms_run
                if any(pending_specs(a, i) for i in range(len(MOTIF_LIST)))]
pending_motifs = [m for i, m in enumerate(MOTIF_LIST)
                  if any(pending_specs(a, i) for a in pending_arms)]
if n_todo:
    preflight_arms(pending_arms, pending_motifs)
else:
    print("[preflight] all matrices can be reused -> skipping background and prediction files, going straight to plotting\n")
shared_bg_raw = None


def get_shared_bg():
    global shared_bg_raw
    if not RAND_BG_CACHE:
        return np.load(RAND_BG_PATH)
    if shared_bg_raw is None:
        shared_bg_raw = np.load(RAND_BG_PATH)
        print(f"[bg] {RAND_BG_PATH} shape={shared_bg_raw.shape} (kept resident in memory, see RAND_BG_CACHE)")
    return shared_bg_raw


print("\n" + "=" * 78)
print("GENERATING LEAVE-ONE-MODEL-OUT QUANTILE-AUC MATRICES (CAGI5, 15 datasets)")
print(f"arms = {arms_run}  (planned {n_planned} scoring runs; "
      f"per motif per arm {len(knockout_cols)} LOO columns + {len(SINGLE_COLS) if SINGLE_BARPLOT else 0} single-model columns)")
print(f"{n_have} runs already ({100.0 * n_have / max(n_planned, 1):.0f}%) -> {n_todo} runs to compute this time")
print("=" * 78)

n_computed = {(a, k): 0 for a in arms_run for k in (["loo", "single"] if SINGLE_BARPLOT else ["loo"])}
selfcheck_done = {a: not VERIFY_FIRST_MOTIF for a in arms_run}

for motif_idx, motif in enumerate(MOTIF_LIST):
    t_motif = time.time()
    todo_by_arm = {a: pending_specs(a, motif_idx) for a in arms_run}
    if not any(todo_by_arm.values()):
        print(f"\n[{motif_idx + 1}/{len(MOTIF_LIST)}] {motif}: both arms already calculated -> skipping entire row", flush=True)
        continue

    alt_raw, ref_raw, labels, valid_cols = load_saturation_raw(motif)
    alt_data = alt_raw[:, valid_cols]
    ref_data = ref_raw[:, valid_cols]
    n_alt = len(alt_data)
    print(f"\n[{motif_idx + 1}/{len(MOTIF_LIST)}] {motif}: "
          f"alt={alt_data.shape} ref={ref_data.shape} labels={len(labels)}", flush=True)

    if len(labels) != n_alt:
        raise ValueError(f"{motif}: number of labels {len(labels)} does not match number of alt rows {n_alt} - "
                         f"the prediction file and tsv are not from the same run; align these two files before running.")
    nan_eff = int(np.isnan(labels).sum())
    if nan_eff:
        print(f"  [warn] {motif}: {nan_eff} NaNs in labels - quantile AUC will be underestimated, please check the tsv.")

    for arm_name in arms_run:
        arm_cfg = ARMS[arm_name]
        metric = arm_cfg["metric"]
        todo = todo_by_arm[arm_name]
        if not todo:
            print(f"  --- arm={arm_name}: already saved -> skip ---", flush=True)
            continue
        t_arm = time.time()
        rand_data = load_arm_background(
            arm_name, motif, valid_cols,
            get_shared_bg() if arm_cfg["bg_kind"] == "shared" else None)
        print(f"  --- arm={arm_name} (metric={metric}) background shape={rand_data.shape} "
              f"computing {len(todo)}/{len(column_specs)} columns this run ---", flush=True)

        nan_rand = int(np.isnan(rand_data).sum())
        if nan_rand:
            print(f"  [warn] {motif} / {arm_name}: {nan_rand} NaNs remain in the background set over retained columns - "
                  f"PCA will most likely fail directly, meaning this background prediction itself needs to be rerun.")

        for col, selected_models, target, j in todo:
            t0 = time.time()

            f_alt = filter_data_by_models(alt_data, selected_models, anno_df)
            f_ref = filter_data_by_models(ref_data, selected_models, anno_df)
            f_rand = filter_data_by_models(rand_data, selected_models, anno_df)

            if len(f_alt) == 0:
                print(f"  [warn] {motif} / {arm_name} / {col}: alt is null, recorded as NaN")
                continue

            combined = np.vstack([f_alt, np.repeat(f_ref, len(f_alt), axis=0), f_rand])
            if combined.shape[1] < N_COMPONENTS:
                uni_selected = combined
                print(f"  [warn] {motif} / {arm_name} / {col}: only {combined.shape[1]} columns, "
                      f"skipping PCA and using the original space")
            else:
                uni_selected = PCA(n_components=N_COMPONENTS,
                                   random_state=PCA_RANDOM_STATE).fit_transform(combined)

            pred_alt_pca = uni_selected[:len(f_alt)]
            pred_rand_pca = uni_selected[len(f_alt) + len(f_alt):]   
            if (not selfcheck_done[arm_name] and motif_idx == 0
                    and target == "loo" and metric == "mahalanobis"):
                check_fast_similarity(pred_alt_pca, pred_rand_pca, VERIFY_N_ALT)
                selfcheck_done[arm_name] = True

            similarity_all = similarity_scores(pred_alt_pca, pred_rand_pca, metric=metric)
            scores = -np.asarray(similarity_all)
            auc = quantile_auc(scores, labels)
            if target == "loo":
                auc_matrices[arm_name][motif_idx, j] = auc
            else:
                single_matrices[arm_name][motif_idx, j] = auc
            n_computed[(arm_name, target)] += 1
            print(f"    [{arm_name}] {col:14s} AUC = {auc:.4f}   ({time.time() - t0:.1f}s)", flush=True)
        print(f"  [{motif} / {arm_name}] cost {time.time() - t_arm:.1f}s", flush=True)
    print(f"  [{motif}] cost {time.time() - t_motif:.1f}s", flush=True)

auc_csv_by_arm = {}
for arm_name in arms_run:
    auc_df = pd.DataFrame(auc_matrices[arm_name], index=MOTIF_LIST, columns=knockout_cols)
    auc_df.index.name = "Motif"
    auc_csv = f"{SUPP_ROOT}/S26_loo_quantile_auc_{arm_name}_cagi5.csv"
    auc_df.to_csv(auc_csv, float_format='%.6f')
    auc_csv_by_arm[arm_name] = auc_csv
    print(f"\n[Saved] leave-one-model-out AUC matrix ({arm_name}) -> {auc_csv}")

single_csv_by_arm = {}
if SINGLE_BARPLOT:
    for arm_name in arms_run:
        single_df = pd.DataFrame(single_matrices[arm_name], index=MOTIF_LIST, columns=SINGLE_COLS)
        single_df.index.name = "Motif"
        single_df["DeepACE"] = auc_matrices[arm_name][:, knockout_cols.index("DeepACE")]
        single_csv = f"{SUPP_ROOT}/S26_single_model_vs_deepace_{arm_name}.csv"
        single_df.to_csv(single_csv, float_format='%.6f')
        single_csv_by_arm[arm_name] = single_csv
        print(f"[Saved] single-model AUC table ({arm_name}) -> {single_csv}")

print("\n" + "-" * 78)
print(f"[resume] computed {sum(n_computed.values())} runs this time (planned {n_planned}, "
      f"reused {n_planned - sum(n_computed.values())})")
for arm_name in arms_run:
    for kind in (["loo", "single"] if SINGLE_BARPLOT else ["loo"]):
        if n_computed[(arm_name, kind)] > 0:
            _write_manifest(arm_name, kind)
            print(f"[resume] wrote fingerprint {arm_name}/{kind} "
                  f"({n_computed[(arm_name, kind)]} columns computed this run) -> {_manifest_path(arm_name, kind)}")
        elif _read_manifest(arm_name, kind) == _fingerprint(arm_name, kind):
            _write_manifest(arm_name, kind)   # content unchanged, just refreshing the timestamp
            print(f"[resume] fingerprint already exists and matches {arm_name}/{kind} (entire matrix reused, no recompute)")
        else:
            print(f"[resume] {arm_name}/{kind} was fully reused from disk, fingerprint **not** written "
                  f"(cannot confirm it was computed with the current config); rerun with RESUME=False to certify it.")
print("-" * 78)


''' ================= Part 2 ~ Part 4: Plotting Figures ================='''

def make_outputs(arm_name, auc_csv):
    print("\n" + "#" * 78)
    print(f"# plots and diagnostics: arm = {arm_name}   (metric = {ARMS[arm_name]['metric']}, "
          f"background = {ARMS[arm_name]['bg_desc']})")
    print("#" * 78)

    ''' ---------------- Part 2: abs_ratio = |AUC| / |DeepACE| × sign(AUC) ----------------'''
    auc_df = pd.read_csv(auc_csv, index_col=0)
    auc_df = auc_df.astype(float)
    auc_df.index.name = "Motif"   
    deepace_auc_knockout = auc_df["DeepACE"].values
    other_cols_knockout = [c for c in auc_df.columns if c != "DeepACE"]
    auc_values_knockout = auc_df[other_cols_knockout].values

    abs_ratio = np.abs(auc_values_knockout) / (np.abs(deepace_auc_knockout)[:, np.newaxis] + 1e-12)
    signs = np.sign(auc_values_knockout)
    color_intensity = abs_ratio * signs
    deepace_intensity = np.ones((len(auc_df), 1))
    intensity_matrix = np.hstack([color_intensity, deepace_intensity])
    ratio_mat = pd.DataFrame(intensity_matrix, index=auc_df.index, columns=auc_df.columns)

    average_row = auc_df.mean()
    average_row.name = "Average"
    auc_df_plot = pd.concat([auc_df, average_row.to_frame().T], axis=0)
    mean_row = intensity_matrix.mean(axis=0)
    intensity_matrix = np.vstack([intensity_matrix, mean_row])

    ''' ---------------- Part 3a: heatmap ---------------- '''

    heat_intensity = intensity_matrix[:, [knockout_cols.index(c) for c in HEATMAP_COLS]]
    plot_df = auc_df_plot[HEATMAP_COLS].copy()
    plot_df.columns = [COL_LABEL_MAP.get(c, c) for c in plot_df.columns]
    fig, ax = plt.subplots(figsize=(len(plot_df.columns) * 1.5 + 4, 8))
    vmin, vmax = HEATMAP_VMIN, HEATMAP_VMAX
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")   

    for i in range(len(plot_df)):
        for j in range(len(plot_df.columns)):
            auc_val = plot_df.iloc[i, j]
            intensity = heat_intensity[i, j]

            if np.isnan(auc_val):
                color = 'lightgray'
                text = 'NaN'
            else:
                color = cmap(norm(intensity))
                text = f"{intensity:.3f}"
            ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                       facecolor=color, edgecolor='gray', linewidth=0.5))
            fontweight = 'normal'
            color_text = 'white' if abs(intensity) > 1.05 or abs(intensity) < 0.95 else 'black'
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                    fontsize=12, fontweight=fontweight, color=color_text)

    ax.set_xlim(0, len(plot_df.columns))
    ax.set_ylim(0, len(plot_df))
    ax.set_xticks(np.arange(len(plot_df.columns)) + 0.5)
    ax.set_yticks(np.arange(len(plot_df)) + 0.5)
    tmp_xlabels = plot_df.columns.str.replace('_', '\n')
    ax.set_xticklabels(tmp_xlabels, rotation=0, ha='center', fontsize=12)
    ax.set_yticklabels(plot_df.index, fontsize=12)
    ax.invert_yaxis()

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label(CBAR_LABEL, rotation=270, labelpad=15, fontsize=11)
    cbar.set_ticks(CBAR_TICKS)
    cbar.set_ticklabels(CBAR_TICKLABELS)
    plt.tight_layout()
    heatmap_path = save_all_formats(
        plt, os.path.join(SUPP_ROOT, f"S26_loo_quantile_auc_{arm_name}_heatmap"),
        dpi=400, bbox_inches='tight')
    plt.close()
    print(f"[Saved] leave-one-model-out heatmap ({arm_name}) -> {heatmap_path}  "
          f"[{','.join(SAVE_FORMATS)}]")

    # saving ratio csv
    ratio_df_knockout = pd.DataFrame(intensity_matrix, index=auc_df_plot.index, columns=auc_df_plot.columns)
    ratio_csv_knockout = os.path.join(SUPP_ROOT, f"S26_loo_quantile_auc_{arm_name}_ratio.csv")
    ratio_df_knockout.to_csv(ratio_csv_knockout, float_format='%.6f')
    print(f"[Saved] normalized ratio matrix ({arm_name}) -> {ratio_csv_knockout}")

    ''' ---------------- Part 3c: Barplot (average of 15 enhancer/promoters) ----------------'''

    mean_ratio = ratio_mat[list(model_list)].mean(axis=0).sort_values(ascending=False)
    x_order = mean_ratio.index.tolist()[::-1]           
    y_delta = (mean_ratio.loc[x_order].values - 1.0)
    bar_colors = [MEAN_DELTA_NEG_COLOR if v < 0 else MEAN_DELTA_POS_COLOR for v in y_delta]

    plt.figure(figsize=(16, 8))
    bars = plt.bar(x_order, y_delta, color=bar_colors, edgecolor='black', linewidth=1.2)
    plt.axhline(0.0, color='black', linewidth=1.0)
    plt.bar_label(bars, labels=[f"{v:+.4f}" for v in y_delta], fontsize=10, padding=3)
    plt.xlabel('Removed model (leave-one-model-out)', fontsize=12)
    plt.ylabel('Δ Normalized AUC\n(relative to DeepACE)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    mean_delta_path = save_all_formats(
        plt, os.path.join(SUPP_ROOT, f"S26_loo_quantile_auc_{arm_name}_mean_delta_barplot"),
        dpi=400, bbox_inches='tight')
    plt.close()
    print(f"[Saved] mean Δ barplot ({arm_name}) -> {mean_delta_path}  [{','.join(SAVE_FORMATS)}]")

    ''' ---------------- Part 4: Analysis ---------------- '''

    ratio_other = ratio_mat[other_cols_knockout]
    auc_other = auc_df[other_cols_knockout]

    summary = pd.DataFrame({
        "mean_auc": auc_other.mean(axis=0),
        "n_motif_finite": auc_other.notna().sum(axis=0),
        "delta_mean_auc": (auc_other - auc_df["DeepACE"].values[:, None]).mean(axis=0),
        "mean_ratio_vs_deepace": ratio_other.mean(axis=0),
        "min_ratio_vs_deepace": ratio_other.min(axis=0),
    })
    if CHANCE_CORRECTED:
        _gap = auc_df["DeepACE"].values - AUC_CHANCE_LEVEL
        _gap_safe = np.where(np.abs(_gap) < 0.01, np.nan, _gap)
        _tmp = (auc_other.values - AUC_CHANCE_LEVEL) / _gap_safe[:, None]
        summary["mean_chance_corrected_ratio"] = pd.DataFrame(_tmp).mean(axis=0).values  # 跳过 NaN, 不报警
    summary["column"] = summary.index
    summary = summary[["column"] + [c for c in summary.columns if c != "column"]]
    summary = summary.sort_values("mean_ratio_vs_deepace", ascending=True)
    summary_path = f"{SUPP_ROOT}/S26_loo_quantile_auc_{arm_name}_summary.csv"
    summary.to_csv(summary_path, float_format='%.6f')

    print("\n" + "=" * 78)
    print(f"[{arm_name}] DeepACE (using all 16 models) motif-averaged AUC = "
          f"{auc_df['DeepACE'].mean():.4f}   (chance level {AUC_CHANCE_LEVEL:.2f})")
    print(f"[{arm_name}] DeepACE's own AUC range across the 15 motifs: "
          f"{auc_df['DeepACE'].min():.4f} ~ {auc_df['DeepACE'].max():.4f}")
    print("=" * 78)
    print(f"[{arm_name}] After removing each model (mean_ratio closer to 1 means the model can be dropped):")
    print(summary.to_string(index=False))
    print(f"\n-> {summary_path}")

    print(f"\n[{arm_name}] 3 models whose removal hurts AUC the most (most critical models):")
    for i, (m, r) in enumerate(summary.head(3)[["column", "mean_ratio_vs_deepace"]].itertuples(index=False), 1):
        print(f"  {i}. {m}: x{r:.4f}")
    print(f"[{arm_name}] 3 models whose removal barely changes AUC (most replaceable contributions):")
    for i, (m, r) in enumerate(summary.tail(3)[["column", "mean_ratio_vs_deepace"]].iloc[::-1].itertuples(index=False), 1):
        print(f"  {i}. {m}: x{r:.4f}")

    hidden_cols = [c for c in knockout_cols if c not in HEATMAP_COLS]
    hidden_mean = ratio_mat[hidden_cols].mean(axis=0)
    print(f"\n[{arm_name}] the {len(hidden_cols)} LOO columns not shown in the heatmap, motif-averaged AUC / DeepACE:")
    for c in hidden_cols:
        print(f"  {c:14s} x{hidden_mean[c]:.4f}")
    print(f"[{arm_name}] range of those means: x{hidden_mean.min():.4f} ~ x{hidden_mean.max():.4f}")

    if np.isnan(auc_df[knockout_cols].values).any():
        print(f"\n[warn] {arm_name}: the matrix contains NaNs - first check which motif is missing labels/predictions; "
              f"do not interpret NaN as 0.")

    return auc_df


auc_df_by_arm = {}
for arm_name in arms_run:
    auc_df_by_arm[arm_name] = make_outputs(arm_name, auc_csv_by_arm[arm_name])


''' ================= Part 5: Borzoi / Enformer only comparison =================

'''

def make_single_vs_deepace_outputs(arm_name, single_csv, loo_csv):
    print("\n" + "#" * 78)
    print(f"# Part 5 single-model showdown: arm = {arm_name}  "
          f"(using only {' / using only '.join(SINGLE_MODEL_COLS)}, reference = DeepACE with all 16 models)")
    print("#" * 78)

    auc_loo = pd.read_csv(loo_csv, index_col=0).astype(float)
    auc_single = pd.read_csv(single_csv, index_col=0).astype(float)

    df = auc_single[SINGLE_COLS].copy()
    df["DeepACE"] = auc_loo["DeepACE"].values

    deepace_auc = df["DeepACE"].values
    vals = df[SINGLE_COLS].values
    ratio = np.abs(vals) / (np.abs(deepace_auc)[:, np.newaxis] + 1e-12) * np.sign(vals)

    df_plot = pd.concat([df, df.mean().rename("Average").to_frame().T], axis=0)
    ratio_plot = pd.DataFrame(
        np.vstack([ratio, ratio.mean(axis=0)[np.newaxis, :]]),
        index=df_plot.index, columns=SINGLE_COLS)
    ratio_csv = f"{SUPP_ROOT}/S26_single_model_vs_deepace_{arm_name}_ratio.csv"
    ratio_plot.to_csv(ratio_csv, float_format='%.6f')
    print(f"[Saved] single-model ratio matrix ({arm_name}) -> {ratio_csv}")

    plot_df = pd.read_csv(ratio_csv, index_col=0)
    plot_df.index = [str(i).replace('_', '\n') for i in plot_df.index]
    plot_df.index.name = "Motif"
    plot_long = plot_df.reset_index().melt(id_vars="Motif", value_vars=SINGLE_COLS,
                                           var_name="Model", value_name="Normalized AUC")
    plot_long['Δ Normalized AUC'] = plot_long["Normalized AUC"] - 1    

    plt.figure(figsize=(16, 8))
    sns.barplot(
        data=plot_long,
        x="Motif",
        y='Δ Normalized AUC',
        hue="Model",
        palette=SINGLE_BAR_COLORS,
        edgecolor='black',
        linewidth=1.2,
        errorbar=None)
    plt.xlabel(SINGLE_XLABEL, fontsize=12)
    plt.ylabel(SINGLE_YLABEL, fontsize=12)
    plt.xticks(fontsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = []
    for lab in labels:
        model = next((m for m, c in zip(SINGLE_MODEL_COLS, SINGLE_COLS) if c == lab), lab)
        new_labels.append(SINGLE_LABEL_TMPL.format(model=model))
    plt.legend(handles, new_labels, title=SINGLE_LEGEND_TITLE,
               bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    bar_path = save_all_formats(
        plt, os.path.join(SUPP_ROOT, f"S26_single_model_vs_deepace_{arm_name}_barplot"),
        dpi=400, bbox_inches='tight')
    plt.close()
    print(f"[Saved] single-model PK barplot ({arm_name}) -> {bar_path}  [{','.join(SAVE_FORMATS)}]")

    print(f"[{arm_name}] reference DeepACE mean AUC for single-model showdown = {df['DeepACE'].mean():.4f}  "
          f"(using all 16 models)")
    for col, m in zip(SINGLE_COLS, SINGLE_MODEL_COLS):
        r = ratio_plot.loc[df.index, col]
        print(f"[{arm_name}] using only {m:9s} mean AUC = {df[col].mean():.4f}   "
              f"mean ratio = {r.mean():.4f} (Δ = {r.mean() - 1:+.4f})   "
              f"{int((r > 1.0).sum())} of 15 datasets with ratio > 1")
    if np.isnan(df[SINGLE_COLS].values).any():
        print(f"[warn] {arm_name}: NaNs in the single-model table - first check which motif is missing labels/predictions "
              f"(same batch of data as the LOO table, NaNs will appear there too).")

    return df_plot, ratio_plot


if SINGLE_BARPLOT:
    single_ratio_by_arm = {}
    for arm_name in arms_run:
        _, _ratio_plot = make_single_vs_deepace_outputs(
            arm_name, single_csv_by_arm[arm_name], auc_csv_by_arm[arm_name])
        single_ratio_by_arm[arm_name] = _ratio_plot
else:
    single_ratio_by_arm = {}

if len(arms_run) > 1:
    print("\n" + "=" * 78)
    print("Two-arm comparison (same CAGI5 data / same LOO and AUC pipeline; only background and metric differ)")
    print("=" * 78)
    for arm_name in arms_run:
        _d = auc_df_by_arm[arm_name]
        print(f"  {arm_name:8s} ({ARMS[arm_name]['metric']:11s})  "
              f"DeepACE = {_d['DeepACE'].mean():.4f}   "
              f"mean of the 16 LOO columns = {_d[list(model_list)].mean().mean():.4f}   "
              f"model with the lowest LOO column = {_d[list(model_list)].mean().idxmin()}")
        if single_ratio_by_arm:
            _r = single_ratio_by_arm[arm_name]
            print("           using single models only: " + "   ".join(
                f"{m} x{_r[c].iloc[:-1].mean():.4f}" for c, m in zip(SINGLE_COLS, SINGLE_MODEL_COLS)))
    print("  (AUCs of the two arms are not comparable cell by cell - the background clouds differ, as do the "
          "\n   score scales; what should be compared is the ranking of ratios and each arm's DeepACE reference column.)")

