import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import numpy as np
import pandas as pd
import os
import sys
import random

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import mahalanobis
from scipy.stats import pearsonr
import joblib

random.seed(42)
np.random.seed(42)


SUPP_ROOT = "./Supps/S13_barplot_motif_substitution_random"
SCORES_DIR = f"{SUPP_ROOT}/scores"
SUMMARY_DIR = f"{SUPP_ROOT}/summary"
SAVE_FORMATS = ["pdf", "png", "svg"]
SAVE_DPI = 400                     
MOTIF_LIST = ["ELF1", "HNF1A", "HNF4A"]     
RANDAUG_ROOT_CANDIDATES = [
    "/home/hyu/DeepACE/Preds/D15_extra_motif_randaug",   
    "./Preds/D15_extra_motif_randaug",
]
RANDAUG_ROOT_PROBE = "S13_randaug_blocks.csv"            
RANDAUG_SEED_TAG = "43-44-45"                           
RANDAUG_DIR_TMPL = "motif_Epigenetics_{motif}_randaug_seed" + RANDAUG_SEED_TAG
REPLICATES = [
    # (label,             random_sample,       randaug_block)
    ("random_sample_2", "random_sample_2", 0),      # seed 43
    ("random_sample_3", "random_sample_3", 1),      # seed 44
    ("random_sample_4", "random_sample_4", 2),      # seed 45
]


DATA_ROOT_CANDIDATES = ["./Preds/D04_deeptfbu", "./fig_dataset"]
DATA_ROOT_PROBE = "motif_Epigenetics_ELF1_alt.csv"
RANDOM_ROOT_CANDIDATES = ["./Preds/D10_random", "./fig_dataset"]
RANDOM_ROOT_PROBE = "random_sample_1/uni_pred.npy"
COLD_PCA_CANDIDATES = [
    "./Preds/D01_screens/pca_model.pkl",    
    "./fig1a_pca_analysis/pca_model.pkl",  
]
ANALYSIS_ROOT = "./Preds/D04_deeptfbu"      
ORIGINAL_REPEAT = "random_sample_1"
INCLUDE_ORIGINAL_REPEAT = True   
ORIGINAL_SCORE_SPECS = [
    ("DeepACE-randaug", "analysis_cosine",      "pca50_variant_scores_{motif}.csv"),
    ("DeepACE-random",  "analysis_mahalanobis", "pca50_variant_scores_{motif}.csv"),
    ("DeepACE-cold",    "analysis_cold",        "pca50_variant_scores_{motif}.csv"),
]
SPREAD_REPEATS = ([ORIGINAL_REPEAT] if INCLUDE_ORIGINAL_REPEAT else []) + [l for l, _, _ in REPLICATES]
BASELINE_SPECS = [
    ("Evo2",            "analysis_evo2",       "evo2_variant_scores_{motif}.csv"),
    ("promoterAI",      "analysis_promoterai", "promoterAI_variant_scores_{motif}.csv"),
    ("phyloP100way",    "analysis_cons",       "phyloP100way_variant_scores_{motif}.csv"),
    ("phyloP470way",    "analysis_cons",       "phyloP470way_variant_scores_{motif}.csv"),
    ("phastCons100way", "analysis_cons",       "phastCons100way_variant_scores_{motif}.csv"),
    ("phastCons470way", "analysis_cons",       "phastCons470way_variant_scores_{motif}.csv"),
    ("gpnmsa",          "analysis_gpnmsa",     "gpnmsa_variant_scores_{motif}.csv"),
]

DEEPACE_ALGOS = ["DeepACE-randaug", "DeepACE-random", "DeepACE-cold"]
ALL_ALGOS = DEEPACE_ALGOS + [name for name, _, _ in BASELINE_SPECS]


def resolve_root(candidates, probe, what):
    for root in candidates:
        if os.path.exists(os.path.join(root, probe)):
            print(f"[path] {what}: {root}")
            return root
    msg = [f"Cannot find {what}, tried:"]
    msg += [f"    {os.path.join(c, probe)}" for c in candidates]
    raise FileNotFoundError("\n".join(msg))


def resolve_file(candidates, what):
    for path in candidates:
        if os.path.exists(path):
            print(f"[path] {what}: {path}")
            return path
    msg = [f"Cannot find {what}, tried:"] + [f"    {c}" for c in candidates]
    raise FileNotFoundError("\n".join(msg))


if hasattr(np, "trapz"):
    trapz = np.trapz
else:                      
    trapz = np.trapezoid


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
            sims_matrix = cosine_similarity(pred_alt, pred_ref)  # (N, M)
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
    auc = trapz(discover_rates, quantile_grid)

    return auc


def calculate_pcc_per_motif(scores, effects):
    scores = np.asarray(scores, dtype=np.float64)
    effects = np.asarray(effects, dtype=np.float64)
    ok = np.isfinite(scores) & np.isfinite(effects)
    if int(ok.sum()) < 2:
        return np.nan
    return float(pearsonr(scores[ok], effects[ok])[0])


def group_label(group_name, motif, aim_cnt, geo_cnt):
    parts = group_name.split('_pos_')
    if len(parts) < 2:
        raise ValueError(f"group_name does not contain '_pos_': {group_name} ")
    tmp = parts[1]
    tmp = "_".join(tmp.split("_")[:-2])
    if 'aim' in tmp:
        aim_cnt += 1
        tmp = f"{motif}_aim{aim_cnt}"
    else:
        geo_cnt += 1
        tmp = f"{motif}_geo{geo_cnt}"
    return tmp, aim_cnt, geo_cnt


def build_rows(motif, label, kept_groups, sims_da, baseline_auc, baseline_pcc,
               df_alt_f02g, variant_effects_f02g):
    aim_cnt, geo_cnt = 0, 0
    rows = []
    for g in kept_groups:
        alt_indices = df_alt_f02g[df_alt_f02g["group_name"] == g].index.tolist()
        ve_tmp = variant_effects_f02g[alt_indices]
        label_g, aim_cnt, geo_cnt = group_label(g, motif, aim_cnt, geo_cnt)

        auc_map = {a: calculate_auc_per_motif(v[alt_indices], ve_tmp) for a, v in sims_da.items()}
        pcc_map = {a: calculate_pcc_per_motif(v[alt_indices], ve_tmp) for a, v in sims_da.items()}
        for algo, _, _ in BASELINE_SPECS:
            auc_map[algo] = baseline_auc[g][algo]
            pcc_map[algo] = baseline_pcc[g][algo]

        for algo in ALL_ALGOS:
            rows.append({"Group": label_g, "Algorithm": algo,
                         "AUC": auc_map[algo], "PCC": pcc_map[algo], "Name": g,
                         "Motif": motif, "Repeat": label})
    return rows


def load_original_deepace_scores(motif, valid_mask, n_raw):
    scores = {}
    for algo, sub_dir, fname in ORIGINAL_SCORE_SPECS:
        path = os.path.join(ANALYSIS_ROOT, sub_dir, fname.format(motif=motif))
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"lack of {path}\n"
            )
        sc = pd.read_csv(path)["scores"].to_numpy()
        if len(sc) == n_raw:
            sc = sc[valid_mask]                     
        print(f"    [path] {ORIGINAL_REPEAT} / {algo}: {path}  ({len(sc)} 行)")
        scores[algo] = sc
    return scores


def resolve_randaug_root():
    for root in RANDAUG_ROOT_CANDIDATES:
        if os.path.exists(os.path.join(root, RANDAUG_ROOT_PROBE)):
            print(f"[path] New randaug (motif-substitution): {root}")
            return root
    msg = ["Cannot find the new motif-substitution randaug, tried:"]
    msg += [f"    {os.path.join(c, RANDAUG_ROOT_PROBE)}" for c in RANDAUG_ROOT_CANDIDATES]
    msg += ["",
            "This is a prerequisite for this script: Please first run on the server",
            "    revisions/S13_randaug_fasta/make_randaug_fasta.py",
            "to generate the FASTA + CSV with 3 seed blocks merged (and S13_randaug_blocks.csv),",
            "then run the prediction for each motif as instructed by the prediction.sh it prints,",
            "so that predictions are saved to <OUT_DIR>/motif_Epigenetics_{motif}_randaug_seed43-44-45/uni_pred.npy."]
    raise FileNotFoundError("\n".join(msg))


_blocks_df = None

def randaug_blocks():
    global _blocks_df
    if _blocks_df is None:
        path = os.path.join(RANDAUG_ROOT, RANDAUG_ROOT_PROBE)
        _blocks_df = pd.read_csv(path)
        need = ("motif", "seed", "block", "idx_start", "idx_end")
        missing = [c for c in need if c not in _blocks_df.columns]
        if missing:
            raise KeyError(f"{path} missing columns {missing}; is this not a product of S13_randaug_fasta/make_randaug_fasta.py?")
    return _blocks_df


def randaug_block_range(motif, block_idx):
    sub = randaug_blocks()
    sub = sub[(sub["motif"] == motif) & (sub["block"] == block_idx)]
    if len(sub) != 1:
        raise KeyError(
            f"There are {len(sub)} rows for (motif={motif}, block={block_idx}) in {RANDAUG_ROOT_PROBE}, expected 1 row. "
            f"Please confirm that MOTIF_LIST used during FASTA generation covers {MOTIF_LIST}"
        )
    r = sub.iloc[0]
    return int(r["idx_start"]), int(r["idx_end"])



def save_scores(path, scores, variant_effects):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"scores": scores, "variant_effects": variant_effects}).to_csv(path, index=False)


def load_saved_scores(path, variant_effects):
    df = pd.read_csv(path)
    if len(df) != len(variant_effects) or not np.allclose(df["variant_effects"], variant_effects,
                                                         rtol=0, atol=1e-9, equal_nan=True):
        raise ValueError(f"{path} does not match the current alt/ref rows, delete it to recompute")
    return df["scores"].to_numpy()


# ============================================================
# Preparation
# ============================================================

DATA_ROOT = resolve_root(DATA_ROOT_CANDIDATES, DATA_ROOT_PROBE, "motif substitution data")
RANDOM_ROOT = resolve_root(RANDOM_ROOT_CANDIDATES, RANDOM_ROOT_PROBE, "random_sample background")
COLD_PCA_PATH = resolve_file(COLD_PCA_CANDIDATES, "frozen PCA model from cold branch")
RANDAUG_ROOT = resolve_randaug_root() if any(b is not None for _, _, b in REPLICATES) else None

for _d in [SUPP_ROOT, SCORES_DIR, SUMMARY_DIR]:
    os.makedirs(_d, exist_ok=True)

if RANDAUG_ROOT is None:
    print("[setting] All randaug_block values are None -> using the old full randaug (mode to reproduce original plots)")
else:
    print(f"[setting] New randaug directory: {RANDAUG_ROOT}; directory template: {RANDAUG_DIR_TMPL}")
if INCLUDE_ORIGINAL_REPEAT:
    print(f"[setting] Cross-replicate fluctuations include the 4th: {ORIGINAL_REPEAT} (directly reading "
          f"{ANALYSIS_ROOT}/analysis_{{cosine,mahalanobis,cold}}/, no plot generated)")
else:
    print("[setting] INCLUDE_ORIGINAL_REPEAT = False -> cross-replicate fluctuations only include the three recalculated replicates")
for _label, _rs, _blk in REPLICATES:
    print(f"[repeat] {_label:16s} random_sample={_rs:16s} randaug_block={_blk}")
print()


# ============================================================
# Main Process
# ============================================================

records = {label: {} for label in SPREAD_REPEATS}
for motif in MOTIF_LIST:
    print(f"=== Processing motif: {motif} ===")

    pred_alt = np.load(f"{DATA_ROOT}/motif_Epigenetics_{motif}_alt/uni_pred.npy")
    df_alt = pd.read_csv(f"{DATA_ROOT}/motif_Epigenetics_{motif}_alt.csv")
    alt_effects = df_alt['measured enhancer activity'].to_numpy()

    pred_ref = np.load(f"{DATA_ROOT}/motif_Epigenetics_{motif}_ref/uni_pred.npy")
    df_ref = pd.read_csv(f"{DATA_ROOT}/motif_Epigenetics_{motif}_ref.csv")
    ref_effects = df_ref['measured enhancer activity'].to_numpy()

    n_raw = len(df_alt)
    valid_mask = np.isfinite(alt_effects) & np.isfinite(ref_effects)
    if int((~valid_mask).sum()):
        print(f"    [info] {motif}: alt/ref measurements missing, dropped {int((~valid_mask).sum())} rows "
              f"({n_raw} -> {int(valid_mask.sum())})")
    alt_effects = alt_effects[valid_mask]
    ref_effects = ref_effects[valid_mask]
    df_alt = df_alt[valid_mask].reset_index(drop=True)

    pred_alt = pred_alt[valid_mask]
    pred_ref = pred_ref[valid_mask]
    variant_effects = np.log2(alt_effects + 1e-6) - np.log2(ref_effects + 1e-6)

    df_alt["group_name"] = df_alt["sequence_name"].str.split(r'_-_|_\+_', expand=True)[0]
    group_names_in_order = df_alt["group_name"].drop_duplicates(keep='first')

    df_baseline = {}
    for algo, sub_dir, fname in BASELINE_SPECS:
        df_baseline[algo] = pd.read_csv(
            os.path.join(ANALYSIS_ROOT, sub_dir, fname.format(motif=motif))
        )[["scores", "variant_effects"]]

    variant_effects_f02g = df_baseline["Evo2"]["variant_effects"].to_numpy()
    df_alt_f02g = df_alt
    variant_effects_f02g = variant_effects_f02g[valid_mask]
    for algo in df_baseline:
        df_baseline[algo] = df_baseline[algo][valid_mask].reset_index(drop=True)

    kept_groups = []
    for g in group_names_in_order:
        alt_indices = df_alt_f02g[df_alt_f02g["group_name"] == g].index.tolist()
        if len(alt_indices) < 10:
            continue
        if np.max(variant_effects_f02g[alt_indices]) - np.min(variant_effects_f02g[alt_indices]) < 0.5:
            continue
        kept_groups.append(g)
    print(f"    groups: {len(group_names_in_order)} -> keep {len(kept_groups)}")

    baseline_auc, baseline_pcc = {}, {}
    for g in kept_groups:
        alt_indices = df_alt_f02g[df_alt_f02g["group_name"] == g].index.tolist()
        ve_tmp = variant_effects_f02g[alt_indices]
        baseline_auc[g], baseline_pcc[g] = {}, {}
        for algo, _, _ in BASELINE_SPECS:
            sc_tmp = df_baseline[algo]["scores"].to_numpy()[alt_indices]
            baseline_auc[g][algo] = calculate_auc_per_motif(sc_tmp, ve_tmp)
            baseline_pcc[g][algo] = calculate_pcc_per_motif(sc_tmp, ve_tmp)

    if INCLUDE_ORIGINAL_REPEAT:
        sims_orig = load_original_deepace_scores(motif, valid_mask, n_raw)
        rows_orig = build_rows(motif, ORIGINAL_REPEAT, kept_groups,
                               {a: -v for a, v in sims_orig.items()},
                               baseline_auc, baseline_pcc, df_alt_f02g, variant_effects_f02g)
        records[ORIGINAL_REPEAT][motif] = rows_orig
        print(f"    [{ORIGINAL_REPEAT}] {motif}: {len(rows_orig)} row AUC/PCC record")

    if RANDAUG_ROOT is None:
        ra_stem = f"{DATA_ROOT}/motif_Epigenetics_{motif}_randaug"
    else:
        ra_stem = os.path.join(RANDAUG_ROOT, RANDAUG_DIR_TMPL.format(motif=motif))
        if not os.path.exists(os.path.join(ra_stem, "uni_pred.npy")):
            raise FileNotFoundError(
                f"Missing {os.path.join(ra_stem, 'uni_pred.npy')}.\n"
                f"This line's randaug requires running prediction once (the 15 you ran before are for the point mutation S23 line):\n"
                f"    First run revisions/S13_randaug_fasta/make_randaug_fasta.py,\n"
                f"    then run prediction for each motif as instructed by the prediction.sh it prints."
            )
    randaug_pred_full = np.load(os.path.join(ra_stem, "uni_pred.npy"))
    randaug_df_full = pd.read_csv(ra_stem + ".csv")

    if len(randaug_pred_full) != len(randaug_df_full):
        raise ValueError(
            f"{motif}: randaug prediction has {len(randaug_pred_full)} rows, but CSV has {len(randaug_df_full)} rows, mismatch.\n"
            f"  This line cannot do [:-1] (no global trailing ref record); if lengths differ, please confirm the prediction used\n"
            f"  is exactly the FASTA generated by make_randaug_fasta.py."
        )
    if not randaug_df_full.index.equals(pd.RangeIndex(len(randaug_df_full))):
        raise ValueError(f"{motif}: randaug CSV index is not 0..n-1, the premise of taking values by position below does not hold")


    randaug_names = randaug_df_full["sequence_name"].tolist()
    randaug_df_full["motif_ref"] = [item.split("_")[-2] for item in randaug_names]
    randaug_df_full["motif_alt"] = [item.split("_")[-1] for item in randaug_names]
    randaug_df_full["group_name"] = randaug_df_full["sequence_name"].str.split(r'_-_|_\+_', expand=True)[0]

    randaug_idxs = np.asarray(
        randaug_df_full[randaug_df_full["motif_ref"] != randaug_df_full["motif_alt"]].index
    )
    print(f"    randaug rows: total {len(randaug_df_full)} -> cross-motif {len(randaug_idxs)}")

    for label, random_sample, randaug_block in REPLICATES:
        random_path = os.path.join(RANDOM_ROOT, random_sample, "uni_pred.npy")
        if not os.path.exists(random_path):
            raise FileNotFoundError(
                f"Missing background sample {random_path}. If there is only 1 random_sample on this machine, "
                f"please modify REPLICATES to the actual existing samples"
            )
        pred_rand_rs = np.load(random_path)

        # ---- (a) DeepACE-random: fig2d  metric == "mahalanobis" branch ----
        mah_path = os.path.join(SCORES_DIR, label, "analysis_mahalanobis",
                                f"pca50_variant_scores_{motif}.csv")
        if os.path.exists(mah_path):
            print(f"    [reuse] {mah_path}")
            sim_random = load_saved_scores(mah_path, variant_effects)
        else:
            combined = np.vstack([pred_alt, pred_ref, pred_rand_rs])
            combined_pca = PCA(n_components=50, random_state=42).fit_transform(combined)
            alt_pca = combined_pca[:len(pred_alt)]
            rand_pca = combined_pca[-len(pred_rand_rs):]

            sim_all = []
            for g in group_names_in_order:
                alt_indices = df_alt_f02g[df_alt_f02g["group_name"] == g].index.tolist()
                sim_all.extend(
                    compute_sample_similarity(alt_pca[alt_indices], rand_pca,
                                              metric="mahalanobis", mode="batch")
                )
            sim_random = np.asarray(sim_all)
            save_scores(mah_path, sim_random, variant_effects)

        # ---- (b) DeepACE-cold: fig2g  metric == "mahalanobis" branch (frozen) ----
        cold_path = os.path.join(SCORES_DIR, label, "analysis_cold",
                                 f"pca50_variant_scores_{motif}.csv")
        if os.path.exists(cold_path):
            print(f"    [reuse] {cold_path}")
            sim_cold = load_saved_scores(cold_path, variant_effects)
        else:
            pca_cold = joblib.load(COLD_PCA_PATH)
            combined = np.vstack([pred_alt, pred_ref, pred_rand_rs])
            combined_pca_cold = pca_cold.transform(combined)
            alt_pca_cold = combined_pca_cold[:len(pred_alt)]
            rand_pca_cold = combined_pca_cold[-len(pred_rand_rs):]

            sim_all = []
            for g in group_names_in_order:
                alt_indices = df_alt_f02g[df_alt_f02g["group_name"] == g].index.tolist()
                sim_all.extend(
                    compute_sample_similarity(alt_pca_cold[alt_indices], rand_pca_cold,
                                              metric="mahalanobis", mode="batch")
                )
            sim_cold = np.asarray(sim_all)
            save_scores(cold_path, sim_cold, variant_effects)

        # ---- (c) DeepACE-randaug: fig2d  metric == "cosine" branch (randaug) ----
        cos_path = os.path.join(SCORES_DIR, label, "analysis_cosine",
                                f"pca50_variant_scores_{motif}.csv")
        if os.path.exists(cos_path):
            print(f"    [reuse] {cos_path}")
            sim_randaug = load_saved_scores(cos_path, variant_effects)
        else:
            if randaug_block is None:
                keep_idxs = randaug_idxs
                keep_names = randaug_df_full.loc[randaug_idxs, "group_name"].to_numpy()
            else:
                s, e = randaug_block_range(motif, randaug_block)
                keep_idxs = randaug_idxs[s:e]
                keep_names = randaug_df_full.loc[keep_idxs, "group_name"].to_numpy()
                print(f"    [{label} | {motif}] randaug block {randaug_block}: "
                      f"filtered {s}:{e}, {len(keep_idxs)} rows")

            pred_rand_ra = randaug_pred_full[keep_idxs]
            randaug_df = randaug_df_full.loc[keep_idxs, :].reset_index(drop=True)

            combined = np.vstack([pred_alt, pred_ref, pred_rand_ra])
            combined_pca = PCA(n_components=50, random_state=42).fit_transform(combined)
            alt_pca = combined_pca[:len(pred_alt)]
            rand_pca = combined_pca[-len(pred_rand_ra):]

            sim_all = []
            empty_groups = []
            for g in group_names_in_order:
                alt_indices = df_alt_f02g[df_alt_f02g["group_name"] == g].index.tolist()
                rand_indices = randaug_df[randaug_df["group_name"] == g].index.tolist()
                if len(rand_indices) == 0:
                    empty_groups.append(g)
                    sim_all.extend([np.nan] * len(alt_indices))
                    continue
                sim_all.extend(
                    compute_sample_similarity(alt_pca[alt_indices], rand_pca[rand_indices],
                                              metric="cosine", mode="batch")
                )
            if empty_groups:
                print(f"    [warn] {motif} | {label}: {len(empty_groups)} groups have no corresponding rows in randaug "
                      f"({empty_groups[:3]}...), DeepACE-randaug for these groups is recorded as NaN (the original script would crash here)")
            sim_randaug = np.asarray(sim_all)
            save_scores(cos_path, sim_randaug, variant_effects)

        sims_da = {"DeepACE-randaug": -sim_randaug,
                   "DeepACE-random": -sim_random,
                   "DeepACE-cold": -sim_cold}
        rows = build_rows(motif, label, kept_groups, sims_da,
                          baseline_auc, baseline_pcc, df_alt_f02g, variant_effects_f02g)
        records[label][motif] = rows
        print(f"    [{label}] {motif}: {len(rows)} rows of AUC/PCC records")


# ============================================================
# Summarize
# ============================================================

table_by_repeat = {}
for label in SPREAD_REPEATS:
    rows = []
    for motif in MOTIF_LIST:
        rows.extend(records[label][motif])
    df_rep = pd.DataFrame(rows)
    df_rep.to_csv(f"{SUMMARY_DIR}/S13_auc_table_{label}.csv", index=False)
    table_by_repeat[label] = df_rep

row_keys = ["Motif", "Group", "Algorithm"]
df_all = pd.concat([table_by_repeat[label] for label in SPREAD_REPEATS], ignore_index=True)

spread = df_all.pivot_table(index=row_keys, columns="Repeat", values="AUC", aggfunc="first")
spread = spread.reset_index()
repeat_cols = list(SPREAD_REPEATS)
spread["mean"] = spread[repeat_cols].mean(axis=1)
spread["min"] = spread[repeat_cols].min(axis=1)
spread["max"] = spread[repeat_cols].max(axis=1)
spread["spread"] = spread["max"] - spread["min"]
spread = spread[row_keys + repeat_cols + ["mean", "min", "max", "spread"]]
spread.round(6).to_csv(f"{SUMMARY_DIR}/S13_auc_spread_across_repeats.csv", index=False)

name_map = {
    "phyloP100way": "phyloP 100",
    "phyloP470way": "phyloP 470",
    "phastCons100way": "phastCons 100",
    "phastCons470way": "phastCons 470",
    "gpnmsa": "GPN-MSA",
}
df_root = df_all[df_all["Repeat"].isin([l for l, _, _ in REPLICATES])].copy()
df_root["Algorithm"] = df_root["Algorithm"].replace(name_map)
df_root = df_root[["Repeat", "Motif", "Group", "Algorithm", "AUC", "Name"]]
df_root.round(6).to_csv(f"{SUPP_ROOT}/S13_barplot_all.csv", index=False)


# ============================================================
# Plotting Figures
# ============================================================

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
    "GPN-MSA": "#636363",
}
sns.set_theme(style="ticks")

for label, _, _ in REPLICATES:
    rep_id = label.rsplit("_", 1)[-1]
    df = df_root[df_root["Repeat"] == label].copy()
    df[['TF_Category', 'Sample_ID']] = df['Group'].str.split('_', n=1, expand=True)
    for i, tf in enumerate(['ELF1', 'HNF1A', 'HNF4A'], start=1):
        subset = df[df['TF_Category'] == tf].copy()
        if subset.empty:
            continue
        plt.figure(figsize=(5, 7))
        sns.barplot(data=subset, y='Algorithm', x='AUC', palette=palette, edgecolor='black',
                    errorbar=None, alpha=0.9)
        sns.stripplot(data=subset, y='Algorithm', x='AUC', color='black', size=4,
                      jitter=0.15, linewidth=0.5)
        plt.title(f'{tf}', fontsize=20, pad=8, loc='left', x=-0.2)
        plt.xlabel('AUC Score', fontsize=20)
        plt.ylabel('', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlim(0.4, 1.0)
        sns.despine(left=True, bottom=False)
        plt.tight_layout()
        save_name = f"S13_barplot_all_{i}_{rep_id}"
        for ext in SAVE_FORMATS:
            plt.savefig(os.path.join(SUPP_ROOT, f"{save_name}.{ext}"),
                        dpi=SAVE_DPI, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_name} -> {SUPP_ROOT}/[{','.join(SAVE_FORMATS)}]  ")


# ============================================================
# Final
# ============================================================

print("\n==== Average DeepACE AUC per repeat (average by group first, then by motif; only includes the three plotted replicates) ====")
header = f"{'Repeat':18s}" + "".join([f"{a:>18s}" for a in DEEPACE_ALGOS]) + f"{'7 baseline mean':>16s}"
print(header)
base_names = [n for n, _, _ in BASELINE_SPECS]
for label, _, _ in REPLICATES:
    df_rep = df_root[df_root["Repeat"] == label]
    vals = [df_rep[df_rep["Algorithm"] == a]["AUC"].mean() for a in DEEPACE_ALGOS]
    base_val = df_rep[df_rep["Algorithm"].isin([name_map.get(n, n) for n in base_names])]["AUC"].mean()
    print(f"{label:18s}" + "".join([f"{v:18.4f}" for v in vals]) + f"{base_val:16.4f}")

df_mean = (df_all.groupby(["Repeat", "Motif", "Algorithm"], as_index=False)
           .agg(n_group=("Group", "nunique"),
                n_AUC=("AUC", "count"), n_PCC=("PCC", "count"),
                mean_AUC=("AUC", "mean"), mean_PCC=("PCC", "mean")))
df_mean.round(6).to_csv(f"{SUMMARY_DIR}/S13_mean_by_motif.csv", index=False)

print("\n" + "=" * 78)
print("Average AUC / PCC per algorithm under each motif (average by group; algorithm names match y-axis labels on the plot)")
print("=" * 78)
for label in SPREAD_REPEATS:
    print(f"\n### {label}" + ("   (= the repeat from original F02g plot, read from previous results)" if label == ORIGINAL_REPEAT else ""))
    for motif in MOTIF_LIST:
        sub = df_mean[(df_mean["Repeat"] == label) & (df_mean["Motif"] == motif)]
        if sub.empty:
            continue
        sub = sub.set_index("Algorithm").reindex(ALL_ALGOS)
        print(f"\n  {motif}  (this motif retains {int(sub['n_group'].max())} groups)")
        print(f"    {'Algorithm':22s}{'mean AUC':>10s}{'mean PCC':>10s}"
              f"{'n(AUC)':>8s}{'n(PCC)':>8s}")
        for algo in ALL_ALGOS:
            nm = name_map.get(algo, algo)
            r = sub.loc[algo]
            if not np.isfinite(r["n_AUC"]):
                print(f"    {nm:22s}{'--':>10s}{'--':>10s}{'--':>8s}{'--':>8s}")
                continue
            a_s = f"{r['mean_AUC']:10.4f}" if np.isfinite(r["mean_AUC"]) else f"{'n/a':>10s}"
            p_s = f"{r['mean_PCC']:10.4f}" if np.isfinite(r["mean_PCC"]) else f"{'n/a':>10s}"
            print(f"    {nm:22s}{a_s}{p_s}{int(r['n_AUC']):8d}{int(r['n_PCC']):8d}")

print(f"\n==== Cross-replicate fluctuations (max - min, taking max over all groups; {len(repeat_cols)} replicates = "
      f"{'original F02g plot + ' if INCLUDE_ORIGINAL_REPEAT else ''}{len(REPLICATES)} recalculated replicates) ====")
spread_pcc = df_all.pivot_table(index=row_keys, columns="Repeat", values="PCC", aggfunc="first")
for a in ALL_ALGOS:
    sub = spread[spread["Algorithm"] == a]
    if sub.empty:
        print(f"  {name_map.get(a, a):16s} -- (no records for this algorithm in this run)")
        continue
    pm = spread_pcc.xs(a, level="Algorithm")
    pspread = pm.max(axis=1) - pm.min(axis=1)
    tag = "   <- baseline, should be 0" if a in base_names else ""
    print(f"  {name_map.get(a, a):16s} AUC: mean = {sub['mean'].mean():.4f}  "
          f"max spread = {sub['spread'].max():.4f}   min spread = {sub['spread'].min():.4f}{tag}")
    print(f"  {'':16s} PCC: mean = {pm.mean(axis=1).mean():.4f}  "
          f"max spread = {pspread.max():.4f}   min spread = {pspread.min():.4f}")

# ============================================================
# DeepACE mean across repeats
# ============================================================

deepace_auc = df_all[df_all["Algorithm"].isin(DEEPACE_ALGOS)]
per_repeat = deepace_auc.pivot_table(index="Algorithm", columns="Repeat", values="AUC", aggfunc="mean")
per_repeat = per_repeat.reindex(index=DEEPACE_ALGOS, columns=repeat_cols)
mean_sd = pd.DataFrame({"Mean": per_repeat.mean(axis=1),
                        "SD": per_repeat.std(axis=1, ddof=1)}).join(per_repeat)
mean_sd.round(6).to_csv(f"{SUMMARY_DIR}/S13_deepace_mean_across_repeats.csv")

print(f"\n==== DeepACE mean AUC over all groups in each repeat, and the mean / SD of it "
      f"across the {len(repeat_cols)} repeats ====")
print(f"{'Algorithm':18s}{'Mean':>10s}{'SD':>10s}" + "".join([f"{lab:>18s}" for lab in repeat_cols]))
for algo in DEEPACE_ALGOS:
    row = mean_sd.loc[algo]
    print(f"{algo:18s}{row['Mean']:10.4f}{row['SD']:10.4f}"
          + "".join([f"{row[lab]:18.4f}" for lab in repeat_cols]))

print(f"\nOutput directory: {SUPP_ROOT}")
print(f"  - Plots ({'/'.join(SAVE_FORMATS)}, total {len(REPLICATES)*len(MOTIF_LIST)} images) and S13_barplot_all.csv in root directory")
print(f"  - Recalculated score tables for each repeat: {SCORES_DIR}/<repeat>/analysis_{{mahalanobis,cold,cosine}}/")
print(f"  - Summary tables: {SUMMARY_DIR}/  (S13_auc_table_*.csv has AUC and PCC per row; "
      f"S13_mean_by_motif.csv = average AUC/PCC per motif as above; "
      f"S13_auc_spread_across_repeats.csv = spread across {len(repeat_cols)} replicates; "
      f"S13_deepace_mean_across_repeats.csv = DeepACE mean AUC per repeat as above)")

