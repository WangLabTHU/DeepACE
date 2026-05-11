'''
/home/hyu/Digital_Platform/manuals/figs6_point_mutation_melting.py

/home/hyu/Digital_Platform/manuals/figs6_virtual_screen_melting/comb_heatmap_normalized.pdf

cp -r /home/hyu/Digital_Platform/manuals/figs6_virtual_screen_melting/epigenetics_* /home/hyu/DeepACE/Figs/F04_interpret_robust/F04d_robust_heatmap/
cp -r /home/hyu/Digital_Platform/manuals/figs6_virtual_screen_melting/MPRA_* /home/hyu/DeepACE/Figs/F04_interpret_robust/F04d_robust_heatmap/

cp /home/hyu/Digital_Platform/manuals/figs6_virtual_screen_melting/single_model_validation.csv /home/hyu/DeepACE/Figs/F04_interpret_robust/F04d_robust_heatmap/
cp /home/hyu/Digital_Platform/manuals/figs6_virtual_screen_melting/comb_model_validation.csv /home/hyu/DeepACE/Figs/F04_interpret_robust/F04d_robust_heatmap/
cp /home/hyu/Digital_Platform/manuals/figs6_virtual_screen_melting/single_model_validation.csv /home/hyu/DeepACE/Figs/F04_interpret_robust/F04d_robust_heatmap/
cp /home/hyu/Digital_Platform/manuals/figs6_virtual_screen_melting/comb_heatmap_normalized.pdf /home/hyu/DeepACE/Figs/F04_interpret_robust/F04d_robust_heatmap/
cp /home/hyu/Digital_Platform/manuals/figs6_virtual_screen_melting/comb_heatmap_normalized.csv /home/hyu/DeepACE/Figs/F04_interpret_robust/F04d_robust_heatmap/
'''

import random
import os, sys
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

random.seed(42)
np.random.seed(42)
from math import pi

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

def analyze_pseudo_similarity_simplified(sample_data, sample_labels, labels, plot_tag, n_neighbors=500, output_dir=None):
    
    pseudo_mask = np.array([g == 'Pseudo' for g in sample_labels])
    real_idx = np.where(~pseudo_mask)[0]
    pseudo_idx = np.where(pseudo_mask)[0]
    real_vectors = sample_data[real_idx]
    pseudo_vectors = sample_data[pseudo_idx]
    n_neighbors = min(n_neighbors, len(pseudo_idx))

    # Nearest neighbors analysis
    var = np.var(pseudo_vectors, axis=0)
    inv_std = 1.0 / np.sqrt(var + 1e-8)
    def diag_mahalanobis(x, y, inv_std=inv_std):
        diff = (x - y) * inv_std
        return np.sqrt(np.dot(diff, diff))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=diag_mahalanobis).fit(pseudo_vectors)
    distances, _ = nbrs.kneighbors(real_vectors)
    mean_distances = distances.mean(axis=1)
    pseudo_similarity = 1 - mean_distances / np.max(mean_distances)


    order = np.argsort(-pseudo_similarity)

    # 1. Compute cumulative sums and proportions
    neg_flags = np.array([sample_labels[i] == 'Negative' for i in real_idx])
    pos_flags = np.array([sample_labels[i] == 'Positive' for i in real_idx])
    cum_neg = np.cumsum(neg_flags[order])
    cum_pos = np.cumsum(pos_flags[order])
    prop_fin = (cum_neg + 1) / (cum_pos + 1)
    pd.DataFrame({
        "order": np.arange(1, len(order) + 1),
        "cum_neg": cum_neg,
        "cum_pos": cum_pos,
        "pseudo_similarity": pseudo_similarity[order],
        "prop_fin": prop_fin
    }).to_csv(f"{output_dir}/pseudo_effect_{plot_tag}.csv", index=False)


    # 2. Mean sorted_expr after removal
    cut_size = 100
    mean_remaining = []
    for k in range(1, len(order) + 1):
        remaining_idx = order[k:]
        if len(remaining_idx) > 0:
            mean_val = labels[remaining_idx].mean()
        else:
            mean_val = np.nan
        mean_remaining.append(mean_val)
    pd.DataFrame({
        "removed_top_n": np.arange(1, len(order) + 1),
        "mean_remaining": mean_remaining
    }).to_csv(f"{output_dir}/screen_effect_{plot_tag}.csv", index=False)


    # 3. Pseudo similarity vs sorted_expr
    group_size = 100
    sorted_expr = labels[order]
    pcc, _ = pearsonr(distances.mean(axis=1), sorted_expr)
    distances_sorted = distances.mean(axis=1)[order]
    groups = []
    group_labels = []
    for i in range(0, len(distances_sorted), group_size):
        end_idx = min(i + group_size, len(distances_sorted))
        group_expr = sorted_expr[i:end_idx]
        groups.append(group_expr)
        group_labels.append(f"{i+1}-{end_idx}")
    pd.DataFrame({
        'Expression': np.concatenate(groups),
        'Group': np.repeat(group_labels, [len(g) for g in groups])
    }).to_csv(f"{output_dir}/scatter_distance_expr_{plot_tag}.csv", index=False)


    # 4. Positive ratio among remaining samples
    cut_size = 100
    pos_ratio_remaining = []
    for k in range(1, len(order) + 1):
        remaining_idx = order[k:]
        if len(remaining_idx) > 0:
            pos_ratio = np.mean(pos_flags[remaining_idx])
        else:
            pos_ratio = np.nan
        pos_ratio_remaining.append(pos_ratio)
    pd.DataFrame({
        "removed_top_n": np.arange(1, len(order) + 1),
        "positive_ratio_remaining": pos_ratio_remaining
    }).to_csv(f"{output_dir}/positive_ratio_{plot_tag}.csv", index=False)


'''
basic settings
'''

print("\n" + "="*60)
print("GENERATING SINGLE-MODEL CROSS-VALIDATION HEATMAP")
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
output_dir = "./Figs/F04_interpret_robust/F04d_robust_heatmap"
os.makedirs(output_dir, exist_ok=True)
anno_df = pd.read_csv(f"./total_features.csv")
scenario_names = [f"{dataset}_{plot_tag}" for dataset, _, _, plot_tag in scenarios_list]
n_scen = len(scenario_names)
fc_matrix_single = np.full((n_scen, len(model_list) + 1), np.nan)


'''
combined models
'''
for scen_idx, (dataset, cell, motif, plot_tag) in enumerate(scenarios_list):
    scenario_name = f"{dataset}_{plot_tag}"
    primary_data, pseudo_data, labels = load_data(cell, motif)

    filt_primary = filter_data_by_models(primary_data, model_list, anno_df)
    filt_pseudo = filter_data_by_models(pseudo_data, model_list, anno_df)
    combined = np.vstack((filt_primary, filt_pseudo)) if len(filt_pseudo) > 0 else filt_primary

    uni_selected = PCA(n_components=50, random_state=42).fit_transform(combined)
    primary_data = uni_selected[:-len(filt_pseudo)] if len(filt_pseudo) > 0 else uni_selected
    pseudo_data = uni_selected[-len(filt_pseudo):] if len(filt_pseudo) > 0 else np.array([])
    sample_data, sample_labels, sorted_labels = preprocess_data(primary_data, pseudo_data, labels)
    fc = compute_fold_change(sample_data, sample_labels, sorted_labels)
    fc_matrix_single[scen_idx, -1] = fc
    print(f"    → {scenario_name}: DeepACE = {fc:.4f}")
      
    output_dir_pr = os.path.join(output_dir, f"{scenario_name}/deepace")
    os.makedirs(output_dir_pr, exist_ok=True)
    analyze_pseudo_similarity_simplified(sample_data, sample_labels, sorted_labels, f"deepace_{scenario_name}", output_dir=output_dir_pr)



'''
single model
'''
for model_idx, model in enumerate(model_list):
    print(f"  Evaluating single model: {model}")
    for scen_idx, (dataset, cell, motif, plot_tag) in enumerate(scenarios_list):
        scenario_name = f"{dataset}_{plot_tag}"
        primary_data, pseudo_data, labels = load_data(cell, motif)
        filt_primary = filter_data_by_models(primary_data, [model], anno_df)
        filt_pseudo = filter_data_by_models(pseudo_data, [model], anno_df)
        combined = np.vstack((filt_primary, filt_pseudo)) if len(filt_pseudo) > 0 else filt_primary

        if combined.shape[1] <= 1:
            continue
        elif combined.shape[1] < 50:
            uni_selected = combined
        else:
            uni_selected = PCA(n_components=50, random_state=42).fit_transform(combined)
        primary_data = uni_selected[:-len(filt_pseudo)] if len(filt_pseudo) > 0 else uni_selected
        pseudo_data = uni_selected[-len(filt_pseudo):] if len(filt_pseudo) > 0 else np.array([])
        sample_data, sample_labels, sorted_labels = preprocess_data(primary_data, pseudo_data, labels)
        fc = compute_fold_change(sample_data, sample_labels, sorted_labels)
        fc_matrix_single[scen_idx, model_idx] = fc
        print(f"    → {scenario_name}: {model} FC={fc:.4f}")
        
single_cols = model_list + ["DeepACE"]
fc_df_single = pd.DataFrame(fc_matrix_single, index=scenario_names, columns=single_cols)
single_csv = os.path.join(output_dir, "single_model_validation.csv")
fc_df_single.to_csv(single_csv, float_format='%.6f')
print(f"\nSingle-model cross-validation matrix saved: {single_csv}")


'''
combination: start from enformer, task=HNF4A
'''

print("\n" + "="*60)
print("GENERATING GREEDY-SEARCH TABLES")
print("="*60)
scenarios_list = [
    ("MPRA", "HepG2", None, "HepG2"),
    ("MPRA", "K562", None, "K562"),
    ("MPRA", "SKNSH", None, "SKNSH"),
    ("epigenetics", "HepG2", "ELF1_1_aim", "ELF1"),
    ("epigenetics", "HepG2", "HNF1A_1_aim", "HNF1A"),
    ("epigenetics", "HepG2", "HNF4A_1_aim", "HNF4A")
]
scenario_results = {}
for _, (dataset, cell, motif, plot_tag) in enumerate(scenarios_list):
    scenario_name = f"{dataset}_{plot_tag}"
    selected_models = []
    fold_change_history = []
    prev_fold_change = 0
    prev_selected_model = None
    primary_data, pseudo_data, labels = load_data(cell, motif)

    for round_cnt in range(0, 5+1):
        print(f"\n--- Round {round_cnt}: Single-model evaluation for {scenario_name} ---")
        fold_change_dict = {}
        for model in model_list:
            print(f"  Evaluating: {model}")
            if model in selected_models:
                continue
            selected_models_temp = selected_models + [model]
            primary_data_temp = filter_data_by_models(primary_data, selected_models_temp, anno_df)
            pseudo_data_temp = filter_data_by_models(pseudo_data, selected_models_temp, anno_df)
            combined_data = np.vstack((primary_data_temp, pseudo_data_temp)) if len(pseudo_data_temp) > 0 else primary_data_temp
            n_features = combined_data.shape[1]
            if n_features <= 1:
                fold_change_dict[model] = np.nan
                continue
            elif n_features < 50:
                uni_selected = combined_data
            else:
                uni_selected = PCA(n_components=50, random_state=42).fit_transform(combined_data)
            primary_data_temp = uni_selected[:-len(pseudo_data_temp)] if len(pseudo_data_temp) > 0 else uni_selected
            pseudo_data_temp = uni_selected[-len(pseudo_data_temp):] if len(pseudo_data_temp) > 0 else np.array([])
            sample_data, sample_labels, sorted_labels = preprocess_data(primary_data_temp, pseudo_data_temp, labels)
            fc = compute_fold_change(sample_data, sample_labels, sorted_labels)
            fold_change_dict[model] = fc
        valid_fcs = {m: fc for m, fc in fold_change_dict.items() if not np.isnan(fc)}
        if not valid_fcs:
            print("No valid improvement. Stopping.")
            break
        selected_model = max(valid_fcs, key=valid_fcs.get)
        current_fc = valid_fcs[selected_model]
        if current_fc <= prev_fold_change:
            print(f"No improvement ({current_fc:.4f} ≤ {prev_fold_change:.4f}). Stopping.")
            break
        ## saving analyzation records
        selected_models_temp = selected_models + [selected_model]
        primary_data_temp = filter_data_by_models(primary_data, selected_models_temp, anno_df)
        pseudo_data_temp = filter_data_by_models(pseudo_data, selected_models_temp, anno_df)
        combined_data = np.vstack((primary_data_temp, pseudo_data_temp)) if len(pseudo_data_temp) > 0 else primary_data_temp
        n_features = combined_data.shape[1]
        if n_features > 1:
            uni_selected = PCA(n_components=50, random_state=42).fit_transform(combined_data) if n_features >= 50 else combined_data
            primary_data_temp = uni_selected[:-len(pseudo_data_temp)] if len(pseudo_data_temp) > 0 else uni_selected
            pseudo_data_temp = uni_selected[-len(pseudo_data_temp):] if len(pseudo_data_temp) > 0 else np.array([])
            sample_data, sample_labels, sorted_labels = preprocess_data(primary_data_temp, pseudo_data_temp, labels)
            output_dir_pr = os.path.join(output_dir, f"{scenario_name}/round{round_cnt}")
            os.makedirs(output_dir_pr, exist_ok=True)
            analyze_pseudo_similarity_simplified(sample_data, sample_labels, sorted_labels, 
                                                 f"round{round_cnt}_{selected_model}", output_dir=output_dir_pr)
        ## reporting and update
        delta_selected = current_fc - prev_fold_change
        selected_models.append(selected_model)
        fold_change_history.append(current_fc)
        prev_fold_change = current_fc
        prev_selected_model = selected_model
        print(f"Round {round_cnt}: +{selected_model} | FC: {current_fc:.4f} | Δ: +{delta_selected:.4f}")
        round_cnt += 1
    df = pd.read_csv( os.path.join(output_dir, "single_model_validation.csv") )
    row = df[df.iloc[:, 0] == scenario_name]    
    scenario_results[scenario_name] = {
        "selected_models": selected_models,
        "fold_change_history": fold_change_history,
        "final_fold_change": fold_change_history[-1] if fold_change_history else np.nan,
        "deepace_fc": row["DeepACE"]}
summary_data = []
for name, res in scenario_results.items():
    summary_data.append({
        "scenario": name,
        "selected_models": " → ".join(res["selected_models"]),
        "rounds": len(res["fold_change_history"]),
        "deepace_fc": res["deepace_fc"],
        "final_fc": res["final_fold_change"],
        "history_fc": " | ".join([f"{fc:.3f}" for fc in res["fold_change_history"]])
    })
summary_data = pd.DataFrame(summary_data)
summary_data.to_csv( os.path.join(output_dir, f"greedy_search.csv") )


'''
combination: cross validations
'''

summary_data = pd.read_csv( os.path.join(output_dir, f"greedy_search.csv") )
optimal_combinations = {}
for _, row in summary_data.iterrows():
    scenario = row["scenario"]
    models_str = row["selected_models"]
    model_list_comb = [m.strip() for m in models_str.split("→")]
    optimal_combinations[scenario] = model_list_comb
uniq_combinations = []
uniq_combinations_ids = []
for scenario, models in optimal_combinations.items():
    name = "_".join(models)
    if name not in uniq_combinations_ids:
        uniq_combinations_ids.append(name)
        uniq_combinations.append(models)

scenarios_list = [
    ("MPRA", "HepG2", None, "HepG2"),
    ("MPRA", "K562", None, "K562"),
    ("MPRA", "SKNSH", None, "SKNSH"),
    ("epigenetics", "HepG2", "ELF1_1_aim", "ELF1"),
    ("epigenetics", "HepG2", "HNF1A_1_aim", "HNF1A"),
    ("epigenetics", "HepG2", "HNF4A_1_aim", "HNF4A")
]
anno_df = pd.read_csv(f"./total_features.csv")
scenario_names = [f"{dataset}_{plot_tag}" for dataset, _, _, plot_tag in scenarios_list]
n_scen = len(scenario_names)
n_combs = len(uniq_combinations)
fc_matrix_comb = np.full((n_scen, n_combs + 1), np.nan)

for comb_idx, comb_models in enumerate(uniq_combinations):
    comb_name = "_".join(comb_models)
    print(f"\nEvaluating combination [{comb_idx+1}/{n_combs}]: {comb_name}")
    
    for scen_idx, (dataset, cell, motif, plot_tag) in enumerate(scenarios_list):
        scenario_name = f"{dataset}_{plot_tag}"
        primary_data, pseudo_data, labels = load_data(cell, motif)
        primary_data = filter_data_by_models(primary_data, comb_models, anno_df)
        pseudo_data = filter_data_by_models(pseudo_data, comb_models, anno_df)
        combined_data = np.vstack((primary_data, pseudo_data)) if len(pseudo_data) > 0 else primary_data
                
        n_features = combined_data.shape[1]
        if n_features <= 1:
            print(" [skip: <2 feats]")
            continue
        elif n_features < 50:
            uni_selected = combined_data
        else:
            uni_selected = PCA(n_components=50, random_state=42).fit_transform(combined_data)
        
        primary_data = uni_selected[:-len(pseudo_data)] if len(pseudo_data) > 0 else uni_selected
        pseudo_data = uni_selected[-len(pseudo_data):] if len(pseudo_data) > 0 else np.array([])
        sample_data, sample_labels, sorted_labels = preprocess_data(primary_data, pseudo_data, labels)
        fc = compute_fold_change(sample_data, sample_labels, sorted_labels)
        
        fc_matrix_comb[scen_idx, comb_idx] = fc
        df = pd.read_csv( os.path.join(output_dir, "single_model_validation.csv") )
        row = df[df.iloc[:, 0] == scenario_name]  
        fc_matrix_comb[scen_idx, -1] = row["DeepACE"].values

columns = uniq_combinations_ids + ["DeepACE"]
fc_df_comb = pd.DataFrame(fc_matrix_comb, index=scenario_names, columns=columns)
output_csv = os.path.join(output_dir, "comb_model_validation.csv")
fc_df_comb.to_csv(output_csv)
print(f"\nFull cross-validation matrix saved: {output_csv}")










'''
saving figures
'''
comb_csv = os.path.join(output_dir, "comb_model_validation.csv")
fc_df_comb = pd.read_csv(comb_csv, index_col=0)
fc_df_comb = fc_df_comb.astype(float)
deepace_fc_comb = fc_df_comb["DeepACE"].values
other_cols_comb = [c for c in fc_df_comb.columns if c != "DeepACE"]
fc_values_comb = fc_df_comb[other_cols_comb].values

# normalized color strength：|fc| / |All| × sign(fc)
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
fig, ax = plt.subplots(figsize=(len(fc_df_comb.columns) * 1.5 + 4, 8))
vmin, vmax = 0.5, 1.5
norm = plt.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.get_cmap("RdBu_r")

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
cbar.set_label('FC / DeepACE', rotation=270, labelpad=15, fontsize=11)
cbar.set_ticks([0.5, 0.75, 1.0, 1.25, 1.5])
cbar.set_ticklabels(['0.5×', '0.75×', '1.0×', '1.25×', '1.5×'])
plt.tight_layout()
output_png_comb = os.path.join(output_dir, "comb_heatmap_normalized.pdf")
plt.savefig(output_png_comb, dpi=400, bbox_inches='tight')
plt.close()
print(f"Comb-model heatmap saved: {output_png_comb}")

# saving ratio csv
ratio_df_comb = pd.DataFrame(intensity_matrix, index=fc_df_comb.index, columns=fc_df_comb.columns)
ratio_csv_comb = os.path.join(output_dir, "comb_heatmap_normalized.csv")
ratio_df_comb.to_csv(ratio_csv_comb, float_format='%.6f')
print(f"Comb-model ratio matrix saved: {ratio_csv_comb}")

# Top 3 models
mean_ratio_comb = ratio_df_comb.iloc[:, :-1].mean(axis=0).sort_values(ascending=False)
print("\nTop 3 Best Combination Models (Avg FC / All_Models):")
for i, (model, ratio) in enumerate(mean_ratio_comb.head(3).items(), 1):
    print(f"  {i}. {model}: ×{ratio:.3f}")