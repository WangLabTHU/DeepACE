'''
enformer/xxx模型组合迁移到cagi5

/home/hyu/Digital_Platform/manuals/figs6_virtual_screen_specific.py

cp /home/hyu/Digital_Platform/manuals/figs6_virtual_screen_melting/comb_model_validation_mprabase.csv /home/hyu/DeepACE/Supps/S20_robust_migrate_cagi5
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

from math import pi


def load_data_point_mutation(motif):
    uni_path = f"./Preds/D05_mprabase/point_MPRABase_{motif}_saturation/uni_pred.npy"
    tsv_path = f"./Preds/D05_mprabase/point_MPRABase_{motif}_saturation.tsv"
    rand_path = f"./Preds/D05_mprabase/random_sample_1/uni_pred.npy"
    uni_pred = np.load(uni_path)
    df = pd.read_csv(tsv_path, sep="\t")
    labels = df['VariantExpressionEffect (log2)'].to_numpy()
    
    alt_data = uni_pred[:-1]
    ref_data = uni_pred[-1:]
    rand_data = np.load(rand_path)
    valid_cols = np.isfinite(alt_data).all(axis=0) & np.isfinite(ref_data).all(axis=0) & np.isfinite(rand_data).all(axis=0)
    alt_data = alt_data[:, valid_cols]
    ref_data = ref_data[:, valid_cols]
    rand_data = rand_data[:, valid_cols]
    return alt_data, ref_data, rand_data, labels


def filter_data_by_models(data, selected_models, anno_df):
    model_indices = {m: anno_df[anno_df['model'] == m].index.tolist() for m in model_list}
    keep_indices = []
    for m in selected_models:
        keep_indices.extend(model_indices[m])
    keep_indices = sorted(keep_indices)
    return data[:, keep_indices] if keep_indices else data

def preprocess_data(alt_data, ref_data, rand_data):
    combined = np.vstack((alt_data, ref_data, rand_data))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined)
    n_alt = len(alt_data)
    n_ref = len(ref_data)
    scaled_alt = scaled[:n_alt]
    scaled_ref = scaled[n_alt:n_alt+n_ref]
    scaled_rand = scaled[n_alt+n_ref:]
    sample_data = np.vstack((scaled_ref, scaled_alt, scaled_rand))
    sample_labels = (['Ref'] * len(scaled_ref) +
                     ['Alt'] * len(scaled_alt) +
                     ['Rand'] * len(scaled_rand))
    return sample_data, sample_labels

def compute_fold_change(sample_data, sample_labels, sorted_labels, n_neighbors=500):
    real_mask = np.array(sample_labels) == 'Alt'
    rand_mask = np.array(sample_labels) == 'Rand'
    real_vectors = sample_data[real_mask]
    rand_vectors = sample_data[rand_mask]
    if len(real_vectors) == 0 or len(rand_vectors) == 0:
        return np.nan
    var = np.var(rand_vectors, axis=0)
    inv_std = 1.0 / np.sqrt(var + 1e-8)
    def diag_mahalanobis(x, y, inv_std=inv_std):
        diff = (x - y) * inv_std
        return np.sqrt(np.dot(diff, diff))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=diag_mahalanobis).fit(rand_vectors)
    distances, _ = nbrs.kneighbors(real_vectors)
    pseudo_similarity = 1 - np.mean(distances, axis=1)
    order = np.argsort(-pseudo_similarity)
    expr_sorted = sorted_labels[order]
    n = len(expr_sorted)
    group_size = n // 10             
    groups = [expr_sorted[i:i + group_size] for i in range(0, n, group_size)]
    means   = [np.mean(g)   for g in groups]
    medians = [np.median(g) for g in groups]
    fc = 2 ** (means[-1] - np.mean(means))
    return fc if len(medians) >= 2 else np.nan


''' Dataset Preparation '''

# print("\n" + "="*60)
# print("GENERATING COMB-MODEL CROSS-VALIDATION HEATMAP")
# print("="*60)
# model_list = [
#     "Malinois", "Basset", "DanQ", "MPRALegNet", "SahuCNN", "APARENT2",
#     "DeepDNAshape", "CLIPNET", "Puffin", "Enformer", "Basenji2",
#     "Expecto", "Sei", "SpliceAI", "Borzoi", "SegmentNT"
# ]

# motifs_full = ["TERT", "HBG1", "LDLR", "F9", "GP1BA", "IRF4", "PKLR", "ZFAND3", "IRF6", "SORT1",
#                "HBB", "UC88", "MYC_rs6983267", "RET", "TCF7L2"]
# anno_df = pd.read_csv(f"./total_features.csv")
# comb_csv = "./Figs/F04_interpret_robust/F04d_robust_heatmap/comb_model_validation.csv"
# fc_df_comb = pd.read_csv(comb_csv, index_col=0)
# fc_df_comb = fc_df_comb.astype(float)
# comb_names = fc_df_comb.columns.tolist()[:-1]

# n_scen = len(motifs_full)
# n_combs = len(comb_names)
# fc_matrix_comb = np.full((n_scen, n_combs + 1), np.nan)

# for comb_idx, comb_name in enumerate(comb_names):
#     print(f"\nEvaluating combination [{comb_idx+1}/{n_combs}]: {comb_name}")
#     selected_model_list = comb_name.split("_")
#     for scen_idx, motif in enumerate(motifs_full):
        
#         alt_data, ref_data, rand_data, labels = load_data_point_mutation(motif)
#         filt_alt = filter_data_by_models(alt_data, selected_model_list, anno_df)
#         filt_ref = filter_data_by_models(ref_data, selected_model_list, anno_df)
#         filt_rand = filter_data_by_models(rand_data, selected_model_list, anno_df)
#         combined = np.vstack((filt_alt, filt_ref, filt_rand))
#         uni_selected = PCA(n_components=50, random_state=42).fit_transform(combined) if combined.shape[1] >= 50 else combined
#         filt_alt = uni_selected[:len(filt_alt)]
#         filt_ref = uni_selected[len(filt_alt):len(filt_alt)+1]
#         filt_rand = uni_selected[len(filt_alt)+1:]
#         sample_data, sample_labels = preprocess_data(filt_alt, filt_ref, filt_rand)
#         fc = compute_fold_change(sample_data, sample_labels, labels)
#         fc_matrix_comb[scen_idx, comb_idx] = fc
#         print(f"    → Evaluating {comb_name} on {motif}: {fc}")
        
#         alt_data, ref_data, rand_data, labels = load_data_point_mutation(motif)
#         filt_alt = filter_data_by_models(alt_data, model_list, anno_df)
#         filt_ref = filter_data_by_models(ref_data, model_list, anno_df)
#         filt_rand = filter_data_by_models(rand_data, model_list, anno_df)
#         combined = np.vstack((filt_alt, filt_ref, filt_rand))
#         uni_selected = PCA(n_components=50, random_state=42).fit_transform(combined) if combined.shape[1] >= 50 else combined
#         filt_alt = uni_selected[:len(filt_alt)]
#         filt_ref = uni_selected[len(filt_alt):len(filt_alt)+1]
#         filt_rand = uni_selected[len(filt_alt)+1:]
#         sample_data, sample_labels = preprocess_data(filt_alt, filt_ref, filt_rand)
#         fc = compute_fold_change(sample_data, sample_labels, labels)
#         fc_matrix_comb[scen_idx, -1] = fc
#         print(f"    → Evaluating DeepACE on {motif}: {fc}")

# comb_cols = comb_names + ["DeepACE"]
# fc_df_comb = pd.DataFrame(fc_matrix_comb, index=motifs_full, columns=comb_cols)
# out_csv = "./Supps/S20_robust_migrate_cagi5/comb_model_validation_mprabase.csv"
# fc_df_comb.to_csv(out_csv, float_format='%.6f')
# print(f"\nComb-model cross-validation matrix saved: {out_csv}")


''' Visualization Heatmap '''

comb_csv = "./Supps/S20_robust_migrate_cagi5/comb_model_validation_mprabase.csv"
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
fig, ax = plt.subplots(figsize=(len(fc_df_comb.columns) * 1.5 + 2, 16))
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
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
cbar.set_label('FC / DeepACE', rotation=270, labelpad=15, fontsize=11)
cbar.set_ticks([0.5, 0.75, 1.0, 1.25, 1.5])
cbar.set_ticklabels(['0.5×', '0.75×', '1.0×', '1.25×', '1.5×'])
plt.tight_layout()
output_dir = "./Supps/S20_robust_migrate_cagi5/"
output_png_comb = os.path.join(output_dir, "robust_migrate_cagi5_heatmap.pdf")
plt.savefig(output_png_comb, dpi=400, bbox_inches='tight')
plt.close()
print(f"Comb-model heatmap saved: {output_png_comb}")

ratio_df_comb = pd.DataFrame(intensity_matrix, index=fc_df_comb.index, columns=fc_df_comb.columns)
ratio_csv_comb = os.path.join(output_dir, "comb_heatmap_normalized_mprabase.csv")
ratio_df_comb.to_csv(ratio_csv_comb, float_format='%.6f')
print(f"Comb-model ratio matrix saved: {ratio_csv_comb}")


''' Visualization Barplot '''

comb_csv = "./Supps/S20_robust_migrate_cagi5/comb_heatmap_normalized_mprabase.csv"
comb_df = pd.read_csv(comb_csv)
tmp_xlabels = comb_df["Unnamed: 0"].tolist()
tmp_xlabels = [item.replace('_', '\n') for item in tmp_xlabels]
comb_df["Unnamed: 0"] = tmp_xlabels
tmp_ylabels = ["Enformer", "Enformer_Basenji2_DeepDNAshape"]
plot_df = comb_df.copy()
plot_df = plot_df.set_index(plot_df.columns[0])
plot_df_long = plot_df.reset_index().melt(
    id_vars=plot_df.index.name,
    value_vars=tmp_ylabels,
    var_name='Model',
    value_name='Normalized Activity'
)
plot_df_long['Δ Normalized Activity'] = plot_df_long['Normalized Activity'] - 1
plt.figure(figsize=(16, 8))
sns.barplot(
    data=plot_df_long,
    x=plot_df_long.columns[0],      
    y='Δ Normalized Activity',             
    hue='Model',
    palette=['#74a892', '#d4a558'],     
    edgecolor='black',
    linewidth=1.2)
plt.xlabel('Variant', fontsize=12)
plt.ylabel('Δ Normalized Activity\n(relative to DeepACE)', fontsize=12) 
plt.xticks(fontsize=12)
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = []
for lab in labels:
    if "Enformer_Basenji2_DeepDNAshape" in lab:
        new_labels.append("Enformer_Basenji2\n_DeepDNAshape")
    else:
        new_labels.append(lab)
    
plt.legend(handles, new_labels, title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
output_dir = "./Supps/S20_robust_migrate_cagi5/"
save_path = os.path.join(output_dir, "robust_migrate_cagi5_barplot.pdf")
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.close()
print(f"[Saved] Delta barplot -> {save_path}")

