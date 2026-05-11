'''
leiden plot of GATA, CTCF, H3K27me3
/home/hyu/Digital_Platform/manuals/3_deciphering.py

hg19_path = "/home/hyu/2_Basset/hg19/hg19.fa"
hg38_path = "/home/hyu/Digital_Platform_Dataset/CREME/GRCh38.primary_assembly.genome.fa"
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

import leidenalg
import networkx as nx
import igraph as ig
from umap import UMAP
import scipy.cluster.hierarchy as sch
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

BASE_DIR = os.path.abspath("./")
sys.path.append(BASE_DIR)
from functions import get_matched, open_fa

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def leiden_cluster_features(X, threshold=0.8, k=50, resolution=1.0,
                                     top_n=None, save_dir="./Preds/D01_screens/",
                                     prefix="leiden_adaptive"):
    """
    Adaptive Leiden clustering with KNN graph.
    Recursively split clusters until mean intra-cluster similarity >= threshold.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        Input data, each column is a feature vector.
    threshold : float
        Minimum required intra-cluster similarity.
    k : int
        Number of neighbors for KNN graph.
    resolution : float
        Resolution parameter for Leiden (controls cluster granularity).
    top_n : int or None
        Number of top clusters to return.
    save_dir : str
        Directory to save output.
    prefix : str
        Prefix for saved files.

    Returns
    -------
    top_clusters : list of (cluster_id, mean_corr, feature_indices)
    """
    print("[INFO] Standardizing input features...")
    X = StandardScaler().fit_transform(X)
    X_T = X.T  # shape = (n_features, n_samples)

    # --- Step 1: Build KNN graph ---
    print(f"[INFO] Building KNN graph with k={k}...")
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X_T)
    distances, indices = nn.kneighbors(X_T)

    rows, cols, data = [], [], []
    for i in range(X_T.shape[0]):
        for j, d in zip(indices[i], distances[i]):
            rows.append(i)
            cols.append(j)
            data.append(float(1 - d))  # cosine similarity

    W = csr_matrix((data, (rows, cols)), shape=(X_T.shape[0], X_T.shape[0]))
    print(f"[INFO] KNN graph built: shape={W.shape}, nnz={W.nnz}")

    # --- Step 2: Convert sparse matrix to igraph ---
    print("[INFO] Converting KNN graph to igraph format...")
    g = ig.Graph()
    g.add_vertices(X_T.shape[0])
    g.add_edges(list(zip(rows, cols)))
    g.es['weight'] = [float(x) for x in data]

    # --- Step 3: Recursive Leiden clustering ---
    cluster_labels = -np.ones(X_T.shape[0], dtype=int)
    next_label = 0
    pbar = tqdm(total=X_T.shape[0], desc="Clusters assigned")

    def recursive_leiden(idx):
        nonlocal next_label
        if len(idx) == 1:
            cluster_labels[idx[0]] = next_label
            next_label += 1
            pbar.update(1)
            return

        subgraph = g.subgraph(idx)
        partition = leidenalg.find_partition(
            subgraph,
            leidenalg.RBConfigurationVertexPartition,
            weights=subgraph.es['weight'],
            resolution_parameter=resolution,
            n_iterations=-1
        )
        labels = np.array(partition.membership)

        for l in np.unique(labels):
            sub_idx = np.array(idx)[labels == l]
            if len(sub_idx) == 1:
                cluster_labels[sub_idx[0]] = next_label
                next_label += 1
                pbar.update(1)
            else:
                sub_sim = cosine_similarity(X_T[sub_idx])
                mean_corr = (np.sum(sub_sim) - len(sub_idx)) / (len(sub_idx)*(len(sub_idx)-1))
                if mean_corr >= threshold:
                    cluster_labels[sub_idx] = next_label
                    next_label += 1
                    pbar.update(len(sub_idx))
                else:
                    recursive_leiden(sub_idx)

    print("[INFO] Running adaptive Leiden clustering...")
    recursive_leiden(np.arange(X_T.shape[0]))
    pbar.close()

    # --- Step 4: Evaluate clusters ---
    cluster_scores = []
    for cid in np.unique(cluster_labels):
        idx = np.where(cluster_labels == cid)[0]
        if len(idx) > 1:
            sub_sim = cosine_similarity(X_T[idx])
            mean_corr = (np.sum(sub_sim) - len(idx)) / (len(idx)*(len(idx)-1))
        else:
            mean_corr = 1.0
        cluster_scores.append((cid, mean_corr, idx))

    if top_n is None:
        top_clusters = sorted(cluster_scores, key=lambda x: x[1], reverse=True)
    else:
        top_clusters = sorted(cluster_scores, key=lambda x: x[1], reverse=True)[:top_n]

    # --- Step 5: Save results ---
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{prefix}_labels.npy"), cluster_labels)
    df = pd.DataFrame(
        [(cid, score, len(idx), idx.tolist()) for cid, score, idx in top_clusters],
        columns=["cluster_id", "mean_corr", "size", "feature_indices"]
    )
    df.to_csv(os.path.join(save_dir, f"{prefix}_top_clusters.csv"), index=False)
    print(f"[INFO] Leiden clusters saved to {prefix}_top_clusters.csv")

    return top_clusters

def refine_clusters_fast(X_T, labels, min_size=5, sim_threshold=0.8):
    """
    Graph-based refinement of Leiden clusters (batch merging) with robustness fixes.
    Returns: refined labels (numpy array)
    """

    labels = labels.copy()
    # initial cluster -> indices mapping
    cluster_indices = {cid: np.where(labels == cid)[0] for cid in np.unique(labels)}
    centroids = {cid: X_T[idx].mean(axis=0) for cid, idx in cluster_indices.items()}
    sizes = {cid: len(idx) for cid, idx in cluster_indices.items()}

    # === Step 1: Merge small clusters among themselves (connected components) ===
    weak_clusters = [cid for cid, sz in sizes.items() if sz < min_size]
    if len(weak_clusters) > 1:
        G = nx.Graph()
        G.add_nodes_from(weak_clusters)

        # build edges between weak clusters if centroid similarity >= threshold
        for i, ci in enumerate(tqdm(weak_clusters, desc="Small cluster pairwise (step1)")):
            for cj in weak_clusters[i + 1:]:
                sim = cosine_similarity(centroids[ci][None, :], centroids[cj][None, :])[0, 0]
                if sim >= sim_threshold:
                    G.add_edge(ci, cj, weight=float(sim))

        # merge each connected component in batch (validate with full similarity)
        for component in tqdm(list(nx.connected_components(G)), desc="Merging connected components"):
            if len(component) < 2:
                continue
            all_idx = np.concatenate([cluster_indices[cid] for cid in component])
            n = len(all_idx)
            if n < 2:
                continue
            sim_mat = cosine_similarity(X_T[all_idx])  # shape (n, n)
            merged_sim = (np.sum(sim_mat) - n) / (n * (n - 1))
            if merged_sim >= sim_threshold:
                new_cid = min(component)
                # assign labels and update maps
                cluster_indices[new_cid] = all_idx
                labels[all_idx] = new_cid
                total_size = sum(sizes[cid] for cid in component)
                new_centroid = sum(sizes[cid] * centroids[cid] for cid in component) / total_size
                centroids[new_cid] = new_centroid
                sizes[new_cid] = total_size
                for cid in component:
                    if cid != new_cid:
                        # safe delete of old entries
                        cluster_indices.pop(cid, None)
                        centroids.pop(cid, None)
                        sizes.pop(cid, None)

        # remap IDs to a compact 0..K-1 to avoid stale ids
        unique_ids = sorted(cluster_indices.keys())
        id_map = {old: new_id for new_id, old in enumerate(unique_ids)}
        # remap labels
        # it's safe because we updated labels[all_idx] = new_cid above for merged nodes
        labels = np.array([id_map[l] for l in labels], dtype=int)
        # remap dictionaries
        cluster_indices = {id_map[old]: cluster_indices[old] for old in unique_ids}
        centroids = {id_map[old]: centroids[old] for old in unique_ids}
        sizes = {id_map[old]: sizes[old] for old in unique_ids}

    # === Step 2: Merge remaining small clusters into large clusters (batch) ===
    small_clusters = [cid for cid, sz in sizes.items() if sz < min_size]
    large_clusters = [cid for cid, sz in sizes.items() if sz >= min_size]

    if small_clusters and large_clusters:
        # build mapping: each small cluster chooses the best large cluster (if any)
        # we'll store edges (scid, lcid, sim)
        edges = []
        for scid in tqdm(small_clusters, desc="Small -> Large cluster mapping"):
            best_target = None
            best_sim = -1.0
            for lcid in large_clusters:
                # small speed: if centroid vectors exist
                sim = float(cosine_similarity(centroids[scid][None, :], centroids[lcid][None, :])[0, 0])
                if sim >= sim_threshold and sim > best_sim:
                    best_target, best_sim = lcid, sim
            if best_target is not None:
                edges.append((scid, best_target, best_sim))

        # process candidate merges; convert to graph where small -> large edges exist
        G2 = nx.Graph()
        for scid, lcid, w in edges:
            G2.add_edge(scid, lcid, weight=w)

        # iterate edges; use a stable order; ensure nodes still exist in cluster_indices
        for scid, lcid, data in tqdm([(u, v, d) for u, v, d in G2.edges(data=True)], desc="Merging small into large clusters"):
            # determine correct orientation: ensure scid is small and lcid is large
            u, v = scid, lcid  # from G2.edges we often get (u,v) = (small, large) but be safe
            # detect actual roles
            if u not in cluster_indices or v not in cluster_indices:
                continue
            # if both are small or both large, decide using sizes
            if sizes.get(u, 0) >= min_size and sizes.get(v, 0) < min_size:
                # swap so 'sc' is the small one
                scid_, lcid_ = v, u
            else:
                scid_, lcid_ = u, v

            # still ensure scid_ is small and lcid_ is large
            if scid_ not in cluster_indices or lcid_ not in cluster_indices:
                continue
            if sizes.get(scid_, 0) >= min_size:
                continue  # skip, not a small cluster anymore

            merged_idx = np.concatenate([cluster_indices[lcid_], cluster_indices[scid_]])
            n = len(merged_idx)
            if n < 2:
                continue
            sim_mat = cosine_similarity(X_T[merged_idx])
            merged_sim = (np.sum(sim_mat) - n) / (n * (n - 1))
            if merged_sim >= sim_threshold:
                # perform merge: assign small into large
                labels[cluster_indices[scid_]] = lcid_
                cluster_indices[lcid_] = merged_idx
                total_size = sizes[lcid_] + sizes[scid_]
                new_centroid = (sizes[lcid_] * centroids[lcid_] + sizes[scid_] * centroids[scid_]) / total_size
                centroids[lcid_] = new_centroid
                sizes[lcid_] = total_size
                # remove small cluster
                cluster_indices.pop(scid_, None)
                centroids.pop(scid_, None)
                sizes.pop(scid_, None)

        # final remap to compact ids and keep labels consistent
        unique_ids = sorted(cluster_indices.keys())
        id_map = {old: new_id for new_id, old in enumerate(unique_ids)}
        labels = np.array([id_map[l] for l in labels], dtype=int)
        # (optionally) remap cluster_indices/centroids/sizes as well
        cluster_indices = {id_map[old]: cluster_indices[old] for old in unique_ids}
        centroids = {id_map[old]: centroids[old] for old in unique_ids}
        sizes = {id_map[old]: sizes[old] for old in unique_ids}

    return labels

def plot_umap_feat(umap_emb, labels, title, filename):
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    base_cmap = plt.cm.get_cmap('tab20')
    colors_base = base_cmap(np.linspace(0, 1, 20))
    if n_labels <= 20:
        label_colors = {label: colors_base[i] for i, label in enumerate(unique_labels)}
    else:
        hsv_colors = plt.cm.hsv(np.linspace(0,1,n_labels))
        label_colors = {label: hsv_colors[i] for i, label in enumerate(unique_labels)}
    point_colors = np.array([label_colors[label] for label in labels])
    plt.figure(figsize=(8,6))
    plt.scatter(umap_emb[:,0], umap_emb[:,1], c=point_colors, s=5)
    handles = [mlines.Line2D([0],[0], marker='o', color='w',
                             markerfacecolor=label_colors[label], markersize=5)
               for label in unique_labels]
    plt.legend(handles, unique_labels, bbox_to_anchor=(1.05,1), loc='upper left', title=title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(filename, dpi=400)
    plt.close()

def plot_cluster_feature_summary(filt_pred, df_merged, random_state=42, prefix="run0_CTCF"):
    n_samples, n_features = filt_pred.shape
    cluster_representatives = df_merged.groupby('cluster_id', as_index=False).first().set_index('cluster_id')
    cluster_representatives = cluster_representatives.apply(
        lambda row: f"{row['index']}_{row['model']}_{row.name}", axis=1
    ).to_dict()
    feature_clusters = df_merged['cluster_id'].values
    feature_models = df_merged['model'].values
    feat_vectors = filt_pred.T  # shape = (n_features, n_samples)
    umap_emb = UMAP(n_components=2, random_state=random_state).fit_transform(feat_vectors)
    print("Generation of Feature embeddings finished!")
    
    feat_model_names = np.array(feature_models)
    plot_umap_feat(umap_emb, feat_model_names, title="Model", 
                   filename=f"./Supps/S01_leiden_details/umap_feat_model_{prefix}.pdf")
    print("Figures of embeddings across models finished!")



''' Dataset Preparation '''

# df = pd.read_csv("./Datas/D01_screens/GRCh38-cCREs.bed", sep="\t", header=None)
# df = df.sample(n=10000, random_state=19)
# genome_path = "./Datas/D02_grch/GRCh38.primary_assembly.genome.fa"
# genome = Fasta(genome_path)

# df_samples = []
# for _, row in df.iterrows():
#     chrom = row.iloc[0]
#     start = int(row.iloc[1])
#     end = int(row.iloc[2])
#     source = row.iloc[-1] 
#     length = end - start
#     if length < 600:
#         center = (start + end) // 2
#         half = 300
#         start = max(0, center - half)
#         end = start + 600
#     else:
#         center = (start + end) // 2
#         start = center - 300
#         end = center + 300
#     seq = genome[chrom][start:end].seq.upper()
#     df_samples.append((chrom, start, end, source, seq))

## saving for 5 iterations
# df_res = pd.DataFrame(df_samples, columns=["chrom", "start", "end", "source", "sequence"])
# df_res.to_csv("./Preds/D01_screens/CRE_samples_1.tsv", sep="\t", index=False)
# seqs = df_res["sequence"].tolist()
# output_file = f"./Preds/D01_screens/CRE_samples_1.fasta"
# with open(output_file, "w") as f:
#     for i, seq in enumerate(seqs):
#         f.write(f">sequence_{i}\n{seq}\n")



''' Leiden clustering '''

# cluster_labels = np.load("./Preds/D01_screens/leiden_run0_labels.npy")
# X_std = StandardScaler().fit_transform(X)
# X_T = X_std.T
# refined_labels = refine_clusters_fast(X_T, cluster_labels, min_size=5, sim_threshold=0.8)
# np.save(os.path.join("./Preds/D01_screens/", f"leiden_{run_id}_labels_refined.npy"), refined_labels)

# refined_labels = np.load("./Preds/D01_screens/leiden_run0_labels_refined.npy")
# refined_cluster_scores = []
# for cid in np.unique(refined_labels):
#     idx = np.where(refined_labels == cid)[0]
#     if len(idx) > 1:
#         sub_sim = cosine_similarity(X_T[idx])
#         mean_corr = (np.sum(sub_sim) - len(idx)) / (len(idx)*(len(idx)-1))
#     else:
#         mean_corr = 1.0
#     refined_cluster_scores.append((cid, mean_corr, len(idx), idx.tolist()))

# df_refined = pd.DataFrame(refined_cluster_scores, columns=["cluster_id", "mean_corr", "size", "feature_indices"])
# df_refined.to_csv(os.path.join("./Preds/D01_screens/", f"leiden_{run_id}_refined_clusters.csv"), index=False)
# print(f"[INFO] Refined clusters saved to leiden_{run_id}_refined_clusters.csv")

# run_id = "run0"
# df_cluster = pd.read_csv(f"./Preds/D01_screens/leiden_{run_id}_refined_clusters.csv")
# df_anno = pd.read_csv("./total_features.csv")
# df_total = []
# for idx in range(len(df_cluster)):
#     cluster_id = df_cluster.loc[idx, "cluster_id"]
#     mean_corr = df_cluster.loc[idx, "mean_corr"]
#     feature_indices = df_cluster.loc[idx, "feature_indices"]
#     feature_indices = [int(item) for item in feature_indices.strip('[]').split(',')]
#     feature_indices = np.array(feature_indices)
#     tmp_df_anno = df_anno.loc[feature_indices].copy()
#     tmp_df_anno["cluster_id"] = cluster_id
#     tmp_df_anno["mean_corr"] = mean_corr
#     tmp_df_anno = tmp_df_anno.drop('Unnamed: 0', axis=1).reset_index()
#     new_columns = ['cluster_id', 'mean_corr', 'index', 'model', 'celltype', 'feature', 'source']
#     tmp_df_anno = tmp_df_anno[new_columns]
#     print(tmp_df_anno)
#     df_total.append(tmp_df_anno)
# df_total = pd.concat(df_total, ignore_index=True)
# df_total.to_csv(f"./Preds/D01_screens/total_{run_id}_refined_clusters.csv")


''' Semantic Figures '''

run_id = "run0"
match_tag = "H3K27me3" # total, GATA, CTCF, H3K27me3
X = np.load(f"./Preds/D01_screens/CRE_samples_{run_id[-1]}/uni_pred.npy")
df_anno = pd.read_csv(f"./Preds/D01_screens/uni_anno.csv")
if match_tag != "total":
    uni_selected, uni_df = get_matched(X, df_anno, [f'{match_tag}'], top_cols=None, match_mode="soft")
else:
    uni_selected = X
    uni_df = df_anno
filt_pred = uni_selected
scaler = StandardScaler()
filt_pred = scaler.fit_transform(filt_pred)
df_cluster = pd.read_csv(f"./Preds/D01_screens/total_{run_id}_refined_clusters.csv")
df_merged = pd.merge(uni_df, 
                     df_cluster.drop(columns=[col for col in df_cluster.columns if col in uni_df.columns and col != "index"]), 
                     on = ["index"])
df_merged = df_merged.drop('Unnamed: 0', axis=1)
plot_cluster_feature_summary(filt_pred, df_merged, prefix=f"{run_id}_{match_tag}_leiden")




''' Interpreting '''

run_id = "run0"
match_tag = "H3K27me3" # GATA, CTCF, H3K27me3
X = np.load(f"./Preds/D01_screens/CRE_samples_{run_id[-1]}/uni_pred.npy")
df_anno = pd.read_csv(f"./Preds/D01_screens/uni_anno.csv")
uni_selected, uni_df = get_matched(X, df_anno, [f'{match_tag}'], top_cols=None, match_mode="soft")

filt_pred = uni_selected
scaler = StandardScaler()
filt_pred = scaler.fit_transform(filt_pred)
df_cluster = pd.read_csv(f"./Preds/D01_screens/total_{run_id}_refined_clusters.csv")
df_merged = pd.merge(uni_df, 
                     df_cluster.drop(columns=[col for col in df_cluster.columns if col in uni_df.columns and col != "index"]), 
                     on = ["index"])
df_merged = df_merged.drop('Unnamed: 0', axis=1)


cluster_sizes = df_merged["cluster_id"].value_counts()
cluster_sizes = cluster_sizes[cluster_sizes >= 5]
cluster_sizes = cluster_sizes.sort_values(ascending=False)
cluster_info = df_merged.groupby("cluster_id")
purity_dict = {}
model_dict = {}
rep_dict = {}
for cid, group in cluster_info:
    model_counts = group["model"].value_counts()
    purity = model_counts.max() / model_counts.sum()
    purity_dict[cid] = round(purity, 2)
    # representative model
    top_model = model_counts.idxmax()
    model_dict[cid] = top_model
    # representative channel
    rep_index = group[group["model"] == top_model].iloc[0]["index"]
    rep_dict[cid] = f"{rep_index}_{top_model}_{cid}"
labels = [rep_dict[cid] for cid in cluster_sizes.index]
models = [model_dict[cid] for cid in cluster_sizes.index]
purity_vals = [purity_dict[cid] for cid in cluster_sizes.index]
plot_df = pd.DataFrame({
    "Cluster": labels,
    "Features": cluster_sizes.values,
    "Model": models,
    "Purity": purity_vals
})
plt.figure(figsize=(14, 6))
ax = sns.barplot(x="Cluster", y="Features", hue="Model", data=plot_df, dodge=False)

for i, row in plot_df.iterrows():
    if row["Purity"] < 1.0:
        ax.text(i, row["Features"], f"{row['Purity']:.2f}", ha="center", va="bottom", fontsize=8, color="black")
        ax.vlines( x=i, ymin=0, ymax=row["Features"], colors="black", linestyles="dashed", linewidth=0.8 )
plt.xticks(rotation=90, fontsize=8)
plt.xlabel("Cluster Representative (index_model_clusterid)")
plt.ylabel("Number of Features")
plt.title("Distribution of Features per Cluster (≥5 features, sorted, purity annotated)")
plt.tight_layout()
plt.savefig(f"./Supps/S01_leiden_details/interpret_cluster_hist_{run_id}_{match_tag}.pdf")
plt.close()


model_stats = (
    df_merged.groupby("model")
    .agg(total_features=("index", "count"),
         unique_clusters=("cluster_id", pd.Series.nunique))
    .reset_index()
)
model_stats["cluster_numbers"] = model_stats["unique_clusters"]
model_stats["diversity"] = model_stats["unique_clusters"] / model_stats["total_features"]
model_stats = model_stats.sort_values("total_features", ascending=False)
fig, ax1 = plt.subplots(figsize=(10, 6))
color = "steelblue"
ax1.bar(model_stats["model"], model_stats["total_features"], color=color, alpha=1, edgecolor='black')
ax1.set_xlabel("Model")
ax1.set_ylabel("Total Features", color=color)
ax1.tick_params(axis="y", labelcolor=color)
ax1.tick_params(axis="x", rotation=45, labelsize=10)
ax2 = ax1.twinx()
color = "indianred"
ax2.plot(model_stats["model"], model_stats["cluster_numbers"], color=color, marker="o", linewidth=2)
ax2.set_ylabel("Cluster Numbers", color=color)
ax2.tick_params(axis="y", labelcolor=color)
for i in range(len(model_stats)):
    feature_per_cluster = model_stats["total_features"].iloc[i] / model_stats["cluster_numbers"].iloc[i]
    ax2.text(
        i,
        model_stats["cluster_numbers"].iloc[i],       
        f"({model_stats['cluster_numbers'].iloc[i]}, {feature_per_cluster:.1f})",  
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
        fontweight="bold"
    )
plt.title("Model Feature Count vs Cluster Numbers and Diversity")
plt.tight_layout()
plt.savefig(f"./Supps/S01_leiden_details/interpret_cluster_diversity_{run_id}_{match_tag}.pdf")
plt.close()



cluster_to_features = df_merged.groupby("cluster_id").indices
centroids = []
cluster_ids = []
models = []
for cid, idxs in cluster_to_features.items():
    feats = filt_pred[:, idxs] 
    centroid = feats.mean(axis=1) 
    centroids.append(centroid)
    cluster_ids.append(cid)
    models.append(df_merged.loc[idxs[0], "model"])
centroids = np.vstack(centroids)  # (n_clusters, n_samples)
sim_matrix = cosine_similarity(centroids)  # (n_clusters, n_clusters)
order_df = pd.DataFrame({"cluster_id": cluster_ids, "model": models})
order_df = order_df.sort_values(["model", "cluster_id"]).reset_index(drop=True)
ordered_idx = order_df.index
sim_matrix_ordered = sim_matrix[ordered_idx][:, ordered_idx]
plt.figure(figsize=(12, 10))
ax = sns.heatmap(sim_matrix_ordered, cmap="RdBu_r", center=0, vmin=-1, vmax=1, 
                 xticklabels=False, yticklabels=False)
ax.invert_yaxis()
tick_positions = []
tick_labels = []
start = 0
for model, group in order_df.groupby("model"):
    end = start + len(group)
    tick_positions.append((start + end - 1) / 2) 
    tick_labels.append(model)
    plt.axhline(start, color='black', linestyle='--', linewidth=1)
    plt.axvline(start, color='black', linestyle='--', linewidth=1)
    start = end
plt.xticks(tick_positions, tick_labels, rotation=90)
plt.yticks(tick_positions, tick_labels)
plt.title("Cluster Centroid Similarity Heatmap (Grouped by Model)")
plt.tight_layout()
plt.savefig(f"./Supps/S01_leiden_details/interpret_cluster_heatmap_{run_id}_{match_tag}.pdf")
plt.close()