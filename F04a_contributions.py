'''
/home/hyu/Digital_Platform/manuals/figs4_virtual_screen_interpret.py

cp /home/hyu/Digital_Platform/manuals/figs4_virtual_screen_interpret/scatters/warm_epigenetics_pseudo_random_mahalanobis/regression* /home/hyu/DeepACE/Preds/D04_deeptfbu/interpret_epigenetics_pseudo_random_mahalanobis
cp /home/hyu/Digital_Platform/manuals/figs4_virtual_screen_interpret/scatters/warm_MPRA_pseudo_random_mahalanobis/regression* /home/hyu/DeepACE/Preds/D06_mpra/interpret_MPRA_pseudo_random_mahalanobis

cp /home/hyu/DeepACE/Preds/D04_deeptfbu/interpret_epigenetics_pseudo_random_mahalanobis/regression_feature_contributions_HNF4A.pdf /home/hyu/DeepACE/Figs/F04_interpret_robust/F04a_contributions.pdf
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

import joblib
import collections, itertools
from itertools import chain
from scipy import stats
from scipy.stats import ranksums
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def load_data_and_model(cell, motif=None):
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
    combined_data = np.vstack((primary_data, pseudo_data)) if len(pseudo_data) > 0 else primary_data
    anno_df = pd.read_csv(f"./total_features.csv")

    pca_model = None
    pca = PCA(n_components=50, random_state=42)
    uni_selected = pca.fit_transform(combined_data)
    pca_model = pca
    primary_data = uni_selected[:-len(pseudo_data)] if len(pseudo_data) > 0 else uni_selected
    pseudo_data = uni_selected[-len(pseudo_data):] if len(pseudo_data) > 0 else np.array([])
    return primary_data, pseudo_data, labels, pca_model

def classify_feature(f):
    if f.startswith("CHIP-seq:H3K"):
        return "Histone"
    elif f.startswith("CHIP-seq:"):
        return "Motif"
    elif "CAGE" in f:
        return "RNA"
    elif "ATAC-seq" in f:
        return "Accessibility"
    else:
        return "Other"

def load_data_raw(cell, motif=None):
    if dataset == "MPRA":
        primary_data = np.load(f"./Preds/D06_mpra/valids_MPRA_AdaLead_{cell}/uni_pred.npy")
        labels_df = pd.read_csv("./Datas/D06_mpra/valids.csv")
        labels_df = labels_df[labels_df["origin"] == "AdaLead"].nlargest(500, f"{cell}_prediction")
        labels = labels_df[f"{cell}_l2fc"].to_numpy()
        ori_preds = None
    elif dataset == "epigenetics":
        primary_data = np.load(f"./Preds/D04_deeptfbu/valids_Epigenetics_{motif}/uni_pred.npy")
        labels_df = pd.read_excel("./Datas/D04_deeptfbu/3TF_MPRA.xlsx")
        labels_df = labels_df[labels_df['sequence_name'].str.contains(motif, na=False)]
        labels = labels_df["measured enhancer activity"].to_numpy()
        labels = np.log2(labels)
        ori_preds = [float(item.split("_")[0]) for item in labels_df['sequence_name']]
    else:
        raise ValueError("Invalid dataset input!")

    df_focus = pd.read_csv(f"./total_features.csv")
    df_focus["original_index"] = df_focus.index.tolist()
    df_focus["feature_clean"] = df_focus["feature"].replace({"CHIP:": "CHIP-seq:","CEBPb": "CEBPB","CHIP-seq:3xFLAG-": "CHIP-seq:"}, regex=True)
    df_focus["feature_group"] = df_focus["feature_clean"].apply(classify_feature)
    df_focus["feature_channel"] = df_focus.apply(lambda row: f"({row['model']})-({row.name})-{row['feature_clean']}", axis=1)
    df_focus = df_focus.drop(columns=["Unnamed: 0"])
    
    return primary_data, labels, df_focus, ori_preds


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
    sample_labels = (['Negative'] * len(neg_data) + ['Positive'] * len(pos_data) + 
                     ['Mid'] * len(mid_data) + ['Pseudo'] * len(scaled_pseudo))
    return sample_data, sample_labels, sorted_labels

def plot_top20_signed_combined(df, csv_path, title_prefix):
    df['signed_contribution'] = pd.to_numeric(df['signed_contribution'], errors='coerce')
    df = df.dropna(subset=['signed_contribution'])

    pos = df.nlargest(10, 'signed_contribution')[['feature_channel', 'signed_contribution']].copy()
    pos['direction'] = 'Low Similarity'
    neg = df.nsmallest(10, 'signed_contribution')[['feature_channel', 'signed_contribution']].copy()
    neg['direction'] = 'High Similarity'
    combined = pd.concat([pos, neg])
    combined = combined.sort_values('signed_contribution', ascending=True)

    colors = combined['signed_contribution'].apply(lambda x: 'skyblue' if x < 0 else 'salmon')
    plt.figure(figsize=(12, 16))
    bars = plt.barh(combined['feature_channel'], combined['signed_contribution'], 
                    color=colors, edgecolor='black', height=0.8)
    plt.axvline(x=0, color='black', linewidth=1.2)

    plt.xlabel('Signed Contribution', fontsize=12)
    plt.ylabel('Feature Channel', fontsize=12)
    plt.title(f'{title_prefix} - Top 20 Features for High vs Low Similarity', fontsize=14, pad=20)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    legend_elements = [
        Patch(facecolor='skyblue', edgecolor='black', label='High Similarity (signed < 0)'),
        Patch(facecolor='salmon', edgecolor='black', label='Low Similarity (signed > 0)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    combined_png = csv_path.replace('.csv', '.pdf')
    plt.savefig(combined_png, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Combined high/low similarity plot -> {combined_png}")


def interpret_pseudo_similarity_new(sample_data, sample_labels, labels, 
                                    pca_model=None, n_bins=5,
                                    output_dir="./figs4_virtual_screen_interpret", plot_tag="default"):
    
    anno_df = pd.read_csv("./total_features.csv")
    n_neighbors = 500
    
    ## calculating similarity
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
    mean_distances = distances.mean(axis=1)
    pseudo_similarity = 1 - mean_distances / np.max(mean_distances)

    ## analyzing the contribution by regression model
    anno_indices = anno_df.index.tolist()
    pca_components = pca_model.components_  # (n_components, n_original)
    n_original = pca_components.shape[1]
    X_all = real_vectors[order]
    y_all = sorted_similarity
    try:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_all, y_all)
        pc_importances = rf.feature_importances_  # (n_components,)
    except Exception as e:
        raise RuntimeError(f"[Error] RandomForest fitting failed: {e}")
    abs_loadings = np.abs(pca_components)                  # (n_comp, n_orig)
    weighted_contrib = abs_loadings.T @ pc_importances     # (n_orig,)
    total = weighted_contrib.sum()
    if total > 0:
        weighted_contrib = weighted_contrib / total
        pc_direction = np.array([np.corrcoef(X_all[:, i], y_all)[0,1] for i in range(X_all.shape[1])])
        pc_direction = np.nan_to_num(pc_direction)
        pc_direction = -pc_direction
        signed_contrib = pca_components.T @ (pc_importances * pc_direction)
    else:
        weighted_contrib = np.zeros(n_original)
        signed_contrib = np.zeros(n_original)

    df_all = anno_df.copy()
    df_all["original_index"] = anno_indices
    df_all["abs_contribution"] = weighted_contrib
    df_all["signed_contribution"] = signed_contrib
    df_all["mapping_method"] = "RandomForest_weighted"
    df_all["feature_clean"] = df_all["feature"].replace({
        "CHIP:": "CHIP-seq:",
        "CEBPb": "CEBPB",
        "CHIP-seq:3xFLAG-": "CHIP-seq:"
    }, regex=True)
    df_all["feature_group"] = df_all["feature_clean"].apply(classify_feature)
    df_all["feature_channel"] = df_all.apply(lambda row: f"({row['model']})-({row['original_index']})-{row['feature_clean']}", axis=1)
    contrib_save = os.path.join(output_dir, f"regression_feature_contributions_{plot_tag}.csv")
    df_all = df_all.sort_values(by="signed_contribution", ascending=False)
    df_all = df_all.drop(columns=["Unnamed: 0"]).reset_index(drop=True)
    df_all.to_csv(contrib_save, index=True)
    print(f"[Info] Global regression feature contributions saved to {contrib_save}")
    
    ## barplot
    regression_csv = os.path.join(output_dir, f"regression_feature_contributions_{plot_tag}.csv")
    if os.path.exists(regression_csv):
        df_reg = pd.read_csv(regression_csv)
        plot_top20_signed_combined(df_reg, regression_csv, "Regression Model")
    else:
        print(f"[Warning] File not found: {regression_csv}")


pseudo_source = "random"
for dataset in ["MPRA", "epigenetics"]:
    if dataset == "MPRA":
        cells = ["HepG2", "K562", "SKNSH"]
    elif dataset == "epigenetics":
        cells = ["HepG2", "HepG2", "HepG2"]
        motifs = ["ELF1_1_aim", "HNF1A_1_aim", "HNF4A_1_aim"]
    else:
        raise ValueError("Invalid dataset input!")
    
    for i, cell in enumerate(cells):
    motif = motifs[i] if dataset == "epigenetics" else None
    plot_tag = motif.split("_")[0] if dataset == "epigenetics" else cell
    print(f"Start Processing: {dataset}, {plot_tag}, {mode}, mahalanobis, pseudo=random")
    
        if dataset == "MPRA":
            output_dir = "./Preds/D04_deeptfbu/interpret_epigenetics_pseudo_random_mahalanobis"
        elif dataset == "epigenetics":
            output_dir = "./Preds/D06_mpra/interpret_MPRA_pseudo_random_mahalanobis"

        for i, cell in enumerate(cells):
            motif = motifs[i] if dataset == "epigenetics" else None
            plot_tag = motif.split("_")[0] if dataset == "epigenetics" else cell
            print(f"Start Processing: {dataset}, {plot_tag}, {mode}, mahalanobis, pseudo=random")
            primary_data, pseudo_data, labels, pca_model = load_data_and_model(cell, motif)
            sample_data, sample_labels, sorted_labels = preprocess_data(primary_data, pseudo_data, labels)
            interpret_pseudo_similarity_new(sample_data, sample_labels, sorted_labels, pca_model=pca_model, output_dir=output_dir, plot_tag=plot_tag)
        