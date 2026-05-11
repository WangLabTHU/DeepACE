'''
/home/hyu/Digital_Platform/manuals/figs4_virtual_screen_interpret.py

cp /home/hyu/Digital_Platform/manuals/figs4_virtual_screen_interpret/scatters/warm_epigenetics_pseudo_random_mahalanobis/boxplot_(Enformer)-(3939)-CHIP-seq:HNF4A.pdf /home/hyu/DeepACE/Preds/D06_mpra/interpret_MPRA_pseudo_random_mahalanobis/
cp /home/hyu/Digital_Platform/manuals/figs4_virtual_screen_interpret/scatters/warm_epigenetics_pseudo_random_mahalanobis/boxplot_(DeepTFBU)-(x)-MPRA.pdf /home/hyu/DeepACE/Preds/D06_mpra/interpret_MPRA_pseudo_random_mahalanobis/
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


def plot_boxplot(pred, label, save_path=None, n_samples=100):
    
    pred = np.asarray(pred).ravel()
    label = np.asarray(label).ravel()
    sorted_indices = np.argsort(pred)

    bottom_idx = sorted_indices[:n_samples]
    bottom_labels = label[bottom_idx]
    top_idx = sorted_indices[-n_samples:]
    top_labels = label[top_idx]
    data_to_plot = [bottom_labels, top_labels]
    positions = [1, 2]
    box_labels = ['Bottom 100\n(Lowest Predicted)', 'Top 100\n(Highest Predicted)']
    stat, p_value = ranksums(top_labels, bottom_labels, alternative='greater')

    plt.figure(figsize=(8, 8))
    bp = plt.boxplot(data_to_plot, positions=positions, labels=box_labels,
                     patch_artist=True, notch=True, widths=0.6)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('sandybrown')
    plt.scatter(positions, [np.mean(bottom_labels), np.mean(top_labels)],
                color='red', s=80, zorder=3, label='Mean')

    y_max = max(np.max(bottom_labels), np.max(top_labels))
    y_min = min(np.min(bottom_labels), np.min(top_labels))
    y_range = y_max - y_min
    y_pos = y_max + y_range * 0.1
    plt.plot([1, 1, 2, 2], [y_pos, y_pos + y_range*0.05, y_pos + y_range*0.05, y_pos], 
             color='black', lw=1.2)
    if p_value < 0.001:
        p_text = 'p < 0.001'
        stars = '***'
    elif p_value < 0.01:
        p_text = f'p = {p_value:.3f}'
        stars = '**'
    elif p_value < 0.05:
        p_text = f'p = {p_value:.3f}'
        stars = '*'
    else:
        p_text = f'p = {p_value:.3f}'
        stars = 'n.s.'
    plt.text(1.5, y_pos + y_range*0.07, f'{p_text}\n{stars}', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')


    plt.ylabel('Observed MPRA Activity (label)', fontsize=12)
    plt.xlabel('Predicted Activity Rank Groups', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.legend(loc='upper left')
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Boxplot (top/bottom {n_samples}) -> {os.path.abspath(save_path)}")
    

pseudo_source = "random"
dataset = "epigenetics"
cell = "HepG2"
motif = "HNF4A_1_aim"
motifs = ["ELF1_1_aim", "HNF1A_1_aim", "HNF4A_1_aim"]

plot_tag = motif.split("_")[0] if dataset == "epigenetics" else cell
print(f"Start Processing: {dataset}, {plot_tag}, {mode}, mahalanobis, pseudo=random")
output_dir = "./Figs/F04_interpret_robust"

plot_boxplot(ori_preds, labels, os.path.join(output_dir, f"F04b_boxplot_deeptfbu"))
idx = 3939
pred_list = primary_data[:, idx]
plot_boxplot(pred_list, labels, os.path.join(output_dir, f"F04b_boxplot_deepace.pdf"))