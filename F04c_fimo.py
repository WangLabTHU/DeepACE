'''
/home/hyu/Digital_Platform/manuals/figs4_virtual_screen_interpret_4.py

cp /home/hyu/Digital_Platform/manuals/figs4_virtual_screen_interpret/debug/HNF4A/fimo.tsv /home/hyu/DeepACE/Preds/D04_deeptfbu/HNF4A_fimo.tsv

cp /home/hyu/Digital_Platform/manuals/figs4_virtual_screen_interpret/debug/HNF4A_measured enhancer activity.pdf 
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



def regroup(count):
    if count <= 8:
        return "<=8"
    elif count >= 14:
        return ">=14"
    else:
        return str(count)

df_focus = pd.read_csv(f"./total_features.csv")
df_focus["original_index"] = df_focus.index.tolist()
df_focus["feature_clean"] = df_focus["feature"].replace({"CHIP:": "CHIP-seq:","CEBPb": "CEBPB","CHIP-seq:3xFLAG-": "CHIP-seq:"}, regex=True)
df_focus["feature_group"] = df_focus["feature_clean"].apply(classify_feature)
df_focus["feature_channel"] = df_focus.apply(lambda row: f"({row['model']})-({row.name})-{row['feature_clean']}", axis=1)
df_focus = df_focus.drop(columns=["Unnamed: 0"])

labels_df = pd.read_excel("/Datas/D04_deeptfbu/3TF_MPRA.xlsx")
labels_df = labels_df[labels_df['sequence_name'].str.contains("HNF4A_1_aim", na=False)]
primary_data = np.load(f"./Preds/D04_deeptfbu/valids_Epigenetics_HNF4A_1_aim/uni_pred.npy")
labels_df["preds"] = [float(item.split("_")[0]) for item in labels_df["sequence_name"]]

for tag in ["preds", "measured enhancer activity"]:
    if tag == "preds":
        plot_tag = "deeptfbu"
    else:
        plot_tag = "mpra"

    df_fimo = pd.read_csv(f"./Preds/D04_deeptfbu/HNF4A_fimo.tsv", sep="\t")[:-3]
    hnf4a_counts = (
        df_fimo[df_fimo["motif_alt_id"].str.contains("HNF4A")] # MA0114.5, MA1494.2
        .groupby("sequence_name")
        .size()
        .reset_index(name="HNF4A_count")
    )
    expr_map = labels_df.set_index("sequence_name")[tag]
    hnf4a_df = (
        labels_df[["sequence_name"]]
        .merge(hnf4a_counts, on="sequence_name", how="left")
        .fillna(0)
        .merge(labels_df[["sequence_name", tag]], on="sequence_name")
    )
    hnf4a_df["HNF4A_count"] = hnf4a_df["HNF4A_count"].astype(int)
    hnf4a_df["HNF4A_group"] = hnf4a_df["HNF4A_count"].apply(regroup)
    base_order = ["<=8"] + [str(i) for i in range(9, 14)] + [">=14"]
    freq_order = [x for x in base_order if x in hnf4a_df["HNF4A_group"].unique()]
    plt.figure(figsize=(6, 6))
    sns.boxplot(x="HNF4A_group", y=tag, data=hnf4a_df, order=freq_order, 
                color="#74a892", flierprops=dict(marker='.', color='black', markersize=5))
    plt.xlabel("HNF4A motif frequency")
    plt.ylabel(tag)
    plt.title("HNF4A motif frequency vs. Expression")
    plt.savefig(f"./Figs/F04_interpret_robust/F04c_measured enhancer activity.pdf")