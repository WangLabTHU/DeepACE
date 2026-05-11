'''
数据集过滤, 可视化展示数据集

/home/hyu/Digital_Platform/manuals/fig2b_dataset_scatter.py
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

from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from umap import UMAP
from matplotlib.colors import TwoSlopeNorm


''' MPRA HepG2-SKNSH'''

df = pd.read_csv("./Datas/D06_mpra/valids.csv")
name = "AdaLead"
cell_type_list = ['HepG2', 'K562', 'SKNSH']
for cell_type in cell_type_list:
    df_screening = df[df["origin"].isin([name])]
    x = df_screening[cell_type+'_prediction']
    y = df_screening[cell_type+'_l2fc']
    top400_values = x.nlargest(500)
    thx = top400_values.min()
    mask_high = x >= thx
    mask_low = x < thx
    tophalf_values = y[mask_high].nlargest(250)
    thy = tophalf_values.min()
    mask_tp = (x >= thx) & (y >= thy)
    mask_fp = (x >= thx) & (y < thy)
    plt.figure(figsize=(5, 4))
    sns.set(style="whitegrid", font_scale=1.2)
    plt.scatter(x[mask_low], y[mask_low], s=40, alpha=0.4, edgecolors='k', linewidth=0, label=f"Low Prediction", color='grey')
    plt.scatter(x[mask_fp], y[mask_fp], s=40, alpha=0.9, edgecolors='k', linewidth=0.8, label=f"False Positive", color='white')
    plt.scatter(x[mask_tp], y[mask_tp], s=40, alpha=0.9, edgecolors='k', linewidth=0.8, label=f"True Positive", 
                c=y[mask_tp], cmap='Greens', vmin=np.min(y[mask_tp]), vmax=np.max(y[mask_tp]))
    pcc = pearsonr(x[mask_high], y[mask_high])[0]  
    plt.xlabel(f"Prediction", fontsize=16)
    plt.ylabel(f"LogFC", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"./Supps/S11_filter_data/dist_mpra_{cell_type}.pdf", dpi=400, bbox_inches='tight')
    plt.close()
    print(f"celltype: {cell_type}, Pearsonr: {pearsonr(x[mask_high], y[mask_high])[0]}")
    print(f"celltype: {cell_type}, Pearsonr: {pearsonr(x, y)[0]}")
    # category = pd.Series("Low Prediction", index=x.index)
    # category[mask_fp] = "False Positive"
    # category[mask_tp] = "True Positive"
    # df_save = pd.DataFrame({
    #     "Prediction": x.values,
    #     "LogFC": y.values,
    #     "Category": category.values
    # })
    # csv_path = f"/home/hyu/DeepACE/Supps/S11_filter_data/dist_mpra_{cell_type}.csv"
    # df_save.to_csv(csv_path, index=False)



''' Epigenetics ELF1-HNF4A'''

motif_list = ["ELF1", "HNF1A", "HNF4A"]
for motif in motif_list:
    df = pd.read_excel("./Datas/D04_deeptfbu/3TF_MPRA.xlsx")
    df_screening = df[df['sequence_name'].str.contains(motif, na=False)]
    name_list = df_screening["sequence_name"].tolist()
    labels_list = df["measured enhancer activity"].tolist()
    x, y = [], []
    for i in range(len(name_list)):
        name = name_list[i]
        label = labels_list[i]
        try:
            x.append( np.log2(float(name.split("_")[0])) )
            y.append( np.log2(label) )  
        except:
            continue
    x , y = np.array(x), np.array(y)
    sorted_x = np.sort(x)[::-1]  
    top500_values = sorted_x[:500] 
    thx = top500_values.min()
    mask_high = x >= thx
    mask_low = x < thx
    y_high_x = y[mask_high]
    sorted_y_high = np.sort(y_high_x)[::-1]
    top250_values = sorted_y_high[:250]
    thy = top250_values.min()     
    mask_tp = (x >= thx) & (y >= thy)
    mask_fp = (x >= thx) & (y < thy)
    plt.figure(figsize=(5, 4))
    sns.set(style="whitegrid", font_scale=1.2)
    plt.scatter(x[mask_low], y[mask_low], s=40, alpha=0.4, edgecolors='k', linewidth=0, label=f"Low Prediction", color='grey')
    plt.scatter(x[mask_fp], y[mask_fp], s=40, alpha=0.9, edgecolors='k', linewidth=0.8, label=f"False Positive", color='white')
    plt.scatter(x[mask_tp], y[mask_tp], s=40, alpha=0.9, edgecolors='k', linewidth=0.8, label=f"True Positive", 
                    c=y[mask_tp], cmap='Greens', vmin=np.min(y[mask_tp]), vmax=np.max(y[mask_tp]))
    pcc = pearsonr(x[mask_high], y[mask_high])[0]
    plt.xlabel(f"Predicted Activity ({motif}_prediction)", fontsize=14)
    plt.ylabel(f"Measured Activity ({motif}_l2fc)", fontsize=14)
    plt.title(f"Scatter Plot ({motif}_{len(x[mask_high])})", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./Supps/S11_filter_data/dist_epigenetics_{motif}.pdf", dpi=400, bbox_inches='tight')
    plt.close()
    print(f"motif: {motif}, Pearsonr: {pearsonr(x[mask_high], y[mask_high])[0]}")