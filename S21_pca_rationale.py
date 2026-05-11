'''
PCA的参数控制过程

/home/hyu/Digital_Platform/manuals/fig1a_pca_rationale.py

cp  /home/hyu/Digital_Platform/manuals/fig1a_pca_rationale/pca_model_*.pkl /home/hyu/DeepACE/Supps/S21_pca_rationale/
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

import joblib
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


''' Training Models '''
# uni_pred = []
# for run_id in range(5):
#     run_pred = np.load(f"./Preds/D01_screens/CRE_samples_{run_id}/uni_pred.npy")  # (10000, 43275) 
#     uni_pred += list(run_pred)
# uni_pred = np.array(uni_pred)
# K_list = [50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
# for K in tqdm(K_list):
#     pca = PCA(n_components=50, random_state=42)
#     uni_selected = pca.fit_transform(uni_pred[:K])  # (50000, 50)
#     joblib.dump(pca, f'./Supps/S21_pca_rationale/pca_model_{K}.pkl')

''' Scatter Plot '''

def get_flattened_pc12(pca_model):
    pc1 = pca_model.components_[0]  # (43275,)
    pc2 = pca_model.components_[1]  # (43275,)
    return np.concatenate([pc1, pc2])

K_list = [50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
pca_models = {}
for K in K_list:
    pca_models[K] = joblib.load(f'./Supps/S21_pca_rationale/pca_model_{K}.pkl')
large_flattened = get_flattened_pc12(pca_models[50000])
pairs = [k for k in K_list if k != 50000]
for k_small in pairs:
    small_flattened = get_flattened_pc12(pca_models[k_small])
    plt.figure(figsize=(8, 8))
    plt.scatter(small_flattened, large_flattened, s=1, alpha=0.6)
    min_val = min(small_flattened.min(), large_flattened.min())
    max_val = max(small_flattened.max(), large_flattened.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1, label='y = x')
    # plt.title(f'PCA Parameters Comparison: K={k_small} vs K=50000\n(PC1 & PC2 loadings, 86550 points)')
    # plt.xlabel(f'Flattened PC1 & PC2 loadings (fitted on {k_small} samples)')
    # plt.ylabel('Flattened PC1 & PC2 loadings (fitted on 50000 samples)')
    plt.grid(True, alpha=0.3)
    r, _ = pearsonr(small_flattened, large_flattened)
    plt.text(0.95, 0.05, f'r = {r:.4f}',
             transform=plt.gca().transAxes,
             fontsize=18, fontweight='bold',
             horizontalalignment='right',
             verticalalignment='top',
             bbox=dict(boxstyle='square, pad=0.5', facecolor='white', alpha=0.7, edgecolor=None))
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.locator_params(axis='x', nbins=5)
    plt.tight_layout()
    plt.savefig(f"./Supps/S21_pca_rationale/scatter_pca_model_{k_small}.png", dpi=400)
    

''' Line Plot '''

pca_large = joblib.load('./Supps/S21_pca_rationale/pca_model_50000.pkl')
explained_var_ratio = pca_large.explained_variance_ratio_
cum_explained_var = np.cumsum(explained_var_ratio)
pc_indices = np.arange(1, len(explained_var_ratio) + 1)
plt.figure(figsize=(10, 6))
line1 = plt.plot(pc_indices, explained_var_ratio * 100,
                 color='steelblue', marker='o', markersize=5, linewidth=2.5,
                 label='Individual Explained Variance')[0]
ax2 = plt.twinx()
line2 = ax2.plot(pc_indices, cum_explained_var * 100,
                 color='darkred', marker='s', markersize=5, linewidth=2.5,
                 label='Cumulative Explained Variance')[0]
ax2.set_ylim(0, 105)
ax2.set_ylabel('Cumulative Explained Variance (%)', color='darkred', fontsize=12)
ax2.tick_params(axis='y', labelcolor='darkred')
plt.xlim(0.5, len(pc_indices) + 0.5)
plt.xlabel('Principal Component', fontsize=12)
plt.ylabel('Individual Explained Variance (%)', fontsize=12)
plt.title('Explained Variance by Principal Components (K=50,000 samples)',
          fontsize=14, pad=20)
i = 9  # index for PC10
plt.text(pc_indices[i], explained_var_ratio[i]*100,
         f'{explained_var_ratio[i]*100:.2f}%',
         ha='center', va='bottom', fontsize=10, color='steelblue',
         fontweight='bold')
ax2.text(pc_indices[i], cum_explained_var[i]*100 + 3,  # +3% upward offset
         f'{cum_explained_var[i]*100:.2f}%',
         ha='center', va='bottom', fontsize=10, color='darkred',
         fontweight='bold')
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.gca().spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.legend(handles=[line1, line2], loc='upper right', frameon=False)
plt.tight_layout()
plt.savefig('./Supps/S21_pca_rationale/lineplot_pca_model.png', dpi=400, bbox_inches='tight')

