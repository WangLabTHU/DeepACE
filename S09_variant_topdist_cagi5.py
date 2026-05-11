'''
画序列的突变后的小提琴图, 在CAGI5的15个数据集

/home/hyu/Digital_Platform/manuals/fig2e_point_mutation.py
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
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import mahalanobis


def compute_sample_similarity(pred_alt, pred_ref):
    pred_alt = np.asarray(pred_alt)
    pred_ref = np.asarray(pred_ref)
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


''' Violin Visualization '''


metrics = ["cosine", "mahalanobis"]
datasets = ["MPRABase"] #   

metric = "mahalanobis"
dataset = "MPRABase"
motif_list = ["TERT", "HBG1", "LDLR", "F9", "GP1BA", "IRF4", "IRF6", "PKLR", "ZFAND3", "SORT1", 
              "HBB", "UC88", "MYC_rs6983267", "RET", "TCF7L2"] 

output_dir = f"./Supps/S09_variant_topdist_cagi5/"
print(f"Processing dataset: {dataset}")
all_similarities, all_groups, all_sources = [], [], []
for motif in motif_list:
    print(f"Processing motif/background: {motif}")
    uni_path = f"./Preds/D05_mprabase/point_{dataset}_{motif}_saturation/uni_pred.npy"
    df_path = f"./Datas/D05_mprabase/point_{dataset}_{motif}_saturation.tsv"
    uni_list = np.load(uni_path)
    df = pd.read_csv(df_path, sep="\t")
    variant_effects = df['VariantExpressionEffect (log2)'].to_numpy()
    pred_alt = uni_list[:-1]
    pred_ref = np.repeat(uni_list[-1][np.newaxis, :], len(pred_alt), axis=0)
    valid_mask = np.isfinite(pred_alt).any(axis=0) & np.isfinite(pred_ref).any(axis=0)
    pred_alt = pred_alt[:, valid_mask]
    pred_ref = pred_ref[:, valid_mask]
    pred_rand = np.load("./Preds/D10_random/random_sample_1/uni_pred.npy")
    pred_rand = pred_rand[:, valid_mask]
    combined = np.vstack([pred_alt, pred_ref, pred_rand])
    combined_pca = PCA(n_components=50, random_state=42).fit_transform(combined)
    pred_alt_pca = combined_pca[:len(pred_alt)]
    pred_ref_pca = combined_pca[len(pred_alt):-len(pred_rand)]
    pred_rand_pca = combined_pca[-len(pred_rand):]
    
    similarity_all = compute_sample_similarity(pred_alt_pca, pred_rand_pca)
    K = 10
    idx_max = np.argsort(variant_effects)[-K:]
    idx_min = np.argsort(variant_effects)[:K]
    all_similarities.extend(list(similarity_all[idx_max]) + list(similarity_all[idx_min]))
    all_groups.extend(['max_score']*K + ['min_score']*K)
    all_sources.extend([motif]*(2*K))

all_similarities = np.array(all_similarities)
all_distances = 1 - all_similarities
data_combined = pd.DataFrame({'distance': all_distances, 'group': all_groups,'source': all_sources})
# data_combined.to_csv(f"{output_dir}/randaug_distance_top10_violin.csv", index=False)
plt.figure(figsize=(18, 6))
sns.boxplot(data=data_combined, x='source', y='distance', hue='group', palette=["tab:red", "tab:blue"])
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.title(f'Distance: Variant vs Original ({dataset}, {metric})', fontsize=16)
plt.ylabel('Distance', fontsize=16)
plt.xlabel('Source', fontsize=16)
plt.xticks(fontsize=16, rotation=45)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig(f"{output_dir}/randaug_distance_top10_violin.pdf", dpi=400)