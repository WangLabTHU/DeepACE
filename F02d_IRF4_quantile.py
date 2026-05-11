'''
/home/hyu/Figures/DeepACE/Fig2.py

cp /home/hyu/Figures/DeepACE/Fig2/Fig2d_curve_IRF4.svg /home/hyu/DeepACE/Figs/F02_variant_effects/F02d_IRF4_quantile.svg
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


df = pd.read_csv("./Preds/D05_mprabase/analysis_cosine/quantile_discover_rate_IRF4.csv")
x = df["quantile"].values

colors = {
    "DeepACE": "#4C72B0",
    "Evo2": "#DD8452",
    "promoterAI": "#55A868",
    "phyloP 100": "#5A3E8C",
    "phyloP 470": "#B39DDB",
    "phastCons 100": "#00A6D6",
    "phastCons 470": "#7FDBFF",
    "GPN-MSA": "#636363"
}

cols = {
    "DeepACE": "discover_rate_PCA",
    "Evo2": "discover_rate_Evo2",
    "promoterAI": "discover_rate_promoterAI",
    "phyloP 100": "discover_rate_phyloP100way",
    "phyloP 470": "discover_rate_phyloP470way",
    "phastCons 100": "discover_rate_phastCons100way",
    "phastCons 470": "discover_rate_phastCons470way",
    "GPN-MSA": "discover_rate_gpnmsa"
}
plt.figure(figsize=(7,5))

for name, col in cols.items():
    y = df[col].values
    auc = np.trapz(y, x)
    if name == "DeepACE":
        lw, alpha = 2.8, 1.0
    else:
        lw, alpha = 1.2, 0.35
    plt.plot(x, y, color=colors[name], linewidth=lw, alpha=alpha,
            label=f"{name} (AUC={auc:.2f})")

plt.xlabel("Score quantile", fontsize=20)
plt.ylabel("Discovery Rate", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(bbox_to_anchor=(0.7, 0), loc="lower left", fontsize=16, frameon=False)
plt.title("IRF4 enhancer discovery rate curves", fontsize=20)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("/home/hyu/Figures/DeepACE/Fig2/Fig2d_curve_IRF4.svg", bbox_inches="tight")
plt.close()