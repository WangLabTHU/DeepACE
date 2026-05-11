'''
/home/hyu/Figures/DeepACE/Fig2.py

cp /home/hyu/Figures/DeepACE/Fig2/Fig2c_DeepACE_scatter_IRF4.svg /home/hyu/DeepACE/Figs/F02_variant_effects/F02c_IRF4_pcc_1.svg
cp /home/hyu/Figures/DeepACE/Fig2/Fig2c_Evo2_scatter_IRF4.svg /home/hyu/DeepACE/Figs/F02_variant_effects/F02c_IRF4_pcc_2.svg
cp /home/hyu/Figures/DeepACE/Fig2/Fig2c_promoterAI_scatter_IRF4.svg /home/hyu/DeepACE/Figs/F02_variant_effects/F02c_IRF4_pcc_3.svg
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




input_file = "./Preds/D05_mprabase/analysis_cosine/pcc_lineplot_IRF4.csv"
output_dir = "./Figs/F02_variant_effects"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_file)
df["model"] = df["model"].replace({"PCA": "DeepACE"})
x_min, x_max = df["normalized_scores"].min(), df["normalized_scores"].max()
y_min, y_max = df["variant_effects"].min(), df["variant_effects"].max()

models = ["DeepACE", "Evo2", "promoterAI"]
colors = {"DeepACE": "#4C72B0", "Evo2": "#DD8452", "promoterAI": "#55A868"}

for i, model in enumerate(models):
    subset = df[df["model"] == model]
    x = subset["normalized_scores"].values
    y = subset["variant_effects"].values
    r, p = pearsonr(x, y)
    coef = np.polyfit(x, y, 1)
    poly_fn = np.poly1d(coef)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.scatter(x, y, s=18, alpha=0.7, color=colors[model], edgecolors="black", linewidths=0.3)
    xx = np.linspace(x_min, x_max, 100)
    ax.plot(xx, poly_fn(xx), linewidth=1.5, color="black")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Normalized Scores", fontsize=20)
    ax.set_ylabel("Variant Effects", fontsize=20)
    ax.set_title(model, fontsize=20)
    ax.text(0.70, 0.10, f"r = {r:.2f}", transform=ax.transAxes,
            verticalalignment="top", fontsize=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='both', labelsize=20)
    output_path = os.path.join(output_dir, f"F02c_IRF4_pcc_{i}.svg")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()