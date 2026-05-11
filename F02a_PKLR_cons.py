'''

point_MPRABase_ZRS_randaug
point_MPRABase_ZRS_randaug.fasta
point_MPRABase_ZRS_saturation
point_MPRABase_ZRS_saturation.fasta
point_MPRABase_ZRS_saturation.tsv

cp /home/hyu/Digital_Platform/manuals/fig_dataset/point_MPRABase_*_saturation.tsv  /home/hyu/DeepACE/Datas/D05_mprabase/
cp /home/hyu/Digital_Platform/manuals/fig_dataset/point_MPRABase_*_saturation.fasta  /home/hyu/DeepACE/Datas/D05_mprabase/
mv /home/hyu/Digital_Platform/manuals/fig_dataset/point_MPRABase_*  /home/hyu/DeepACE/Preds/D05_mprabase

mv /home/hyu/Digital_Platform/modals/VEP_scores/PhyloP470/hg38.phyloP470way.bw /home/hyu/DeepACE/Datas/D08_phylop
cp -r /home/hyu/Digital_Platform/manuals/fig2f_point_mutation_final/MPRABase_cosine/* /home/hyu/DeepACE/Preds/D05_mprabase/analysis_cosine/
cp -r /home/hyu/Digital_Platform/manuals/fig2f_point_mutation_final/MPRABase_mahalanobis/* /home/hyu/DeepACE/Preds/D05_mprabase/analysis_mahalanobis/

cp /home/hyu/Digital_Platform/manuals/fig1e_mutational_effects/PKLR_logo_and_PCA_LFC.pdf /home/hyu/DeepACE/Figs/F02_variant_effects/F02a_PKLR_cons.pdf
cp /home/hyu/Digital_Platform/manuals/fig1e_mutational_effects/PKLR_logo_and_PCA_LFC.csv /home/hyu/DeepACE/Figs/F02_variant_effects/F02a_PKLR_cons.csv
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


motif = 'PKLR'
name = motif
file_path = f"./Preds/D05_mprabase/point_MPRABase_{name}_saturation.tsv"
df_var = pd.read_csv(file_path, sep="\t")
start = df_var['Position'].iloc[0]-1
end = df_var['Position'].iloc[-1]
chrom = 'chr'+str(df_var['Chromosome'].iloc[0])

df_min10 = df_var.nsmallest(20, "VariantExpressionEffect (log2)")
min_positions = df_min10["Position"].tolist()
bw = pyBigWig.open("./Datas/D08_phylop")
scores = bw.values(chrom, start, end)
scores = np.nan_to_num(scores, nan=0.0)
bw.close()

fasta = Fasta("./Datas/D02_grch/GRCh38.primary_assembly.genome.fa")
seq = fasta[chrom][start:end].seq.upper()
seq = seq[50:110]
scores = scores[50:110]
df_logo = pd.DataFrame(0, index=range(len(seq)), columns=['A','C','G','T'])
for i, base in enumerate(seq):
    df_logo.loc[i, base] = scores[i]

# -----------------------------
# PCA_Conservation
# -----------------------------
# path = "/home/hyu/Digital_Platform/manuals/fig2f_point_mutation_final/conservation/per_base_conservation_PKLR.csv"
path = "./Preds/D05_mprabase/analysis_cosine/conservation/per_base_conservation_PKLR.csv"
df_conserve = pd.read_csv(path)
values_pca = df_conserve["PCA_Conservation"] + 1
subset_pca = values_pca.iloc[50 : 110]

# -----------------------------
# LFC_Conservation
# -----------------------------
values_lfc = df_conserve["LFC_Conservation"] + 1
subset_lfc = values_lfc.iloc[50 : 110]

fig, axes = plt.subplots(3, 1, figsize=(7, 5))
logo = logomaker.Logo(df_logo, ax=axes[0], shade_below=0, fade_below=0, stack_order='big_on_top')
logo.style_xticks(rotation=0, fmt='%d', anchor=0)
axes[0].set_xticks([])
axes[0].set_xlabel('')
axes[0].set_ylabel('phyloP score', fontsize=16)
axes[0].tick_params(axis='y', labelsize=16)
axes[0].set_ylim(-12, 12)
axes[0].set_title(f"{chrom}:{start+50:,}-{start+110-1:,}", fontsize=16)

# PCA_Conservation
x = range(len(subset_pca))
axes[1].plot(x, subset_pca.values, linewidth=2, color='#e5c185')
axes[1].set_xticks([])
axes[1].set_xlabel('')
axes[1].set_ylabel("Cosine distance \n (model)", fontsize=16)
axes[1].tick_params(axis='y', labelsize=16)
axes[1].set_xlim(0, len(subset_pca)-1)
axes[1].axhline(y=1, color='k', linestyle='--', linewidth=1)

# LFC_Conservation
axes[2].plot(x, subset_lfc.values, linewidth=2, color='green')
axes[2].set_xticks([])
axes[2].set_xlabel('')
axes[2].set_ylabel("ΔLogFC \n (Exp)", fontsize=16)
axes[2].tick_params(axis='y', labelsize=16)
axes[2].set_xlim(0, len(subset_lfc)-1)
axes[2].axhline(y=0, color='k', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig("./Figs/F02_variant_effects/F02a_PKLR_cons.pdf", dpi=400)
plt.close()

df_save = df_logo.copy()
df_save["Cosine_distance"] = subset_pca.values
df_save["Delta_LogFC"] = subset_lfc.values
df_save.to_csv("./Figs/F02_variant_effects/F02a_PKLR_cons.csv", index=True)


