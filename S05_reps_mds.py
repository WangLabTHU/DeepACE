'''
画序列的表征在不同采样设置下的MDS (默认非cold, 附加上cold的形式)

/home/hyu/Digital_Platform/manuals/fig1c_mds.py
/home/hyu/Digital_Platform/manuals/fig1c_mds_cold.py

cp /home/hyu/Digital_Platform/manuals/fig1a_pca_analysis/pca_model.pkl /home/hyu/DeepACE/Preds/D01_screens
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
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

def silhouette_single_class(X):
    dist = pairwise_distances(X, metric='euclidean')
    a = np.mean(dist, axis=1)
    return float(np.mean(a))

def draw_graph_with_embedding(embedding, labels, G, save_path, title='Embedding'):
    pos = {i:(embedding[i,0], embedding[i,1]) for i in range(embedding.shape[0])}
    pos_nodes = [i for i in range(embedding.shape[0]) if labels[i]==1]
    neg_nodes = [i for i in range(embedding.shape[0]) if labels[i]==0]
    plt.figure(figsize=(5,4))
    pos_scatter =nx.draw_networkx_nodes(G, pos, nodelist=pos_nodes, node_color='#e5c185', node_size=30, alpha=0.8)
    neg_scatter =nx.draw_networkx_nodes(G, pos, nodelist=neg_nodes, node_color='#74a892', node_size=30, alpha=0.8)
    pos_edges = [(u, v) for u, v in G.edges() if labels[u] == 1 and labels[v] == 1]
    neg_edges = [(u, v) for u, v in G.edges() if labels[u] == 0 and labels[v] == 0]
    sil_pos = silhouette_single_class(embedding[pos_nodes])
    sil_neg = silhouette_single_class(embedding[neg_nodes])
    plt.legend([pos_scatter, neg_scatter], 
               [f'Positive', f'Negative'], 
               scatterpoints=1, loc='upper right', fontsize=14, frameon=True, handletextpad=0.05)
    ax = plt.gca()
    plt.tick_params(axis='both', labelsize=16)
    plt.title('Structural Patterns in Functional Space', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()


''' Random MDS '''

# for dataset in ["DS", "MPRA", "Epigenetics", "lentiMPRA"]: 
#     if dataset == "DS":
#         plot_tag_list = ["DS0001-SID01", "DS0001-SID02", "DS0002-SID02"]  
#     elif dataset == "MPRA":
#         plot_tag_list = ["HepG2", "K562", "SKNSH"]  
#     elif dataset == "Epigenetics":
#         plot_tag_list = ["118TF"]
#     elif dataset == "lentiMPRA":
#         plot_tag_list = ["HepG2", "K562", "WTC11"] 
#     ratio_list = ["P10-N1", "P1-N10", "P10-N10"] 
#     for plot_tag in plot_tag_list:
#         if dataset == "MPRA":
#             file_pos = f"./Preds/D06_mpra/control_{dataset}_{plot_tag}_pos/uni_pred.npy"
#             file_neg = f"./Preds/D06_mpra/control_{dataset}_{plot_tag}_neg/uni_pred.npy"
#         elif dataset == "DS":
#             file_pos = f"./Preds/D05_mprabase/control_{plot_tag}_HepG2_pos/uni_pred.npy"
#             file_neg = f"./Preds/D05_mprabase/control_{plot_tag}_HepG2_neg/uni_pred.npy"
#         elif dataset == "Epigenetics":
#             file_pos = f"./Preds/D04_deeptfbu/control_{dataset}_{plot_tag}_pos/uni_pred.npy"
#             file_neg = f"./Preds/D04_deeptfbu/control_{dataset}_{plot_tag}_neg/uni_pred.npy"
#         elif dataset == "lentiMPRA":
#             file_pos = f"./Preds/D07_lentimpra/control_{dataset}_{plot_tag}_pos/uni_pred.npy"
#             file_neg = f"./Preds/D07_lentimpra/control_{dataset}_{plot_tag}_neg/uni_pred.npy"
#         for ratio_tag in ratio_list:
#             category = f"{dataset}_{plot_tag}-{ratio_tag}"
#             data_pos = np.load(file_pos)
#             data_neg = np.load(file_neg)
#             if ratio_tag == "P10-N10":
#                 pos_num = 85
#                 neg_num = 85
#             elif ratio_tag == "P10-N1":
#                 pos_num = 85
#                 neg_num = 8
#             elif ratio_tag == "P1-N10":
#                 pos_num = 8
#                 neg_num = 85
#             np.random.seed(42)
#             idx_pos = np.random.choice(data_pos.shape[0], pos_num, replace=False)
#             idx_neg = np.random.choice(data_neg.shape[0], neg_num, replace=False)
#             data_pos = data_pos[idx_pos]
#             data_neg = data_neg[idx_neg]
#             data = np.vstack([data_pos, data_neg])
#             scaler = StandardScaler()
#             data = scaler.fit_transform(data)
#             labels = np.array([1]*data_pos.shape[0] + [0]*data_neg.shape[0])
#             pca = PCA(n_components=50, random_state=42)
#             data_pca = pca.fit_transform(data)
#             mds = MDS(n_components=2, max_iter=300, n_init=4, random_state=42, dissimilarity='euclidean')
#             emb_mds = mds.fit_transform(data_pca)
#             G = nx.Graph()
#             k = 3
#             for label in [0, 1]:
#                 idx = np.where(labels == label)[0]
#                 X = emb_mds[idx]
#                 nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
#                 distances, indices = nbrs.kneighbors(X)
#                 for i in range(len(idx)):
#                     for j in indices[i, 1:]:
#                         G.add_edge(idx[i], idx[j])
#             draw_graph_with_embedding(emb_mds, labels, G, f'./Supps/S05_reps_mds/proj_mds_rand_{category}.pdf', title='mds')
            # np.savez(f'./Supps/S05_reps_mds/proj_mds_rand_{category}.npz', emb_mds=emb_mds, labels=labels)


''' Cold MDS '''

for dataset in ["DS", "MPRA", "Epigenetics", "lentiMPRA"]: 
    if dataset == "DS":
        plot_tag_list = ["DS0001-SID01", "DS0001-SID02", "DS0002-SID02"]  
    elif dataset == "MPRA":
        plot_tag_list = ["HepG2", "K562", "SKNSH"]  
    elif dataset == "Epigenetics":
        plot_tag_list = ["118TF"]
    elif dataset == "lentiMPRA":
        plot_tag_list = ["HepG2", "K562", "WTC11"] 
    ratio_list = ["P10-N1", "P1-N10", "P10-N10"] 
    for plot_tag in plot_tag_list:
        if dataset == "MPRA":
            file_pos = f"./Preds/D06_mpra/control_{dataset}_{plot_tag}_pos/uni_pred.npy"
            file_neg = f"./Preds/D06_mpra/control_{dataset}_{plot_tag}_neg/uni_pred.npy"
        elif dataset == "DS":
            file_pos = f"./Preds/D05_mprabase/control_{plot_tag}_HepG2_pos/uni_pred.npy"
            file_neg = f"./Preds/D05_mprabase/control_{plot_tag}_HepG2_neg/uni_pred.npy"
        elif dataset == "Epigenetics":
            file_pos = f"./Preds/D04_deeptfbu/control_{dataset}_{plot_tag}_pos/uni_pred.npy"
            file_neg = f"./Preds/D04_deeptfbu/control_{dataset}_{plot_tag}_neg/uni_pred.npy"
        elif dataset == "lentiMPRA":
            file_pos = f"./Preds/D07_lentimpra/control_{dataset}_{plot_tag}_pos/uni_pred.npy"
            file_neg = f"./Preds/D07_lentimpra/control_{dataset}_{plot_tag}_neg/uni_pred.npy"
        for ratio_tag in ratio_list:
            category = f"{dataset}_{plot_tag}-{ratio_tag}"
            data_pos = np.load(file_pos)
            data_neg = np.load(file_neg)
            if ratio_tag == "P10-N10":
                pos_num = 85
                neg_num = 85
            elif ratio_tag == "P10-N1":
                pos_num = 85
                neg_num = 8
            elif ratio_tag == "P1-N10":
                pos_num = 8
                neg_num = 85
            np.random.seed(42)
            idx_pos = np.random.choice(data_pos.shape[0], pos_num, replace=False)
            idx_neg = np.random.choice(data_neg.shape[0], neg_num, replace=False)
            data_pos = data_pos[idx_pos]
            data_neg = data_neg[idx_neg]
            data = np.vstack([data_pos, data_neg])
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            labels = np.array([1]*data_pos.shape[0] + [0]*data_neg.shape[0])
            pca = joblib.load('./Preds/D01_screens/pca_model.pkl')
            data_pca = pca.fit_transform(data)
            mds = MDS(n_components=2, max_iter=300, n_init=4, random_state=42, dissimilarity='euclidean')
            emb_mds = mds.fit_transform(data_pca)
            G = nx.Graph()
            k = 3
            for label in [0, 1]:
                idx = np.where(labels == label)[0]
                X = emb_mds[idx]
                nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
                distances, indices = nbrs.kneighbors(X)
                for i in range(len(idx)):
                    for j in indices[i, 1:]:
                        G.add_edge(idx[i], idx[j])
            draw_graph_with_embedding(emb_mds, labels, G, f'./Supps/S05_reps_mds/proj_mds_cold_{category}.pdf', title='mds')
            # np.savez(f'./Supps/S05_reps_mds/proj_mds_cold_{category}.npz', emb_mds=emb_mds, labels=labels)