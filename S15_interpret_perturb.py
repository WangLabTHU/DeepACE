'''
分析HNF4A的扰动

/home/hyu/Digital_Platform/manuals/figs4_virtual_screen_interpret_2.py

cp -r /home/hyu/Digital_Platform/manuals/figs4_virtual_screen_interpret/analysis/warm_epigenetics_pseudo_random_mahalanobis/checks /home/hyu/DeepACE/Preds/D04_deeptfbu/
mv /home/hyu/Digital_Platform/manuals/figs4_virtual_screen_interpret/analysis/warm_epigenetics_pseudo_random_mahalanobis/fp_perturb_seqs_* /home/hyu/DeepACE/Preds/D04_deeptfbu/HNF4A_perturbs/
mv /home/hyu/Digital_Platform/manuals/figs4_virtual_screen_interpret/analysis/warm_epigenetics_pseudo_random_mahalanobis/tp_perturb_seqs_* /home/hyu/DeepACE/Preds/D04_deeptfbu/HNF4A_perturbs/

mv /home/hyu/Digital_Platform/manuals/figs4_virtual_screen_interpret/analysis/warm_epigenetics_pseudo_random_mahalanobis/*_perturb_lineplot_mean.npy /home/hyu/DeepACE/Preds/D04_deeptfbu/HNF4A_perturbs/
mv /home/hyu/Digital_Platform/manuals/figs4_virtual_screen_interpret/analysis/warm_epigenetics_pseudo_random_mahalanobis/*_perturb_lineplot_delta.npy /home/hyu/DeepACE/Preds/D04_deeptfbu/HNF4A_perturbs/

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

import re
BASE_DIR = "./Preds/D04_deeptfbu/HNF4A_checks/"
sys.path.append(BASE_DIR)
from SeqRegressionModel import DenseLSTM_classi


def write_txt(file, data):
    f = open(file,'w')
    i = 0
    while i < len(data):
        f.write(data[i] + '\n')
        i = i + 1
    f.close()

def open_fa(file):
    record = []
    f = open(file,'r')
    for item in f:
        if '>' not in item:
            record.append(item[0:-1])
    f.close()
    return record

def perturb_seq(ref_seq, start, end, rep=100):
    """Randomly mutate region [start, end) in ref_seq."""
    if start < 0 or end > len(ref_seq) or start >= end:
        raise ValueError("Invalid start or end for perturbation region.")
    alt_seqs = []
    for _ in range(rep):
        rand_region = ''.join(random.choice('ATCG') for _ in range(end - start))
        alt_seqs.append(ref_seq[:start] + rand_region + ref_seq[end:])
    return alt_seqs

def write_fasta(path, seqs, tags):
    """Write sequences and headers to FASTA."""
    with open(path, "w") as f:
        for tag, seq in zip(tags, seqs):
            f.write(f">{tag}\n{seq}\n")
    print(f"[✓] FASTA written to: {path}")

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


def seq_to_onehot(seq_list, length):
    data = np.zeros((len(seq_list),length,4))
    num_dic = {'A':0,'C':1,'G':2,'T':3}
    for i in range(len(seq_list)):
        for j in range(length):
            if seq_list[i][j]=='N':
                continue
            data[i][j][num_dic[seq_list[i][j]]]=1
    return data

def parse_tag(tag):
    m = re.match(r"deeptfbu_ref(\d+)_(\d+_\d+)_tile\d+_rep(\d+)", tag)
    return pd.Series({
        "ref": int(m.group(1)),
        "region": m.group(2),   # e.g. "0_20"
        "rep": int(m.group(3))
    })


''' Global Settings '''

labels_df = pd.read_excel("./Datas/D04_deeptfbu/3TF_MPRA.xlsx")
labels_df = labels_df[labels_df['enhancer sequence'].str.len() == 168]
labels_df = labels_df[labels_df['sequence_name'].str.contains("HNF4A_1_aim", na=False)] # 
df_focus = pd.read_csv(f"./total_features.csv")
df_focus["original_index"] = df_focus.index.tolist()
df_focus["feature_clean"] = df_focus["feature"].replace({"CHIP:": "CHIP-seq:","CEBPb": "CEBPB","CHIP-seq:3xFLAG-": "CHIP-seq:"}, regex=True)
df_focus["feature_group"] = df_focus["feature_clean"].apply(classify_feature)
df_focus["feature_channel"] = df_focus.apply(lambda row: f"({row['model']})-({row.name})-{row['feature_clean']}", axis=1)
df_focus = df_focus.drop(columns=["Unnamed: 0"])
primary_data = np.load(f"./Preds/D04_deeptfbu/valids_Epigenetics_HNF4A_1_aim/uni_pred.npy")
labels_df = pd.read_excel("./Datas/D04_deeptfbu/3TF_MPRA.xlsx")
labels_df = labels_df[labels_df['sequence_name'].str.contains("HNF4A_1_aim", na=False)]
labels_df["preds"] = [float(item.split("_")[0]) for item in labels_df["sequence_name"]]
idx = 3939
pred_list = primary_data[:, idx]
task_name = df_focus.iloc[idx]["feature_channel"]
labels_df[task_name] = pred_list
idx = 2217
pred_list = primary_data[:, idx]
task_name = df_focus.iloc[idx]["feature_channel"]
labels_df[task_name] = pred_list

tile_list = [10, 20, 30, 40, 50, 60, 70]
seq_len = 168
rep = 10
stride = 10

''' Dataset Preparation '''
# output_dir = "./Preds/D04_deeptfbu/HNF4A_perturbs"
# mode = "fp"
# if mode == "tp":
#     top_df_sorted = labels_df.sort_values(by="measured enhancer activity", ascending=False)
# elif mode == "fp":
#     top_df_sorted = labels_df.sort_values(by="preds", ascending=False)
# top_df = top_df_sorted[:50].reset_index(drop=True)
# seqs = top_df["enhancer sequence"].tolist()
# center_positions = list(range(40, 130+1, 10))  # 60, 70, ..., 110
# for tile in tile_list:
#     half = tile // 2
#     total_seqs = []
#     total_tags = []
#     for ref_idx, ref_seq in enumerate(seqs, start=1):
#         for center in tqdm(center_positions, desc=f"Tile {tile}", leave=False):
#             start = max(0, center - half)
#             end = min(seq_len, center + half)
#             mutated = perturb_seq(ref_seq, start, end, rep=rep)
#             for mut_idx, mut_seq in enumerate(mutated, start=1):
#                 tag = f"deeptfbu_ref{ref_idx}_{start}_{end}_tile{tile}_rep{mut_idx}"
#                 total_seqs.append(mut_seq)
#                 total_tags.append(tag)
#     pd.DataFrame({"seqs": total_seqs, "tags": total_tags}).to_csv(
#         os.path.join(output_dir, f"{mode}_perturb_seqs_{tile}.csv"),index=False)
#     write_fasta( os.path.join(output_dir, f"{mode}_perturb_seqs_{tile}.txt"),
#                 total_seqs, total_tags)


''' DeepTFBU Prediction '''

# mode = "tp"
# total_mean_preds = []
# total_delta_preds = []
# if mode == "tp":
#     top_df_sorted = labels_df.sort_values(by="measured enhancer activity", ascending=False)
# elif mode == "fp":
#     top_df_sorted = labels_df.sort_values(by="preds", ascending=False)
# model_list = []
# for i in range(10):
#     model_temp = DenseLSTM_classi(input_nc=4, growth_rate=32, block_config=(2, 2, 4, 2), num_init_features=400, bn_size=4, drop_rate=0.2, input_length=168)
#     model_temp=model_temp.cuda()
#     model_path = f"./Preds/D04_deeptfbu/HNF4A_checks/train_{i}/test_denselstm_mc_0.001_mask_168_HNF4A.pth"
#     model_temp = torch.load(model_path)
#     model_temp.eval()
#     model_list.append(model_temp)

# top_df = top_df_sorted[:50].reset_index(drop=True)
# for tile in tile_list:    
#     df = pd.read_csv(f"./Preds/D04_deeptfbu/HNF4A_perturbs/{mode}_perturb_seqs_{tile}.csv")
#     df[["ref", "region", "rep"]] = df["tags"].apply(parse_tag)
#     seqs = open_fa(f"./Preds/D04_deeptfbu/HNF4A_perturbs/{mode}_perturb_seqs_{tile}.txt")
#     onehot_new = seq_to_onehot(seqs, 168)
#     onehot_new = torch.tensor(onehot_new).float().cuda(non_blocking=True)
#     scores_new = []
#     batch_size = 64
#     num_seqs = onehot_new.shape[0]
#     num_models = 10
#     scores_new = torch.zeros((num_seqs, num_models))
#     for start_idx in tqdm(range(0, num_seqs, batch_size)):
#         end_idx = min(start_idx + batch_size, num_seqs)
#         batch_seqs = onehot_new[start_idx:end_idx]
#         for tr_num in range(num_models):
#             with torch.no_grad():
#                 result_temp = model_list[tr_num](batch_seqs)
#             result_temp = result_temp.detach().cpu().float()
#             result_temp = torch_F.softmax(result_temp, dim=1)
#             scores_new[start_idx:end_idx,tr_num] = result_temp[:, 1]
#     new_preds = scores_new.numpy().mean(axis=-1)
#     df["new_preds"] = new_preds

#     col = "preds"
#     baseline = top_df[col].values
#     baseline_expanded = np.repeat(baseline, 100)
#     delta_df = df.copy()
#     delta_df["new_preds"] = df["new_preds"].values - baseline_expanded

#     region_mean_preds = (df.groupby("region")["new_preds"].mean().sort_index(key=lambda idx: idx.str.split("_").str[0].astype(int)).values)
#     total_mean_preds.append(region_mean_preds)
#     region_delta_preds = (delta_df.groupby("region")["new_preds"].mean().sort_index(key=lambda idx: idx.str.split("_").str[0].astype(int)).values)
#     total_delta_preds.append(region_delta_preds)
    
# total_mean_preds = np.array(total_mean_preds)
# total_delta_preds = np.array(total_delta_preds)
# np.save(f"./Preds/D04_deeptfbu/HNF4A_perturbs/deeptfbu_{mode}_perturb_lineplot_mean.npy", total_mean_preds)
# np.save(f"./Preds/D04_deeptfbu/HNF4A_perturbs/deeptfbu_{mode}_perturb_lineplot_delta.npy", total_delta_preds)

''' DeepTFBU Lineplot '''

for mode in ["tp", "fp"]:
    tile_list = [10, 20, 30, 40, 50, 60, 70]
    total_mean_preds = np.load(f"./Preds/D04_deeptfbu/HNF4A_perturbs/deeptfbu_{mode}_perturb_lineplot_mean.npy")
    total_delta_preds = np.load(f"./Preds/D04_deeptfbu/HNF4A_perturbs/deeptfbu_{mode}_perturb_lineplot_delta.npy")
    centers = np.array([40, 50, 60, 70, 80, 90, 100, 110, 120, 130])
    
    # plt.figure(figsize=(4, 5))
    # tmp_tile_list = [tile_list[3]]
    # tmp_total_mean_preds = [total_mean_preds[3]]
    # linestyle = "solid" if mode == "tp" else "dashed"
    # for i, tile in enumerate(tmp_tile_list):
    #     plt.plot(
    #         centers,
    #         tmp_total_mean_preds[i],
    #         marker="o",
    #         label=f"tile={tile}",
    #         color="tab:green",
    #         linestyle=linestyle
    #     )
    # plt.ylim(0.4, 1.05)
    # plt.grid(alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(f"./Supps/S15_interpret_perturb/lineplot_deeptfbu_{mode}_mean.pdf", dpi=400)
    
    plt.figure(figsize=(4, 5))
    tmp_tile_list = [tile_list[3]]
    tmp_total_delta_preds = [total_delta_preds[3]]
    linestyle = "solid" if mode == "tp" else "dashed"
    for i, tile in enumerate(tmp_tile_list):
        plt.plot(
            centers,
            tmp_total_delta_preds[i],
            marker="o",
            label=f"tile={tile}",
            color="tab:green",
            linestyle=linestyle
        )
    plt.ylim(-0.35, 0.05)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./Supps/S15_interpret_perturb/lineplot_deeptfbu_{mode}_delta.pdf", dpi=400)


''' DeepACE Prediction '''

# mode = "fp"
# total_delta_preds = []
# total_mean_preds = []
# top_df_sorted = labels_df.sort_values(by="preds", ascending=False)
# top_df = top_df_sorted[:50].reset_index(drop=True)
    
# for tile in tile_list:
#     df = pd.read_csv(f"./Preds/D04_deeptfbu/HNF4A_perturbs/{mode}_perturb_seqs_{tile}.csv")
#     df[["ref", "region", "rep"]] = df["tags"].apply(parse_tag)
#     seqs = open_fa(f"./Preds/D04_deeptfbu/HNF4A_perturbs/{mode}_perturb_seqs_{tile}.txt")
#     onehot_new = seq_to_onehot(seqs, 168)
#     onehot_new = torch.tensor(onehot_new).float().cuda(non_blocking=True)
#     scores_new = []
#     batch_size = 64
#     num_seqs = onehot_new.shape[0]
#     num_models = 10
#     scores_new = torch.zeros((num_seqs, num_models))
#     uni_pred = np.load(f"./Preds/D04_deeptfbu/HNF4A_perturbs/{mode}_perturb_seqs_{tile}/uni_pred.npy")
#     df["(Enformer)-(3939)-CHIP-seq:HNF4A"] = uni_pred[:, 2827]
#     df["(Enformer)-(2217)-CHIP-seq:H3K9me3"] = uni_pred[:, 1105]
    
#     col = "(Enformer)-(3939)-CHIP-seq:HNF4A"
#     baseline = top_df[col].values
#     baseline_expanded = np.repeat(baseline, 100)
#     delta_df = df.copy()
#     delta_df[col] = df[col].values - baseline_expanded
    
#     region_mean_preds = (df.groupby("region")["(Enformer)-(3939)-CHIP-seq:HNF4A"].mean().sort_index(key=lambda idx: idx.str.split("_").str[0].astype(int)).values)
#     region_delta_preds = (delta_df.groupby("region")["(Enformer)-(3939)-CHIP-seq:HNF4A"].mean().sort_index(key=lambda idx: idx.str.split("_").str[0].astype(int)).values)
#     total_mean_preds.append(region_mean_preds)
#     total_delta_preds.append(region_delta_preds)

# total_mean_preds = np.array(total_mean_preds)
# total_delta_preds = np.array(total_delta_preds)
# np.save(f"./Preds/D04_deeptfbu/HNF4A_perturbs/deepace_{mode}_perturb_lineplot_mean.npy", total_mean_preds)
# np.save(f"./Preds/D04_deeptfbu/HNF4A_perturbs/deepace_{mode}_perturb_lineplot_delta.npy", total_delta_preds)

''' DeepACE Lineplot '''

for mode in ["tp", "fp"]:
    tile_list = [10, 20, 30, 40, 50, 60, 70]
    total_mean_preds = np.load(f"./Preds/D04_deeptfbu/HNF4A_perturbs/deepace_{mode}_perturb_lineplot_mean.npy")
    total_delta_preds = np.load(f"./Preds/D04_deeptfbu/HNF4A_perturbs/deepace_{mode}_perturb_lineplot_delta.npy")
    centers = np.array([40, 50, 60, 70, 80, 90, 100, 110, 120, 130])
    
    # plt.figure(figsize=(4, 5))
    # tmp_tile_list = [tile_list[3]]
    # tmp_total_mean_preds = [total_mean_preds[3]]
    # linestyle = "solid" if mode == "tp" else "dashed"
    # for i, tile in enumerate(tmp_tile_list):
    #     plt.plot(
    #         centers,
    #         tmp_total_mean_preds[i],
    #         marker="o",
    #         label=f"tile={tile}",
    #         color="tab:green",
    #         linestyle=linestyle
    #     )
    # plt.ylim(10, 55)
    # plt.grid(alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(f"./Supps/S15_interpret_perturb/lineplot_deepace_{mode}_mean.pdf", dpi=400)
    
    plt.figure(figsize=(4, 5))
    tmp_tile_list = [tile_list[3]]
    tmp_total_delta_preds = [total_delta_preds[3]]
    linestyle = "solid" if mode == "tp" else "dashed"
    for i, tile in enumerate(tmp_tile_list):
        plt.plot(
            centers,
            tmp_total_delta_preds[i],
            marker="o",
            label=f"tile={tile}",
            color="tab:green",
            linestyle=linestyle
        )
    plt.ylim(-35, 5)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./Supps/S15_interpret_perturb/lineplot_deepace_{mode}_delta.pdf", dpi=400)

