'''
/home/hyu/Digital_Platform/manuals/fig2f_point_mutation_phastcons.py

mv /home/hyu/Digital_Platform/modals/VEP_scores/PhyloP100/hg38.phyloP100way.bw /home/hyu/DeepACE/Datas/D08_phylop
mv /home/hyu/Digital_Platform/modals/VEP_scores/phastCons100/hg38.phastCons100way.bw /home/hyu/DeepACE/Datas/D09_phastcons
mv /home/hyu/Digital_Platform/modals/VEP_scores/phastCons470/hg38.phastCons470way.bw /home/hyu/DeepACE/Datas/D09_phastcons

cp /home/hyu/Digital_Platform/manuals/fig2f_point_mutation_phastcons/* /home/hyu/DeepACE/Preds/D05_mprabase/analysis_cons

'''


import os
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
import pysam
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, spearmanr
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

random.seed(42)
np.random.seed(42)



def Generate_model_scores(chrom, start, end):
    bw_paths = [
    "./Datas/D08_phylop/hg38.phyloP100way.bw",
    "./Datas/D08_phylop/hg38.phyloP470way.bw",
    "./Datas/D09_phastcons/hg38.phastCons100way.bw",
    "./Datas/D09_phastcons/hg38.phastCons470way.bw"
    ]

    positions = np.arange(start, end)
    score_dict = {}

    for path in bw_paths:
        bw = pyBigWig.open(path)        
        scores = bw.values(chrom, start, end, numpy=True)
        bw.close()
        scores = np.array([float("nan") if v is None else -v for v in scores])
        score_dict[path.split(".")[-2]] = scores    
    return score_dict


motif_map = {
    "MPRABase": ["TERT", "HBG1", "LDLR", "F9", "GP1BA", "IRF4", "IRF6", "PKLR", "ZFAND3", "SORT1",
                 "HBB", "UC88", "MYC_rs6983267", "RET", "TCF7L2"] 
    }
datasets = ["MPRABase"] 
output_root = "./Preds/D05_mprabase/analysis_cons"
os.makedirs(output_root, exist_ok=True)

for dataset in datasets:
    print(f"Processing dataset: {dataset}")    
    motif_list = motif_map[dataset]
    for motif in motif_list:
        print(f"Processing motif/background: {motif}")
        df_path = f"./Preds/D05_mprabase/analysis_cons/point_{dataset}_{motif}_saturation.tsv"
        df = pd.read_csv(df_path, sep="\t")
        variant_effects = df['VariantExpressionEffect (log2)'].to_numpy()

        chrom = df.iloc[0]["Chromosome"]
        start = df.iloc[0]["Position"]
        end = df.iloc[-1]["Position"]
        
        chrom = str(chrom)
        start = int(start) - 1
        end = int(end)
        if not chrom.lower().startswith("chr"):
            chrom = "chr" + chrom
        
        scores_dict = Generate_model_scores(chrom, start, end)
        position_list = df["Position"].tolist()
        index_list = [item - start - 1 for item in position_list]
        
        for tag in ["phyloP100way", "phyloP470way", "phastCons100way", "phastCons470way"]:
            scores = scores_dict[tag]
            scores = [scores[index] for index in index_list]
            output_path = f"{output_root}/{tag}_variant_scores_{motif}.csv"
            pd.DataFrame({
                "scores": scores,
                "variant_effects": variant_effects
            }).to_csv(output_path, index=False)