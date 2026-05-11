'''
/home/hyu/Digital_Platform/manuals/fig2f_point_mutation_gpnmsa.py

mv /home/hyu/DeepACE/Preds/D05_mprabase/analysis_cons/gpnmsa_variant_* /home/hyu/DeepACE/Preds/D05_mprabase/analysis_gpnmsa
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
    bgz_path = "./Datas/D12_gpnmsascores.tsv.bgz" # 0-based
    tb = pysam.TabixFile(bgz_path)
    chrom_key = chrom.replace("chr", "")
    # 0-4 -> | chrom | pos | ref | alt | GPN-MSA score |

    pos_list = []
    ref_list = []
    alt_list = []
    scores_list = []
    for line in tb.fetch(chrom_key, start, end):
        fields = line.strip().split("\t") 
        pos_list.append( int(fields[1]) )
        ref_list.append( fields[2] )
        alt_list.append( fields[3] )
        scores_list.append( float(fields[4]) )
    
    df = pd.DataFrame({"Position": pos_list, "Ref": ref_list, "Alt": alt_list, "scores": scores_list})
    return df



motif_map = {
    "MPRABase": ["TERT", "HBG1", "LDLR", "F9", "GP1BA", "IRF4", "IRF6", "PKLR", "ZFAND3", "SORT1",
                 "HBB", "UC88", "MYC_rs6983267", "RET", "TCF7L2"] 
    }

datasets = ["MPRABase"] 
output_root = "./Preds/D05_mprabase/analysis_gpnmsa"
os.makedirs(output_root, exist_ok=True)

for dataset in datasets:
    print(f"Processing dataset: {dataset}")    
    motif_list = motif_map[dataset]
    
    for motif in motif_list:
        print(f"Processing motif/background: {motif}")
        df_path = f"./Preds/D05_mprabase/point_{dataset}_{motif}_saturation.tsv"
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
        
        scores_df = Generate_model_scores(chrom, start, end)
        scores_lookup = scores_df.set_index(['Position', 'Ref', 'Alt'])['scores']
        scores_list = df.apply(lambda row: scores_lookup.get((row['Position'], row['Ref'], row['Alt'])), axis=1)
        
        output_path = f"{output_root}/gpnmsa_variant_scores_{motif}.csv"
        pd.DataFrame({
            "scores": scores_list,
            "variant_effects": variant_effects
        }).to_csv(output_path, index=False)