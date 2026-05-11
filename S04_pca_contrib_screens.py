'''
Plotting the contribution of features to DeepACE's main PC

/home/hyu/Digital_Platform/manuals/fig1a_pca_heatmap.py
/home/hyu/Digital_Platform/manuals/fig1a_pca_analysis.py
mv /home/hyu/Digital_Platform/manuals/fig1a_pca_analysis/PCA_features_filtered.csv /home/hyu/DeepACE/Preds/D01_screens
'''

import os, sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap


INPUT_FILE ="./Preds/D01_screens/PCA_features_filtered.csv"
OUTPUT_DIR = "./Supps/S04_pca_contrib_screens/"
OUTPUT_HEATMAP_PNG = os.path.join( OUTPUT_DIR, "final_optimized_PCA_epigenetic_heatmap.png" )
OUTPUT_HEATMAP_PDF = os.path.join( OUTPUT_DIR, "final_optimized_PCA_epigenetic_heatmap.pdf" )
OUTPUT_STATS = os.path.join( OUTPUT_DIR, "final_PCA_epigenetic_statistics.csv" )
PC_LIST = [f'PC{i}' for i in range(1, 11)]
EPI_TYPE_ORDER = [
    'CAGE/PRO-cap',
    'Transcription factor',
    'Histone activation',
    'Histone repression',
    'RNA polymerase',
    'Others',
    'Chromatin accessibility',
    'Other'
]
TF_KEYWORDS = ['ctcf', 'rad21', 'myc', 'rbfox2', 'cebpb', 'spi1', 'runx3',
               'gabpb1', 'hnf4a', 'irf4', 'tcf12', 'ebf1', 'bcl11a', 'nfic',
               'sox5', 'smad4', 'nfil3', 'foxa3', 'sox13', 'rara', 'gatad2a',
               'sap130', 'nr2f6', 'cebpg']
ACTIVE_HISTONE_KEYWORDS = ['h3k27ac', 'h3k4me3', 'h3k4me2', 'h3k4me1',
                           'h3k9ac', 'h4k91ac', 'h2afz', 'h2bk120ac']
REP_HISTONE_KEYWORDS = ['h3k27me3', 'h3k9me3', 'ezh2']
POL_KEYWORDS = ['polr2a', 'rnapii']

def load_and_preprocess_data(input_path):
    df = pd.read_csv(input_path)
    print(f"Data loaded, shape: {df.shape}")
    df_pc = df[df['PC'].isin(PC_LIST)].copy()
    df_pc['PC'] = pd.Categorical(df_pc['PC'], categories=PC_LIST, ordered=True)
    return df_pc

def assign_detailed_epi_type(feature):
    feature_lower = feature.lower()
    if any(keyword.lower() in feature_lower for keyword in ['cage', 'pro-cap']):
        return 'CAGE/PRO-cap'
    elif 'chip-seq:' in feature_lower or 'chip:' in feature_lower:
        if any(tf in feature_lower for tf in TF_KEYWORDS):
            return 'Transcription factor'
        elif any(act in feature_lower for act in ACTIVE_HISTONE_KEYWORDS):
            return 'Histone activation'
        elif any(rep in feature_lower for rep in REP_HISTONE_KEYWORDS):
            return 'Histone repression'
        elif any(pol in feature_lower for pol in POL_KEYWORDS):
            return 'RNA polymerase'
        else:
            return 'CHIP-seq Others'
    elif 'atac-seq' in feature_lower:
        return 'Chromatin accessibility'
    else:
        return 'Other'

def process_epigenetic_types(df):
    df['detailed_epi_type'] = df['feature'].apply(assign_detailed_epi_type)
    df['detailed_epi_type'] = pd.Categorical(df['detailed_epi_type'],
                                             categories=EPI_TYPE_ORDER, ordered=True)
    epi_counts = df['detailed_epi_type'].value_counts()
    epi_ratio = (epi_counts / len(df) * 100).round(2)
    print("\n=== Epigenetic type statistics ===")
    print(epi_counts)
    print("\nPercentage of total (%):")
    for epi_type, ratio in epi_ratio.items():
        print(f"{epi_type}: {ratio}%")
    return df, epi_counts, epi_ratio

def calculate_pc_epi_ratio(df):
    pc_epi_cross = pd.crosstab(df['PC'], df['detailed_epi_type'])
    pc_epi_ratio = pc_epi_cross.div(pc_epi_cross.sum(axis=1), axis=0)
    pc_epi_cross = pc_epi_cross.reindex(PC_LIST)
    pc_epi_ratio = pc_epi_ratio.reindex(PC_LIST)
    print("\n=== PC-wise epigenetic type ratios ===")
    print(pc_epi_ratio.round(4))
    return pc_epi_cross, pc_epi_ratio

def draw_final_optimized_heatmap(pc_epi_ratio, output_png_path, output_pdf_path):
    plt.figure(figsize=(10, 7), dpi=400)
    sns.set(style='white', font_scale=1.2)
    custom_cmap = LinearSegmentedColormap.from_list(
        "GreenWhiteYellow",
        ["#74a892", "#ffffff", "#e5c185"]
    )
    ax = sns.heatmap(
        pc_epi_ratio,
        annot=True, fmt=".2f",
        cmap=custom_cmap,
        cbar_kws={'label': 'Proportion', 'shrink': 0.8},
        linewidths=0.5, linecolor='white',
        annot_kws={"fontsize":14, "weight":"medium"}
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=14, weight='medium')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf_path, dpi=400, bbox_inches='tight', format='pdf', facecolor='white')
    plt.close()
    print(f"\nHeatmap saved as PNG: {output_png_path}")
    print(f"Heatmap saved as PDF: {output_pdf_path}")

def generate_statistics_file(pc_epi_cross, pc_epi_ratio, output_stats_path):
    stats_df = pd.DataFrame({'PC': pc_epi_cross.index})
    for epi_type in pc_epi_cross.columns:
        stats_df[f'{epi_type} - Count'] = pc_epi_cross[epi_type].values
        stats_df[f'{epi_type} - Ratio'] = pc_epi_ratio[epi_type].values.round(4)
    stats_df.to_csv(output_stats_path, index=False, encoding='utf-8-sig')
    print(f"Statistics file saved at: {output_stats_path}")
    return stats_df


''' Dataset Preparation '''

# df = pd.read_csv("./Datas/D01_screens/GRCh38-cCREs.bed", sep="\t", header=None)
# df = df.sample(n=10000, random_state=19)
# genome_path = "./Datas/D02_grch/GRCh38.primary_assembly.genome.fa"
# genome = Fasta(genome_path)
# df_samples = []
# for _, row in df.iterrows():
#     chrom = row.iloc[0]
#     start = int(row.iloc[1])
#     end = int(row.iloc[2])
#     source = row.iloc[-1] 
#     length = end - start
#     if length < 600:
#         center = (start + end) // 2
#         half = 300
#         start = max(0, center - half)
#         end = start + 600
#     else:
#         center = (start + end) // 2
#         start = center - 300
#         end = center + 300
#     seq = genome[chrom][start:end].seq.upper()
#     df_samples.append((chrom, start, end, source, seq))

## saving for 5 iterations
# df_res = pd.DataFrame(df_samples, columns=["chrom", "start", "end", "source", "sequence"])
# df_res.to_csv("./Preds/D01_screens/CRE_samples_1.tsv", sep="\t", index=False)
# seqs = df_res["sequence"].tolist()
# output_file = f"./Preds/D01_screens/CRE_samples_1.fasta"
# with open(output_file, "w") as f:
#     for i, seq in enumerate(seqs):
#         f.write(f">sequence_{i}\n{seq}\n")


# uni_pred = []
# for run_id in range(5):
#     run_pred = np.load(f"./Preds/D01_screens/SCREEN_samples_{run_id}/uni_pred.npy")  # (10000, 43275) 
#     uni_pred += list(run_pred)
# uni_pred = np.array(uni_pred) # (50000, 43275) 
# uni_anno = pd.read_csv(f"./total_features.csv")  # (43275, 5)
# pca = PCA(n_components=50, random_state=42)
# uni_selected = pca.fit_transform(uni_pred)  # (50000, 50)
# loadings = pca.components_  # shape = (50, 43275)
# explained_ratio = pca.explained_variance_ratio_
# principal_features = []
# for i in range(50):
#     comp = loadings[i, :]
#     top_idx = np.where(np.abs(comp) > 0.075)[0]
#     if len(top_idx) == 0:
#         continue  
#     top_features = uni_anno.iloc[top_idx].copy()
#     top_features["loading_value"] = comp[top_idx]
#     top_features["PC"] = f"PC{i+1}"
#     top_features["explained_var_ratio"] = explained_ratio[i]
#     principal_features.append(top_features)
# for i in range(50):
#     comp = loadings[i, :]
#     top_idx = np.argsort(np.abs(comp))[-100:]
#     top_features = uni_anno.iloc[top_idx].copy()
#     top_features["loading_value"] = comp[top_idx]
#     top_features["PC"] = f"PC{i+1}"
#     top_features["explained_var_ratio"] = explained_ratio[i]
#     principal_features.append(top_features)
# principal_df = pd.concat(principal_features, ignore_index=True)
# print(principal_df.head(20))
# principal_df.to_csv("./Preds/D01_screens/PCA_features_filtered.csv", index=False)




''' Heatmap Analysis '''
df_pc_data = load_and_preprocess_data(INPUT_FILE)
df_pc_data, epi_counts, epi_ratio = process_epigenetic_types(df_pc_data)
pc_epi_cross, pc_epi_ratio = calculate_pc_epi_ratio(df_pc_data)
draw_final_optimized_heatmap(pc_epi_ratio, OUTPUT_HEATMAP_PNG, OUTPUT_HEATMAP_PDF)
stats_df = generate_statistics_file(pc_epi_cross, pc_epi_ratio, OUTPUT_STATS)

print("\n="*60)
print("Workflow completed successfully!")
print(f"Heatmap PNG: {OUTPUT_HEATMAP_PNG}")
print(f"Heatmap PDF: {OUTPUT_HEATMAP_PDF}")
print(f"Statistics CSV: {OUTPUT_STATS}")
print("="*60)