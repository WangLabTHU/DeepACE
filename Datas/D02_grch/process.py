import glob
import pandas as pd
import seaborn as sns
import numpy as np
import sys, os
from tqdm import tqdm

from models.enformer import Enformer # (Digital_Platform_transformers)
from utils import SequenceParser, get_summary

"""
step1: loading datasets and estimate_TSS_activity
"""

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
fasta_path = "../Datas/D02_grch/GRCh38.primary_assembly.genome.fa"
tss_csv_path = "./tss_positions.csv"
tss_df = pd.read_csv(tss_csv_path, index_col=None)
tss_df = tss_df.sample(frac = 1)

seq_parser = SequenceParser(fasta_path)
N = tss_df.shape[0]
print(N)

model_name = "Enformer"
if model_name.lower() == 'enformer':
    track_index = [4824, 5110, 5111]
    bin_index = [447, 448]
    dp_model = Enformer()
    seq_len = 196608

elif model_name.lower() == 'borzoi':
    target_df = pd.read_csv('.../checks/Borzoi/borzoi_target_human.txt', sep='\t')
    cage_tracks = [i for i, t in enumerate(target_df['description']) if
                   ('CAGE' in t) and (t.split(':')[-1].strip() in ['K562 ENCODE, biol_',
                                                                   'GM12878 ENCODE, biol_',
                                                                   'PC-3'])]
    bin_index = list(np.arange(16352 // 2 - 4, 16352 // 2 + 4, 1))
    dp_model = Borzoi()
    seq_len = 524288

save_dir = ".../Preds/D03_creme/gencode_tss_predictions"
tss_df = tss_df.sort_values(by=['Chromosome', 'Start']).reset_index(drop=True)

for j, (i, row) in tqdm(enumerate(tss_df.iterrows()), total=N):
        chrom, start = row[:2]
        strand = row['Strand']
        assert j < N, 'bad index'
        
        save_path = f"{save_dir}/{get_summary(row)}"
        
        sequence = seq_parser.extract_seq_centered(chrom, start, strand, seq_len, onehot=False)
        
        if model_name == 'Enformer':
            pred, _ = dp_model.predict([sequence])
            pred = pred.numpy().squeeze()[bin_index, :][:, track_index]
            np.save(f'{save_path}.npy', pred)



"""
step2: selecting top-predicting sequences in enformer
"""


model_name = "Enformer"
cell_lines = [4824, 5110, 5111]
bin_index = [447, 448]
target_df = pd.read_csv('.../libs/enformer/targets_human.txt', sep='\t')
column_names = [t.split(':')[-1].split(' ENCODE')[0].strip() for t in target_df.iloc[cell_lines]['description'].values]
print(column_names)

tss_csv_path = "./tss_positions.csv"
tss_df = pd.read_csv(tss_csv_path, index_col=None)
N = tss_df.shape[0]

save_dir = ".../Preds/D03_creme/gencode_tss_predictions"
all_tss = np.empty((N, len(cell_lines)))
for i, row in tqdm(tss_df.iterrows(), total=N):
    pred = np.load(f'{save_dir}/{get_summary(row)}.npy')
    all_tss[i] = pred.mean(axis=0)
save_dir = "./"
np.save(f'{save_dir}/summary_cage.npy', all_tss)

# selecting top 10000 sequences -> top 100 sequences
save_dir = ".../Preds/D03_creme"
for i in range(len(cell_lines)):
    save_path = f'{save_dir}/{cell_lines[i]}_{column_names[i]}_selected_genes.csv'
    cell_line_df = tss_df.copy()
    cell_line_df[column_names[i]] = all_tss[:, i]
    max_tss_set = cell_line_df.sort_values(column_names[i], ascending=False).drop_duplicates(['gene_name'])
    max_tss_set = max_tss_set.sort_values(column_names[i]).iloc[-100:]
    max_tss_set.to_csv(save_path)




"""
step3: selecting the fasta file
"""

def open_fna_file(fna_file):
    with open(fna_file, 'r') as f:
        lines = f.readlines()
    
    records, tags = [], []
    
    tmp = []
    for line in lines:
        if '>' not in line:
            tmp.append(line[0:-1])
        else:
            tags.append(line[1:-1])
            records.append("".join(tmp))
            tmp = []
    records.append("".join(tmp))
    records = records[1:]
    
    return records, tags


def write_fasta_file(file_path, seqs, tags):
    f = open(file_path,'w')
    i = 0
    while i < len(seqs):
        f.write('>' + tags[i] + '\n')
        f.write(seqs[i] + '\n')
        i = i + 1
    f.close()


save_dir = ".../Preds/D03_creme/gencode_tss_summary"
df_list = ["4824_PC-3_selected_genes", "5110_GM12878_selected_genes", "5111_K562_selected_genes"]

fasta_path = "../Datas/D02_grch/GRCh38.primary_assembly.genome.fa"
seq_parser = SequenceParser(fasta_path)

for cell_line in df_list:
    df_summary = pd.read_csv(f"{save_dir}/{cell_line}.csv")
    seqs_list, tags_list = [], []
    for i, row in tqdm(df_summary.iterrows()):
        chrom = row["Chromosome"]
        start = row["Start"]
        strand = row["Strand"]
        gene_name = row["gene_name"]
        gene_id = row["gene_id"]
        seq_len = 196608
        
        sequence = seq_parser.extract_seq_centered(chrom, start, strand, seq_len, onehot=False)
        seqs_list.append(sequence)
        tags_list.append(f"{chrom}_{gene_name}_{gene_id}_{strand}")
    
    write_fasta_file(f"{save_dir}/{cell_line}.fa", seqs_list, tags_list)


"""
creme.context_dependence_test: half_window_size = 5000 // 2
https://github.com/p-koo/creme-nn/blob/4f36864c5feece6fc25a2c8520892a9f2555191c/creme/creme.py#L11
"""

