'''

/home/hyu/Digital_Platform/manuals/fig2f_point_mutation_promoterAI.py
mv /home/hyu/Digital_Platform/manuals/fig_dataset/checks_promoterAI/promoterAI_v1_hg38 /home/hyu/DeepACE/Datas/D11_promoterAI/
cp /home/hyu/Digital_Platform/manuals/fig2f_point_mutation_promoterAI/MPRABase_promoterAI/promoterAI_variant* /home/hyu/DeepACE/Preds/D05_mprabase/analysis_promoterai/
'''

import os
import random
import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from pyfaidx import Fasta
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import tensorflow.keras as tk
import tensorflow.keras.initializers as tki
import tensorflow.keras.layers as tkl
from tensorflow.keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = ""
tf.config.threading.set_intra_op_parallelism_threads(32)
tf.config.threading.set_inter_op_parallelism_threads(32)

random.seed(42)
np.random.seed(42)


def _get_human_output(outputs):
    if isinstance(outputs, tuple) or isinstance(outputs, list):
        return outputs[0]
    elif isinstance(outputs, dict):
        return outputs.get('human', outputs[list(outputs.keys())[0]])
    else:
        return outputs

def twin_wrap(model):
    for layer in model.layers:
        layer.trainable = 'output0' in layer.name
    input_ref = tk.Input(shape=model.input_shape[1:])
    input_alt = tk.Input(shape=model.input_shape[1:])
    output_ref = _get_human_output(model(input_ref))
    output_alt = _get_human_output(model(input_alt))
    output_ = tkl.Subtract()([output_alt, output_ref])
    output_ = tkl.Lambda(lambda x: tk.backend.mean(x, axis=(1, 2)))(output_)
    return tk.Model(inputs=(input_ref, input_alt), outputs=output_)

def _onehot_encode(seq: str, input_length: int):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.zeros((input_length, 4), dtype=np.float32)
    for i, s in enumerate(seq):
        if s in mapping:
            arr[i, mapping[s]] = 1
    return arr

def find_mutation_position(ref_seq: str, alt_seq: str):
    min_len = min(len(ref_seq), len(alt_seq))
    for i in range(min_len):
        if ref_seq[i] != alt_seq[i]:
            return i
    return min_len

def center_variant(seq: str, mut_pos: int, input_length: int) -> str:
    center = input_length // 2
    pad_left = center - mut_pos
    pad_right = input_length - len(seq) - pad_left
    return "N" * pad_left + seq + "N" * pad_right

class SequencePairGenerator(tk.utils.Sequence):
    def __init__(self, seq_refs, seq_alts, input_length, batch_size=32):
        assert len(seq_refs) == len(seq_alts)
        self.seq_refs = seq_refs
        self.seq_alts = seq_alts
        self.input_length = input_length
        self.batch_size = batch_size
        self.indexes = np.arange(len(seq_refs))
        self.processed_refs = []
        self.processed_alts = []
        for ref, alt in zip(seq_refs, seq_alts):
            pos = find_mutation_position(ref, alt)
            ref_centered = center_variant(ref, pos, input_length)
            alt_centered = center_variant(alt, pos, input_length)
            self.processed_refs.append(ref_centered)
            self.processed_alts.append(alt_centered)

    def __len__(self):
        return int(np.ceil(len(self.seq_refs) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_refs = [self.processed_refs[i] for i in batch_idx]
        batch_alts = [self.processed_alts[i] for i in batch_idx]
        xs_ref = np.zeros((len(batch_idx), self.input_length, 4))
        xs_alt = np.zeros((len(batch_idx), self.input_length, 4))
        ys = np.zeros(len(batch_idx))
        for i, (ref, alt) in enumerate(zip(batch_refs, batch_alts)):
            xs_ref[i] = _onehot_encode(ref, self.input_length)
            xs_alt[i] = _onehot_encode(alt, self.input_length)
        return (xs_ref, xs_alt), ys


def promoterAI_predict(model_folder, seqs_alt, seqs_ref, variant_effects, input_length=20480, output_path=None):
    model_folder = Path(model_folder)
    model = tk.models.load_model(model_folder, compile=False)
    try:
        twin_model = twin_wrap(model)
    except Exception as e:
        raise RuntimeError(f"Unable to wrap model: {e}")
    gen_var = SequencePairGenerator(seqs_ref, seqs_alt, input_length=input_length, batch_size=32)
    print(f"Predicting on {len(seqs_ref)} variants with input_length={input_length}...")
    preds = twin_model.predict(gen_var, verbose=1).flatten()
    print("Raw predictions sample:", preds[:10])
    print("Any NaN?", np.isnan(preds).any())
    scores = np.tanh(np.round(preds, 4))
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame({
            "scores": scores,
            "variant_effects": variant_effects
        }).to_csv(output_path, index=False)
        print(f"Saved scores to {output_path}")
    return scores


def build_ref_alt_sequences(df, win=10240):
    seqs_ref = []
    seqs_alt = []
    center_bases_ref = []
    center_bases_alt = []
    for chrom, pos, ref, alt in zip(df["Chromosome"], df["Position"], df["Ref"], df["Alt"]):
        chrom = str(chrom)
        pos = int(pos)
        if not chrom.lower().startswith("chr"):
            chrom = "chr" + chrom
        start = pos - win
        end   = pos + win - 1
        seq = fa[chrom][start-1:end].seq.upper()
        center = win
        r = seq[:center] + ref.upper()[0] + seq[center+1:]
        a = seq[:center] + alt.upper()[0] + seq[center+1:]
        seqs_ref.append(r)
        seqs_alt.append(a)
        center_bases_ref.append(r[center])
        center_bases_alt.append(a[center])
    return seqs_ref, seqs_alt, center_bases_ref, center_bases_alt


motif_map = {
    "MPRABase": ["TERT", "HBG1", "LDLR", "F9", "GP1BA", "IRF4", "IRF6", "PKLR", "ZFAND3", "SORT1",
                 "HBB", "HNF4A", "ZRS", "UC88", "MSMB", "MYC_rs6983267", "RET", "TCF7L2"] 
}

model = load_model("./Datas/D11_promoterAI/promoterAI_v1_hg38", compile=False)
print("Model input shape:", model.input_shape)
datasets = ["MPRABase"] #   
GENOME = "./Datas/D02_grch/GRCh38.primary_assembly.genome.fa"
fa = Fasta(GENOME, sequence_always_upper=True)

output_dir = f"./Preds/D05_mprabase/analysis_promoterai"
os.makedirs(output_dir, exist_ok=True)

for dataset in datasets:
    print(f"Processing dataset: {dataset}")
    motif_list = motif_map[dataset]
    for motif in motif_list:
        print(f"Processing motif/background: {motif}")
        df_path = f"./Preds/D05_mprabase/point_{dataset}_{motif}_saturation.tsv"
        df = pd.read_csv(df_path, sep="\t")
        variant_effects = df['VariantExpressionEffect (log2)'].to_numpy()
        seqs_ref, seqs_alt, _, _ = build_ref_alt_sequences(df, win=10240)
        promoterAI_predict(
            model_folder = "./Datas/D11_promoterAI/promoterAI_v1_hg38",
            seqs_alt=seqs_alt,
            seqs_ref=seqs_ref,
            variant_effects = variant_effects,
            output_path=f"{output_dir}/promoterAI_variant_scores_{motif}.csv")