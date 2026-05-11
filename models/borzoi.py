import shutil
import os, sys

import numpy as np
import pandas as pd

import json
import tensorflow as tf

from .utils import *

import pysam
from kipoiseq.transforms import ReorderedOneHot
from libs.borzoi.seqnn import SeqNN
from libs.borzoi.dataset import targets_prep_strand
from libs.borzoi.gene import Transcriptome
from valids.Borzoi.plot import plot_coverage_track_pair_bins

## Digital_Platform_lightning


class Borzoi():
    def __init__(self, 
                 params_file = "./checks/Borzoi/borzoi_params_pred.json",
                 targets_file = "./checks/Borzoi/borzoi_target_human.txt",
                 model_dir = "./checks/Borzoi/", n_reps = 1
                 ):
        
        ## params processing
        with open(params_file) as params_open:
            params = json.load(params_open)
            params_model = params['model']
            params_train = params['train']
        
        self.targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')
        target_index = self.targets_df.index # output channels
        strand_pair = self.targets_df.strand_pair
        target_slice_dict = {ix : i for i, ix in enumerate(target_index.values.tolist())}
        slice_pair = np.array([
            target_slice_dict[ix] if ix in target_slice_dict else ix for ix in strand_pair.values.tolist()
            ], dtype='int32') # output strand index
        
        
        n_reps = 1
        self.models = []
        for rep_ix in range(n_reps) :
            model_file = f"{model_dir}f3c" + str(rep_ix) + "/model0_best.h5"
            self.seqnn_model = SeqNN(params_model)
            self.seqnn_model.restore(model_file, 0)
            self.seqnn_model.build_slice(target_index)
            self.seqnn_model.strand_pair.append(slice_pair)
            self.seqnn_model.build_ensemble(True, [0])
            self.models.append(self.seqnn_model)
        
        self.input_len = 524288 
    
    ## only allow for single-sample prediction
    def encode(self, seqs):
        # nt_map = ['A','C','G','T']
        seq_len = len(seqs[0])
        func_oh = ReorderedOneHot("ACGT", neutral_value=0)
        seqs_encode = torch.tensor([func_oh(seq) for seq in seqs]) # (524288, 4)
        seqs_encode = seqs_encode.permute(0, 2, 1)
        
        # cropping steps, if length < 524288, pad to 524288; else, cut to 524288
        seqs_encode = padding_and_cropping(seqs_encode, seq_len, self.input_len, "Borzoi")
        seqs_encode = seqs_encode.permute(0, 2, 1)
        seqs_encode = seqs_encode.numpy().astype("float32")
        return seqs_encode
        
    def predict(self, seqs):
        seqs_encode = self.encode(seqs)
        predicted_tracks_all = []
        
        # Loop over samples
        for enc in tqdm.tqdm(seqs_encode): 
            # (524288, 4)
            
            # Loop over model replicates
            predicted_tracks = []
            for rep_ix in range(len(self.models)):
                with tf.device('/CPU:0'):
                    yh = self.models[rep_ix](enc[None, ...])[:, None, ...].astype("float16")
                    predicted_tracks.append(yh)
        
            # Concatenate across replicates
            predicted_tracks = np.concatenate(predicted_tracks, axis=1) 
            predicted_tracks_all.append(predicted_tracks)
            # [1, 1, 16352, 7611]
            
        # Concatenate across samples
        predicted_tracks_all = np.concatenate(predicted_tracks_all, axis=0) # [batch_size, reps, out_len, tracks]
        return predicted_tracks_all, None
    
    def quick_valid(self):
        
        self.__init__(targets_file = "./checks/Borzoi/borzoi_target_gtex.txt")
        
        # Load GTF (optional; needed to compute exon coverage attributions for example gene)
        fasta_open = pysam.Fastafile('./valids/Borzoi/chr10.fa')
        transcriptome = Transcriptome('./valids/Borzoi/gencode41_basic_nort.gtf')

        seq_len = 524288
        search_gene = 'ENSG00000187164'
        center_pos = 116952944
        chrom = 'chr10'
        poses = [116952944]
        alts = ['C']

        start = center_pos - seq_len // 2
        end = center_pos + seq_len // 2
        
        # Get exon bin range
        gene_keys = [gene_key for gene_key in transcriptome.genes.keys() if search_gene in gene_key]
        gene = transcriptome.genes[gene_keys[0]]
        
        # Determine output sequence start, and the output positions of gene exons
        seq_out_start = start + self.seqnn_model.model_strides[0] * self.seqnn_model.target_crops[0] # 116691312
        seq_out_len = self.seqnn_model.model_strides[0] * self.seqnn_model.target_lengths[0] # 523264
        gene_slice = gene.output_slice(seq_out_start, seq_out_len, self.seqnn_model.model_strides[0], False)
        
        # Print index of GTEx blood and muscle tracks in targets file
        self.targets_df['local_index'] = np.arange(len(self.targets_df))
        print("blood tracks = " + str(self.targets_df.loc[self.targets_df['description'] == 'RNA:blood']['local_index'].tolist())) # [9, 10, 11]
        print("muscle tracks = " + str(self.targets_df.loc[self.targets_df['description'] == 'RNA:muscle']['local_index'].tolist())) # [47, 48, 49]

        # Predict for chr10_116952944_T_C
        save_figs = True
        save_suffix = '_chr10_116952944_T_C'

        # Loading testing sequences
        seq_dna = fasta_open.fetch(chrom, start, end)
        sequence_wt = seq_dna.upper()
        
        sequence_mut = list(sequence_wt)
        for pos, alt in zip(poses, alts):
            sequence_mut[pos-start-1] = alt
        sequence_mut = "".join(sequence_mut)
        
        # Make predictions
        [y_wt, y_mut], _ = self.predict([sequence_wt, sequence_mut])
        y_wt, y_mut = y_wt[None, ...], y_mut[None, ...]
        
        # Visualize quantized tracks over SNP
        
        plot_window = 131072
        bin_size = 32
        pad = 16

        untransform_old = True
        normalize_counts = False
        anno_df = None # splice_df

        track_indices = [
            np.arange(0, 89).tolist(),
            [9, 10, 11],
            [47, 48, 49],
        ]

        track_names = [
            'GTEx Coverage (All tissues)',
            'GTEx Coverage (Blood)',
            'GTEx Coverage (Muscle)',
        ]

        track_scales = [0.01]*3
        track_transforms = [3./4.]*3
        soft_clips = [384.]*3

        print("-- Counts --")
        plot_coverage_track_pair_bins(
            y_wt,
            y_mut,
            chrom,
            start,
            center_pos,
            poses,
            track_indices,
            track_names,
            track_scales,
            track_transforms,
            soft_clips,
            plot_window=plot_window,
            normalize_window=1 * plot_window,
            bin_size=bin_size,
            pad=pad,
            normalize_counts=normalize_counts,
            save_figs=save_figs,
            save_suffix=save_suffix,
            gene_slice=gene_slice,
            anno_df=anno_df,
            untransform_old=untransform_old,
            plot_dir="./valids/Borzoi/"
        )
        
        print("Checking results in https://github.com/calico/borzoi/blob/main/examples/borzoi_example_eqtl_chr10_116952944_T_C.ipynb for comparisons.")
        
    
    def predict_bash(self, seqs, csv_features, mode="total"):
        
        seqs_encode = self.encode(seqs)
        total_samples = len(seqs_encode)
        batch_size = 8
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        predicted_tracks_all = []
        
        # Loop over samples
        # for enc in tqdm.tqdm(seqs_encode): 
        for batch_idx in tqdm.tqdm(range(num_batches)):
            
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_samples)
            batch_enc = seqs_encode[start_idx:end_idx]

            predicted_tracks = []
            for rep_ix in range(len(self.models)):
                with tf.device('/CPU:0'):
                    yh = self.models[rep_ix](batch_enc)[:, None, ...].astype("float16") # (4, 1, 16352, 7611)
                    yh = torch.tensor(yh)
                    
                    if mode == "center":
                        out_len = yh.shape[2]
                        mid_start = (out_len - 1) // 2
                        mid_end = mid_start + 2 - (out_len % 2)
                        yh = yh[:, :, mid_start:mid_end, :].mean(dim=2)

                    predicted_tracks.append(yh.tolist())
        
                # Concatenate across replicates
                predicted_tracks = np.concatenate(predicted_tracks, axis=1) 
                predicted_tracks_all.append(predicted_tracks)
            
        # Concatenate across samples
        # [batch_size, reps, out_len, tracks] / [batch_size, reps, tracks]
        
        predicted_tracks_all = np.concatenate(predicted_tracks_all, axis=0)
        pred_list = torch.tensor(predicted_tracks_all)

        if mode == "center":
            pred_list = pred_list.mean(dim=1)
        
        anno_df = pd.read_csv(csv_features)
        anno_df = anno_df[anno_df["model"] == "Borzoi"]
        anno_df = anno_df.drop('Unnamed: 0', axis=1).reset_index()
        return pred_list, anno_df