'''
/home/hyu/Figures/DeepACE/Fig2.py
/home/hyu/Digital_Platform/manuals/fig2f_point_mutation_evo2.py

cp /home/hyu/Digital_Platform/manuals/fig2f_point_mutation_evo2/MPRABase_evo2/evo2_variant_* /home/hyu/DeepACE/Preds/D05_mprabase/analysis_evo2
'''

import base64
import io
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import tensorflow as tf
import torch
from typing import List, Union
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = ""
tf.config.threading.set_intra_op_parallelism_threads(32)
tf.config.threading.set_inter_op_parallelism_threads(32)

random.seed(42)
np.random.seed(42)

class CharLevelTokenizer:
    """Character Level Tokenizer"""

    def __init__(self, vocab_size):
        self.name = "CharLevelTokenizer"
        self._vocab_size = vocab_size
        self.eod_id = 0
        self.eos_id = 0
        self.pad_id = 1

    def clamp(self, n):
        return max(32, min(n, self.vocab_size))

    @property
    def vocab_size(self):
        return self._vocab_size

    def decode_token(self, token: int):
        return chr(self.clamp(token))

    def tokenize(self, text: str):
        return list(np.fromstring(text, dtype=np.uint8))

    def tokenize_batch(self, text_batch: Union[List[str], str]):
        if isinstance(text_batch, list):
            return [self.tokenize(s) for s in text_batch]
        else:
            return self.tokenize(text_batch)

    def detokenize(self, token_ids):
        return "".join(map(self.decode_token, token_ids))

    def detokenize_batch(self, token_ids: Union[List[str], torch.Tensor, str]):
        if isinstance(token_ids, list):
            return [self.detokenize(s) for s in token_ids]
        elif isinstance(token_ids, torch.Tensor):
            return [self.detokenize(s) for s in token_ids.tolist()]
        else:
            return self.detokenize(token_ids)

    @property
    def eod(self):
        return self.eod_id

    @property
    def eos(self):
        return self.eod_id
    
EVO2_URL = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward"

class CharLevelTokenizer:
    """Character Level Tokenizer"""

    def __init__(self, vocab_size):
        self.name = "CharLevelTokenizer"
        self._vocab_size = vocab_size
        self.eod_id = 0
        self.eos_id = 0
        self.pad_id = 1

    def clamp(self, n):
        return max(32, min(n, self.vocab_size))

    @property
    def vocab_size(self):
        return self._vocab_size

    def decode_token(self, token: int):
        return chr(self.clamp(token))

    def tokenize(self, text: str):
        return list(np.fromstring(text, dtype=np.uint8))

    def tokenize_batch(self, text_batch: Union[List[str], str]):
        if isinstance(text_batch, list):
            return [self.tokenize(s) for s in text_batch]
        else:
            return self.tokenize(text_batch)

    def detokenize(self, token_ids):
        return "".join(map(self.decode_token, token_ids))

    def detokenize_batch(self, token_ids: Union[List[str], torch.Tensor, str]):
        if isinstance(token_ids, list):
            return [self.detokenize(s) for s in token_ids]
        elif isinstance(token_ids, torch.Tensor):
            return [self.detokenize(s) for s in token_ids.tolist()]
        else:
            return self.detokenize(token_ids)

    @property
    def eod(self):
        return self.eod_id

    @property
    def eos(self):
        return self.eod_id
    
#--------------------------------------------------------------------------------#

EVO2_URL = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward"


############################################
# 1) Single API call (with automatic retry)
############################################
def evo2_single(seq, key, max_retry=5):
    for attempt in range(max_retry):
        try:
            r = requests.post(
                url=EVO2_URL,
                headers={"Authorization": f"Bearer {key}"},
                json={"sequence": seq, "output_layers": ["unembed"]},
                timeout=60,
            )

            # zip response format
            if "application/zip" in r.headers.get("Content-Type", ""):
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    file_list = z.namelist()
                    data = z.read(file_list[0])
                    data = json.loads(data.decode())
                    decoded = base64.b64decode(data["data"])
                    embeddings = np.load(io.BytesIO(decoded))["unembed.output"]
                return embeddings

            # json response format
            if "application/json" in r.headers.get("Content-Type", ""):
                data = json.loads(r.text)
                decoded = base64.b64decode(data["data"])
                embeddings = np.load(io.BytesIO(decoded))["unembed.output"]
                return embeddings

            print(f"Abnormal response format: {r.status_code} {r.headers}")
            time.sleep(1)

        except Exception as e:
            print(f"[Retry {attempt+1}] API call failed: {e}")
            time.sleep(1)

    # Final failure
    raise RuntimeError(f"Evo2 call failed (after {max_retry} retries)")


############################################
# 2) logprob calculation (keeping original logic)
############################################
def compute_logprob_from_embeddings(embeddings, seq):
    tokenizer = CharLevelTokenizer(512)

    input_ids = torch.tensor(tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0)
    input_ids = input_ids[:, 1:].type(torch.int64)

    logits = torch.tensor(embeddings)
    softmax_logprobs = torch.log_softmax(logits, dim=-1)
    softmax_logprobs = softmax_logprobs[:, :-1]

    logprobs = torch.gather(
        softmax_logprobs,
        2,
        input_ids.unsqueeze(-1)
    ).squeeze(-1)

    return float(np.mean(logprobs.cpu().numpy()))


############################################
# 3) Multi-threaded call (safe without None)
############################################
def evo2_batch_parallel(seqs, key, max_workers=10):
    results = [None] * len(seqs)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(evo2_single, seqs[i], key): i for i in range(len(seqs))}

        for fut in tqdm(as_completed(futures), total=len(seqs)):
            i = futures[fut]
            try:
                emb = fut.result()          
                results[i] = compute_logprob_from_embeddings(emb, seqs[i])
            except Exception as e:
                print(f"[Index {i}] Complete failure: {e}")
                results[i] = np.nan         
    return results


motif_map = {
    "MPRABase": ["TERT", "HBG1", "LDLR", "F9", "GP1BA", "IRF4", "IRF6", "PKLR", "ZFAND3", "SORT1",
                 "HBB", "HNF4A", "ZRS", "UC88", "MSMB", "MYC_rs6983267", "RET", "TCF7L2"] 
}
key = "your-key"
datasets = ["MPRABase"]
motif_map = {
    "MPRABase": ["HBB", "HNF4A", "ZRS", "UC88", "MSMB", "MYC_rs6983267", "RET", "TCF7L2"] # 
}

output_root = "./Preds/D05_mprabase/analysis_evo2"
os.makedirs(output_root, exist_ok=True)

for dataset in datasets:
    print(f"Processing dataset: {dataset}")
    output_dir = f"{output_root}/{dataset}_evo2"
    os.makedirs(output_dir, exist_ok=True)
    motif_list = motif_map[dataset]

    for motif in motif_list:
        print(f"\n======================= Processing motif: {motif} =======================\n")
        df_path = f"./Preds/D05_mprabase/point_{dataset}_{motif}_saturation.tsv"
        df = pd.read_csv(df_path, sep="\t")

        seqs_alt = df["alt_seq"].tolist()
        seqs_ref = df["ref_seq"].tolist()
        variant_effects = df['VariantExpressionEffect (log2)'].to_numpy()

        ###################
        # 计算 ref score
        ###################
        r = requests.post(
            url=EVO2_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={"sequence": seqs_ref[0], "output_layers": ['unembed']}
        )

        tokenizer = CharLevelTokenizer(512)
        input_ids = torch.tensor(tokenizer.tokenize(seqs_ref[0]), dtype=torch.int).unsqueeze(0)
        input_ids = input_ids[:, 1:].type(torch.int64)

        if "application/json" in r.headers.get("Content-Type", ""):
            data = json.loads(r.text)
            decoded = base64.b64decode(data['data'])
            embeddings = np.load(io.BytesIO(decoded))['unembed.output']

        elif "application/zip" in r.headers.get("Content-Type", ""):
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                file_list = z.namelist()
                data = z.read(file_list[0])
                data = json.loads(data.decode())
                decoded = base64.b64decode(data['data'])
                embeddings = np.load(io.BytesIO(decoded))['unembed.output']

        logits = torch.tensor(embeddings)
        softmax_logprobs = torch.log_softmax(logits, dim=-1)
        softmax_logprobs = softmax_logprobs[:, :-1]

        logprobs = torch.gather(
            softmax_logprobs,
            2,
            input_ids.unsqueeze(-1)
        ).squeeze(-1)
        score_ref = np.mean(logprobs.cpu().numpy())

        print("Running Evo2 for ALT sequences with multithreading (10 workers)...")
        score_alt_list = evo2_batch_parallel(seqs_alt, key, max_workers=10)
        score_ref_list = np.array([score_ref] * len(seqs_alt))
        score_alt_list = np.array(score_alt_list)
        score_list = score_alt_list - score_ref_list

        df_out = pd.DataFrame({
            "scores": score_list,
            "variant_effects": variant_effects,
            "score_alt": score_alt_list,
            "score_ref": score_ref_list
        })
        df_out.to_csv(f"{output_dir}/evo2_variant_scores_{motif}.csv", index=False)

        print(f"Saved: {output_dir}/evo2_variant_scores_{motif}.csv")