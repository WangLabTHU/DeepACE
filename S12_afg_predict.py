from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from alphagenome_pytorch import AlphaGenome

WINDOW_LEN = 131072
BASE2IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
BASE_LUT = np.full(256, -1, dtype=np.int64)
for _b, _i in BASE2IDX.items():
    BASE_LUT[ord(_b)] = _i
    BASE_LUT[ord(_b.lower())] = _i
MPRABASE_DATASETS = [
    ("TERT",          "SF7996",     "MPRABase_TERT"),
    ("HBG1",          "HEL 92.1.7", "MPRABase_HBG1"),
    ("LDLR",          "HepG2",      "MPRABase_LDLR"),
    ("F9",            "HepG2",      "MPRABase_F9"),
    ("GP1BA",         "HEL 92.1.7", "MPRABase_GP1BB"),
    ("IRF4",          "SK-MEL-28",  "MPRABase_IRF4"),
    ("IRF6",          "HaCaT",      "MPRABase_IRF6"),
    ("PKLR",          "K562",       "MPRABase_PKLR"),
    ("ZFAND3",        "MIN6",       "MPRABase_ZFAND3"),
    ("SORT1",         "HepG2",      "MPRABase_SORT1"),
    ("HBB",           "HEL 92.1.7", "MPRABase_HBB"),
    ("UC88",          "Neuro-2a",   "MPRABase_UC88"),
    ("MYC_rs6983267", "Hek293T",    "MPRABase_MYC_rs6983267"),
    ("RET",           "Neuro-2a",   "MPRABase_RET"),
    ("TCF7L2",        "MIN6",       "MPRABase_TCF7L2"),
]

HEADS = ["dnase", "atac"]
ASSAY_OF_HEAD = {"dnase": "DNase-seq", "atac": "ATAC-seq"}
WEIGHT_CANDIDATES = [
    "./checks/AlphaGenome/model_all_folds.safetensors",
]


# ===========================================================================
# Reference geome
# ===========================================================================

def build_genome_index(fasta: str, cache_path: str | None = None) -> dict:
    size = os.path.getsize(fasta)
    mtime = os.path.getmtime(fasta)
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as fh:
                blob = json.load(fh)
            if blob.get("fasta_size") == size and blob.get("fasta_mtime") == mtime:
                print(f"[genome] reusing index cache {cache_path} ({len(blob['records'])} records)")
                return blob["records"]
        except Exception as e:                                    
            print(f"[genome] index cache unusable ({type(e).__name__}), rebuilding")

    t0 = time.time()
    records = {}
    line_bases = None                    
    with open(fasta, "rb") as fh:
        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line:
                break
            if line[:1] == b">":
                name = line[1:].split()[0].decode()
                records[name] = dict(offset=None, length=0, line_bases=None, line_width=0)
                line_bases = None
                continue
            if line_bases is None:
                n_bases = len(line.rstrip(b"\r\n"))
                n_bytes = len(line)
                records[name]["offset"] = pos
                records[name]["line_bases"] = n_bases
                records[name]["line_width"] = n_bytes
                line_bases = n_bases
                records[name]["length"] = n_bases
            else:
                records[name]["length"] += len(line.rstrip(b"\r\n"))
    records = {k: v for k, v in records.items() if v["offset"] is not None}
    print(f"[genome] index built: {len(records)} records in {time.time() - t0:.1f}s")
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        try:
            with open(cache_path, "w") as fh:
                json.dump({"fasta_size": size, "fasta_mtime": mtime, "records": records}, fh)
            print(f"[genome] index cached -> {cache_path}")
        except Exception as e:
            print(f"[genome] failed to write index cache ({type(e).__name__}), ignoring")
    return records


class GenomeReader:
    def __init__(self, fasta: str, index: dict):
        self.fasta = fasta
        self.index = index
        self._fh = None

    def _fh_open(self):
        if self._fh is None:
            self._fh = open(self.fasta, "rb")
        return self._fh

    def fetch(self, chrom: str, start: int, end: int) -> str:
        key = chrom if chrom in self.index else ("chr" + chrom if "chr" + chrom in self.index else None)
        if key is None:
            raise KeyError(f"chromosome {chrom!r} not in the reference genome")
        rec = self.index[key]
        L = rec["length"]
        out = []
        if start < 0:
            out.append("N" * min(-start, end - start))
            start = 0
        if end > L:
            out.append("")                                     
        s, e = max(start, 0), min(end, L)
        if e > s:
            fh = self._fh_open()
            lb, lw, off = rec["line_bases"], rec["line_width"], rec["offset"]
            b0 = off + (s // lb) * lw + (s % lb)
            n_lines = (e - 1) // lb - s // lb + 1
            fh.seek(b0)
            chunk = fh.read(n_lines * lw)
            out.append(chunk.replace(b"\n", b"").replace(b"\r", b"")[: e - s].decode())
        if end > L:
            out.append("N" * (end - L))
        res = "".join(out)
        if len(res) != end - start:
            raise RuntimeError(
                f"{chrom}: fetched {len(res)} bp, expected {end - start} bp "
                f"(record length {L}, line_bases={rec['line_bases']}, line_width={rec['line_width']}) "
                f"-- genome index is wrong")
        return res

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None



# ===========================================================================
# track metadata: cell line -> track_index
# ===========================================================================

def load_channel_tables(meta_dir: str) -> dict:
    tables = {}
    for head in HEADS:
        path = os.path.join(meta_dir, f"{head}_1bp_channels.csv")
        df = pd.read_csv(path)
        df = df[df["track_name"] != "Padding"].reset_index(drop=True)
        tables[head] = df
        n_out = 384 if head == "dnase" else 256
        print(f"[channel] {head}: {len(df)} 条有名通道 / 头输出 {n_out} 列 "
              f"(track_index {int(df['track_index'].min())}..{int(df['track_index'].max())})")
    return tables


def resolve_channel(tables: dict, cell_line: str, head: str) -> int | None:
    df = tables[head]
    if df is None:
        return None
    sub = df[(df["biosample_name"].astype(str) == cell_line) & (df["assay_title"] == ASSAY_OF_HEAD[head])]
    if len(sub) == 0:
        return None
    if len(sub) > 1:
        raise ValueError(f"{cell_line} 在 {head} metadata 里有 {len(sub)} 行, 无法唯一确定通道")
    return int(sub["track_index"].iloc[0])


def load_channel_map_override(path: str) -> dict:
    df = pd.read_csv(path)
    need = {"motif", "head", "track_index"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} 缺少列 {sorted(need - set(df.columns))}")
    out = {}
    for r in df.itertuples():
        out[(str(r.motif), str(r.head))] = int(r.track_index)
    return out


# ===========================================================================
# padding with genome context (same as AlphaGenome settings)
# ===========================================================================

def element_geometry(df: pd.DataFrame, motif: str) -> tuple:
    chroms = df["Chromosome"].astype(str).unique()
    if len(chroms) != 1:
        raise ValueError(f"{motif}: multiple chromosomes in tsv: {list(chroms)}")
    chrom = chroms[0]
    idx_list, starts, alts, refs = [], set(), [], []
    for r in df.itertuples():
        a, b = str(r.ref_seq).upper(), str(r.alt_seq).upper()
        if len(a) != len(b):
            raise ValueError(f"{motif}: ref_seq / alt_seq length mismatch")
        d = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if len(d) != 1:
            raise ValueError(f"{motif}: {len(d)} differing positions between ref_seq and alt_seq (expected exactly 1)")
        i = d[0]
        if a[i] != str(r.Ref).upper() or b[i] != str(r.Alt).upper():
            raise ValueError(f"{motif}: differing base {a[i]}/{b[i]} does not match Ref/Alt {r.Ref}/{r.Alt}")
        idx_list.append(i)
        starts.add(int(r.Position) - 1 - i)
        alts.append(b[i])
        refs.append(a[i])
    if len(starts) != 1:
        raise ValueError(f"{motif}: inconsistent element start inferred across rows: {sorted(starts)[:5]}")
    seq = str(df["ref_seq"].iloc[0]).upper()
    if df["ref_seq"].nunique() != 1:
        raise ValueError(f"{motif}: ref_seq is not the same sequence across rows")
    return chrom, starts.pop(), seq, np.array(idx_list, dtype=np.int64), refs, alts



# ===========================================================================
# forward
# ===========================================================================

def window_to_onehot(window: str) -> np.ndarray:
    idx = BASE_LUT[np.frombuffer(window.encode("ascii"), dtype=np.uint8)]
    arr = np.zeros((len(idx), 4), dtype=np.float32)
    m = idx >= 0
    arr[np.nonzero(m)[0], idx[m]] = 1.0
    return arr


def embed_core_into_window(seq: str, window_len: int = WINDOW_LEN) -> str:
    seq = seq.upper()
    window = ["N"] * window_len
    start = (window_len - len(seq)) // 2
    window[start: start + len(seq)] = list(seq)
    return "".join(window)


def build_genomic_window(reader: GenomeReader, chrom: str, elem_start: int, core_len: int) -> str:
    """Real genomic context; the element sits at the exact same offset as in the N-padded version (off = (W-L)//2)."""
    off = (WINDOW_LEN - core_len) // 2
    return reader.fetch(chrom, elem_start - off, elem_start - off + WINDOW_LEN).upper()


class AfgScorer:

    def __init__(self, weights: str, device: str = "cuda"):
        self.device = torch.device(device)
        print(f"[model] loading {weights} on {self.device} ...")
        self.model = AlphaGenome.from_pretrained(str(weights), device=str(self.device))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.use_amp = self.device.type == "cuda"

    def _forward(self, batch: torch.Tensor):
        org = torch.full((batch.shape[0],), 0, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return self.model.predict(
                        batch, org, resolutions=(1, 128), return_embeddings=False)
            return self.model.predict(
                batch, org, resolutions=(1, 128), return_embeddings=False)

    def forward_core(self, dna_np: np.ndarray, off: int, core_len: int,
                     batch_size: int = 4, heads=HEADS, quiet: bool = False):
        '''dna_np: (N, 131072, 4) float32  ->  per block yield (i, j, {head: (j-i, core_len, T) float32 numpy}).

        '''
        n = len(dna_np)
        i, bs = 0, max(1, int(batch_size))
        pbar = None if quiet else tqdm(total=n, desc="  forward", leave=False)
        while i < n:
            j = min(i + bs, n)
            try:
                batch = torch.from_numpy(np.ascontiguousarray(dna_np[i:j])).to(self.device)
                outputs = self._forward(batch)
                del batch
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise RuntimeError(
                        "OOM at batch size 1: use a different GPU, or reduce memory with --context npad "
                        "(window length is fixed)")
                bs = max(1, bs // 2)
                print(f"\n[model] CUDA OOM, halving batch size -> {bs}, resuming from item {i} "
                      f"(completed items are kept)")
                continue
            out = {}
            for h in heads:
                t = outputs[h][1]                       # (b, 131072, T) 
                out[h] = t[:, off: off + core_len, :].float().cpu().numpy()
            del outputs
            if pbar is not None:
                pbar.update(j - i)
            yield i, j, out
            i = j
        if pbar is not None:
            pbar.close()

    def forward_core_alt(self, ref_oh: np.ndarray, off: int, core_len: int, idx_arr, alt_bases,
                         batch_size: int = 4, heads=HEADS, label: str = "", quiet: bool = False):
        ''' yield (i, j, {head: (j-i, core_len, T) float32 numpy}) '''
        n = len(idx_arr)
        i, bs = 0, max(1, int(batch_size))
        pbar = None if quiet else tqdm(total=n, desc="  forward", leave=False)
        while i < n:
            j = min(i + bs, n)
            dna = np.repeat(ref_oh[None, :, :], j - i, axis=0)
            for b in range(j - i):
                base = BASE2IDX.get(str(alt_bases[i + b]).upper(), -1)
                if base < 0:
                    raise ValueError(f"{label}: row {i + b}: Alt base {alt_bases[i + b]!r} is not ACGT")
                p = off + int(idx_arr[i + b])
                dna[b, p, :] = 0.0
                dna[b, p, base] = 1.0
            try:
                batch = torch.from_numpy(dna).to(self.device)
                outputs = self._forward(batch)
                del batch, dna
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                del dna
                if bs == 1:
                    raise RuntimeError("OOM at batch size 1: use a different GPU")
                bs = max(1, bs // 2)
                print(f"\n[model] CUDA OOM, halving batch size -> {bs}, resuming from item {i} "
                      f"(completed items are kept)")
                continue
            out = {}
            for h in heads:
                t = outputs[h][1]
                out[h] = t[:, off: off + core_len, :].float().cpu().numpy()
            del outputs
            if pbar is not None:
                pbar.update(j - i)
            yield i, j, out
            i = j
        if pbar is not None:
            pbar.close()

def score_dataset(scorer: AfgScorer, motif: str, cell: str, dataset: str,
                  tsv_path: str, reader: GenomeReader | None, context: str,
                  channels: dict, batch_size: int, score_column: str,
                  smoke: int | None, label: str = ""):
    df = pd.read_csv(tsv_path, sep="\t")
    chrom, elem_start, ref_seq, idx_arr, ref_bases, alt_bases = element_geometry(df, motif)
    core_len = len(ref_seq)
    off = (WINDOW_LEN - core_len) // 2

    effects_all = df["VariantExpressionEffect (log2)"].to_numpy(dtype=np.float64)
    if smoke is not None:
        keep = np.arange(min(smoke, len(df)))
        df = df.iloc[keep].reset_index(drop=True)
        idx_arr, ref_bases, alt_bases = idx_arr[keep], [ref_bases[k] for k in keep], [alt_bases[k] for k in keep]
        effects_all = effects_all[keep]

    if context == "genome":
        window = build_genomic_window(reader, chrom, elem_start, core_len)
    else:
        window = embed_core_into_window(ref_seq)
    if len(window) != WINDOW_LEN:
        raise ValueError(f"{motif}: window length {len(window)} != {WINDOW_LEN}")
    if window[off: off + core_len] != ref_seq:
        raise ValueError(f"{motif}: element sequence from the genome does not match the tsv ref_seq "
                         f"(chr{chrom}:{elem_start + 1}-{elem_start + core_len}); "
                         f"--context npad bypasses this, but more likely the coordinate convention or genome build is wrong")
    for k, (i0, rb) in enumerate(zip(idx_arr, ref_bases)):
        if window[off + int(i0)] != rb:
            raise ValueError(f"{motif}: row {k}: window base {window[off + int(i0)]} != Ref {rb}")

    ref_oh = window_to_onehot(window)

    _, _, out = next(scorer.forward_core(ref_oh[None], off, core_len,
                                         batch_size=batch_size, heads=list(channels.keys()),
                                         quiet=True))
    ref_core = {h: out[h][0] for h in channels}            # head -> (core_len, T)

    n = len(df)
    n_forward = 0
    alt_1bp = {h: np.full(n, np.nan) for h in channels}
    alt_core_mean = {h: np.full(n, np.nan) for h in channels}
    t0 = time.time()
    for i, j, out in scorer.forward_core_alt(ref_oh, off, core_len, idx_arr, alt_bases,
                                             batch_size=batch_size, heads=list(channels.keys()),
                                             label=label):
        for h in channels:
            track = channels[h]
            blk = out[h]
            for b in range(j - i):
                k = i + b
                alt_1bp[h][k] = blk[b, int(idx_arr[k]), track]
                alt_core_mean[h][k] = blk[b, :, track].mean()
        n_forward += (j - i)
    secs = time.time() - t0

    # ---- saving ----
    out_dfs, log_rows = {}, []
    for h in channels:
        track = channels[h]
        ref_1bp = ref_core[h][idx_arr, track]
        ref_core_mean = float(ref_core[h][:, track].mean())
        scores = (alt_1bp[h] - ref_1bp) if score_column == "delta_1bp" else (alt_core_mean[h] - ref_core_mean)
        sub = df[["Chromosome", "Position", "Ref", "Alt"]].copy()
        sub["dataset"] = dataset
        sub["cell_line"] = cell
        sub["idx_in_element"] = idx_arr
        sub["afg_ref_1bp"] = ref_1bp
        sub["afg_alt_1bp"] = alt_1bp[h]
        sub["afg_ref_coremean"] = ref_core_mean
        sub["afg_alt_coremean"] = alt_core_mean[h]
        sub["afg_delta_1bp"] = alt_1bp[h] - ref_1bp
        sub["afg_delta_coremean"] = alt_core_mean[h] - ref_core_mean
        sub["scores"] = scores
        sub["variant_effects"] = effects_all
        out_dfs[h] = sub
        both = pd.DataFrame({"s": sub["afg_delta_1bp"], "c": sub["afg_delta_coremean"],
                             "e": sub["variant_effects"]}).dropna()
        log_rows.append(dict(
            dataset=dataset, motif=motif, cell_line=cell, head=h, track_index=track,
            n_variants=n, n_forward=n_forward, seconds=round(secs, 1),
            mean_ref_coremean=round(ref_core_mean, 6),
            pcc_scores_effects=round(_pcc(scores, effects_all), 4),
            pcc_delta_1bp_effects=round(_pcc(both["s"].to_numpy(), both["e"].to_numpy()), 4),
            pcc_delta_coremean_effects=round(_pcc(both["c"].to_numpy(), both["e"].to_numpy()), 4),
            score_column=score_column,
        ))
    return out_dfs, log_rows


def _pcc(x, y) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    x, y = x[m], y[m]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ===========================================================================
# main process
# ===========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Score MPRABase saturation mutagenesis libraries with AlphaGenome DNase/ATAC tracks (S12 task 7)")
    p.add_argument("--weights", default=None,
                   help=f"AlphaGenome weights; defaults to the first existing of: {WEIGHT_CANDIDATES}")
    p.add_argument("--device", default="cuda")
    p.add_argument("--channel-metadata-dir", default="./checks/AlphaGenome/channel_metadata",
                   help="directory containing dnase_1bp_channels.csv / atac_1bp_channels.csv")
    p.add_argument("--channel-map", default=None,
                   help="fallback table for manual track assignment (columns: motif, head, track_index); overrides automatic matching entirely when given")
    p.add_argument("--tsv-dir", default="./tmp",
                   help="directory containing point_MPRABase_*_saturation.tsv")
    p.add_argument("--genome", default="./Datas/D02_grch/GRCh38.primary_assembly.genome.fa")
    p.add_argument("--context", choices=["genome", "npad"], default="genome",
                   help="genome: feed real genomic context (default, recommended); npad: reproduce the N-padded convention of F01n")
    p.add_argument("--out-dir", default="./Supps/S12_afg_variant_heatmap/afg",
                   help="output directory for prediction csvs; this is the path after copying to the server")
    p.add_argument("--batch-size", type=int, default=4, help="initial batch size; halved automatically on CUDA OOM")
    p.add_argument("--score-column", choices=["delta_1bp", "delta_coremean"], default="delta_1bp",
                   help="which column to write into scores (plotting only reads scores); both values are kept in the csv, no need to rerun the model")
    p.add_argument("--motifs", nargs="*", default=None, help="run only these datasets, e.g. --motifs LDLR PKLR")
    p.add_argument("--smoke", type=int, default=None, help="take only the first N variants per dataset (smoke test / timing)")
    p.add_argument("--list-only", action="store_true", help="print track coverage only, do not load the model")
    return p.parse_args(argv)


def resolve_weights(explicit: str | None) -> str:
    cands = [explicit] if explicit else WEIGHT_CANDIDATES
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(f"AlphaGenome weights not found, tried: {[c for c in cands if c]}\n  specify with --weights")


def main(argv=None) -> int:
    args = parse_args(argv)
    tsv_of = {m: os.path.join(args.tsv_dir, f"point_MPRABase_{m}_saturation.tsv")
              for m, _c, _d in MPRABASE_DATASETS}

    tables = load_channel_tables(args.channel_metadata_dir)
    override = load_channel_map_override(args.channel_map) if args.channel_map else {}

    plan, coverage = [], []
    wanted = set(args.motifs) if args.motifs else None
    for motif, cell, dataset in MPRABASE_DATASETS:
        if wanted is not None and motif not in wanted:
            continue
        tsv = tsv_of[motif]
        chans, why = {}, []
        for head in HEADS:
            if (motif, head) in override:
                chans[head] = override[(motif, head)]
                why.append(f"{head}={chans[head]}")
                continue
            ti = resolve_channel(tables, cell, head) if tables.get(head) is not None else None
            if ti is None:
                why.append(f"{head}=NaN")
            else:
                chans[head] = ti
                why.append(f"{head}={ti}")
        status = ("OK" if len(chans) == 2 else ("only 1 track" if len(chans) == 1 else "duplicate"))
        coverage.append(dict(dataset=dataset, motif=motif, cell_line=cell,
                             dnase_track_index=chans.get("dnase", ""),
                             atac_track_index=chans.get("atac", ""),
                             n_channels=len(chans), status=status,
                             tsv=tsv, tsv_exists=os.path.exists(tsv), detail=", ".join(why)))
        if os.path.exists(tsv) and chans:
            plan.append((motif, cell, dataset, tsv, chans))

    cov_df = pd.DataFrame(coverage)
    keep = cov_df[cov_df.status != "duplicate"]
    print("\n" + "=" * 96)
    print("AlphaGenome tracks cover (DNase 305 tracks / ATAC 167 tracks)")
    print("=" * 96)
    print(cov_df[["dataset", "motif", "cell_line", "dnase_track_index", "atac_track_index",
                  "status", "detail"]].to_string(index=False))
    print(f"\ndatasets going into the plot: {len(plan)} / {len(cov_df)} -> "
          f"{', '.join(d for _m, _c, d, _t, _h in plan) if plan else '(none)'}")
    print("=" * 96 + "\n")

    if args.list_only:
        return 0
    if not plan:
        raise SystemExit("no dataset could be predicted, nothing was run")

    # ---- 2) refernce genome ----
    reader = None
    if args.context == "genome":
        idx_cache = os.path.join(args.out_dir, "S12_afg_genome_index.json")
        index = build_genome_index(args.genome, idx_cache)
        reader = GenomeReader(args.genome, index)

    # ---- 3) loading model weights ----
    weights = resolve_weights(args.weights)
    scorer = AfgScorer(weights, device=args.device)

    # ---- 4) datasets ----
    os.makedirs(args.out_dir, exist_ok=True)
    log_rows, used_rows = [], []
    for motif, cell, dataset, tsv, chans in plan:
        print(f"\n{'=' * 96}\n[{dataset}] cell={cell}  channels={chans}  "
              f"context={args.context}  score={args.score_column}\n{'=' * 96}", flush=True)
        out_dfs, rows = score_dataset(scorer, motif, cell, dataset, tsv, reader, args.context,
                                      chans, args.batch_size, args.score_column, args.smoke,
                                      label=dataset)
        for head, sub in out_dfs.items():
            path = os.path.join(args.out_dir, f"afg_{head}_variant_scores_{motif}.csv")
            sub.to_csv(path, index=False)
            print(f"  -> {path}  ({len(sub)} rows)")
            for r in rows:
                if r["head"] == head:
                    r["out_csv"] = path
        log_rows.extend(rows)
        for head, ti in chans.items():
            row = tables[head][tables[head]["track_index"] == ti].iloc[0]
            used_rows.append(dict(dataset=dataset, motif=motif, cell_line=cell, head=head,
                                  track_index=int(ti), track_name=row["track_name"],
                                  biosample_name=row["biosample_name"],
                                  biosample_type=row["biosample_type"],
                                  assay_title=row["assay_title"],
                                  nonzero_mean=row["nonzero_mean"],
                                  npy_file=row["npy_file"]))

    # ---- 5) saving ----
    cov_df.to_csv(os.path.join(args.out_dir, "S12_afg_coverage.csv"), index=False)
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(os.path.join(args.out_dir, "S12_afg_predict_log.csv"), index=False)
    pd.DataFrame(used_rows).to_csv(os.path.join(args.out_dir, "S12_afg_channel_used.csv"), index=False)

    print("\n" + "=" * 96)
    print("sign self-check (PCC > 0 means scores orientation is right: lower = more disruptive)")
    print("=" * 96)
    cols = ["dataset", "head", "n_variants", "n_forward", "seconds", "mean_ref_coremean",
            "pcc_scores_effects", "pcc_delta_1bp_effects", "pcc_delta_coremean_effects"]
    print(log_df[cols].to_string(index=False))
    bad = log_df[log_df["pcc_scores_effects"] <= 0]
    if len(bad):
        print(f"\nnote: {len(bad)} (dataset, track) pairs with PCC <= 0")
    if args.score_column == "delta_1bp":
        d1 = log_df["pcc_delta_1bp_effects"].mean()
        dc = log_df["pcc_delta_coremean_effects"].mean()
        print(f"\nconvention comparison (mean PCC): delta_1bp={d1:.4f}  delta_coremean={dc:.4f}")
        if dc > d1 + 0.02:
            print(f"  coremean might be more stable -> --score-column delta_coremean could rewrite scores")
    print(f"\noutput directory: {os.path.abspath(args.out_dir)}")
    print("=" * 96)
    if reader is not None:
        reader.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
