#!/usr/bin/env bash
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh

trap 'echo "[INFO] Caught signal, terminating all subprocesses…"; kill 0; exit 1' SIGINT SIGTERM

# ----------------------------------------------------------------------------
# run_unified_prediction.sh
#
# This script generates and runs model-specific Python prediction scripts
# for a set of genomics models, then merges their outputs into unified files.
#
# Usage:
#   bash run_unified_prediction.sh \
#     -i SEQS_PATH      Path to input FASTA/sequences file                \
#     -o OUTS_PATH      Path to output directory                           \
#     -t TRACKS_PATH    Path to tracks/features CSV file                  \
#     [-l LOC]          Location/mode tag for model outputs (default: center) \
#     [-d DEV]          CUDA device index (default: 0)                    \
#     [-m MODEL_INDEXES]Comma-separated list of model indices (e.g. 0,3,5; default: all) \
#     [-n|--nohup]      Relaunch entire pipeline under nohup to OUTS_PATH/nohup.out \
#     [-c|--clear]      After completion, clear all files in OUTS_PATH except uni_pred.npy, uni_anno.csv, reports.txt, script_unified.py \
#
# Available models (index: name):
#     0: Malinois     | [input L=200 | output L=1 | MPRA   | celltypes: K562,HepG2,SK-N-SH]
#     1: Basset       | [input L=600 | output L=1 | DNase  | celltypes: 164]
#     2: DanQ         | [input L=1000| output L=1 | 919 features: DNase/TFs/histones]
#     3: MPRALegNet   | [input L=230±15|output L=1| lentiMPRA | HepG2,K562,WTC11]
#     4: SahuCNN      | [input L=170 & 120|output L=1| STARR-seq & isPromoter | GP5d]
#     5: APARENT2     | [input L=205|output L=1| APA MPRA cleavage scores]
#     6: DeepDNAshape | [input L=any| output L=var| 14 DNA shape features]
#     7: CLIPNET      | [input L=1000|output L=1| PRO-cap | LCL]
#     8: Puffin       | [input L=1000|output L=350| 12 CAGE/RAMPAGE tracks]
#     9: Enformer     | [input L=196608|output L=896| 5313 regulatory tracks]
#    10: Basenji2     | [input L=196608|output L=1408| 5313 regulatory tracks]
#    11: Expecto      | [input L=2000|output L=1| 2002 epigenomic tracks]
#    12: Sei          | [input L=4096|output L=1| 21907 regulatory profiles]
#    13: SpliceAI     | [input L=any|output L=any+10000| 3 splice-site probs]
#    14: Borzoi       | [input L=524288|output L=16352| 7611 regulatory tracks]
#    15: SegmentNT    | [input L=any|output L=any| 14 genomic annotation probabilities]
#
# ----------------------------------------------------------------------------

declare -a MODEL_LIST=(
  "Malinois" "Basset" "DanQ" "MPRALegNet" "SahuCNN" "APARENT2" "DeepDNAshape" 
  "CLIPNET" "Puffin" "Enformer" "Basenji2" "Expecto" "Sei" "SpliceAI" "Borzoi" "SegmentNT"
)
declare -a ENV_LIST=(
  "lightning" "lightning" "danq" "lightning" "lightning" "aparent2" "deepdnashape" 
  "lightning" "lightning" "transformers" "lightning" "lightning" "lightning" "lightning" "lightning" "transformers"
)

declare -a MODEL_DESCRIPTIONS=(
  # 0: Malinois
  "[input length] 200 | [output length] 1 | [tracks] MPRA | [celltype] K562, HepG2, SK-N-SH | [origin] Machine-guided design of cell-type-targeting cis-regulatory elements"
  # 1: Basset
  "[input length] 600 | [output length] 1 | [tracks] DNase-seq | [celltype] 164 | [origin] Basset: learning the regulatory code of the accessible genome with deep convolutional neural networks"
  # 2: DanQ
  "[input length] 1000 | [output length] 1 | [tracks] 919: 125 DNase features, 690 TF features, 104 histone features | [origin] DanQ: a hybrid convolutional and recurrent deep neural network for quantifying the function of DNA sequences"
  # 3: MPRALegNet
  "[input length] 230 (±15 paddings) | [output length] 1 | [tracks] lentiMPRA | [celltype] HepG2, K562, WTC11 | [origin] Massively parallel characterization of transcriptional regulatory elements in three diverse human cell types "
  # 4: SahuCNN
  "[input length] 170 & 120 | [output length] 1 | [tracks] STARR-seq & isPromoter | [celltype] GP5d | [origin] Sequence determinants of human gene regulatory elements"
  # 5: APARENT2
  "[input length] 205 | [output length] 1 | [tracks] APA MPRA, non-normalized cleavage scores, 205 (3' cleavage distribution, equals sequence length) + 1 (extra position represents the total isoform score of the distal signal) |\
  [origin] Deciphering the impact of genetic variation on human polyadenylation using APARENT2"
  # 6: DeepDNAshape
  "[input length] any | [tracks] 14: 6 inter-bp features (shift, slide, rise, tilt, roll, helix twist), 6 intra-bp features (shear, stretch, stragger, buckle, propeller twist, opening),\
  and 2 minor groove features (minor groove width, electrostatic potential) | [origin] Predicting DNA structure using a deep learning method"
  # 7: CLIPNET
  "[input length] 1000 | [output length] 1 | [tracks] PRO-cap quantity | [celltype] LCL | [origin] Dissection of core promoter syntax through single nucleotide resolution modeling of transcription initiation"
  # 8: Puffin
  "[input length] L (1000) | [output length] L-650 | [tracks] 10: FANTOM_CAGE, ENCODE_CAGE, ENCODE_RAMPAGE, GRO_CAP, PRO_CAP, rev strand FANTOM_CAGE, rev strand ENCODE_CAGE, rev strand ENCODE_RAMPAGE, rev strand GRO_CAP, rev strand PRO_CAP |\
   [origin] Sequence basis of transcription initiation in the human genome"
  # 9: Enformer
  "[input length] 196608 | [output length] 896 | [tracks] 5313: 2131 TF, 1860 histone modification, 684 DNase-seq or ATAC-seq, and 638 CAGE tracks | [origin] Effective gene expression prediction from sequence by integrating long-range interactions"
  # 10: Basenji2
  "[input length] 196608 | [output length] 1408 | [tracks] 5313: 2131 TF, 1860 histone modification, 684 DNase-seq or ATAC-seq, and 638 CAGE tracks | [origin] Cross-species regulatory sequence activity prediction"
  # 11: Expecto
  "[input length] 2000 | [output length] 1 | [tracks] 2002: histone marks, TF-binding and  chromatin accessibility profiles | [origin] Deep learning sequence-based ab initio prediction of variant effects on expression and disease risk"
  # 12: Sei
  "[input length] 4096 | [output length] 1 | [tracks] 21907: 9,471 TF binding, 10,064 histone mark and 2,372 chromatin accessibility profiles from Cistrome Project, ENCODE2 and Roadmap Epigenomics projects |\
  [origin] A sequence-based global map of regulatory activity for deciphering human genetics"
  # 13: SpliceAI
  "[input length] L | [output length] L+10000 | [tracks] 3: probs of neither/acceptor_prob/donor_prob | [origin] Predicting Splicing from Primary Sequence with Deep Learning"
  # 14: Borzoi
  "[input length] 524288 | [output length] 16352 | [tracks] 7611: CAGE, DNase/ATAC, CAGE and RNA-seq | [origin] Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation"
  # 15: SegmentNT
  "[input length] L | [output length] L | [tracks] 14: probabilities/non-probabilities of procoding gene, lncRNA, exon, intron, splice donor, splice acceptor, 5UTR, 3UTR, CTCF-bound, polyA signal, enhancer tissue-specific,\
  enhancer tissue-invariant, promoter tissue-specific, promoter tissue-invariant | [origin] SegmentNT: annotating the genome at single-nucleotide resolution with DNA foundation models"
)

# define functions
# dDNAshape
generate_py() {
  local NAME="$1" OUTS="$2"
  local LNAME="${NAME,,}"      
  local SCRIPT="$OUTS/script_${LNAME}.py"

  local CLASS_NAME="$NAME"
  if [[ "$NAME" == "DeepDNAshape" ]]; then
    CLASS_NAME="dDNAshape"
  fi

  cat > "$SCRIPT" <<EOF

import os, sys
import numpy as np
import pandas as pd
import argparse

sys.path.append("./")

from models.${LNAME} import ${CLASS_NAME}

def open_fa(file):
    record = []
    with open(file, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                record.append(line.strip())
    return record

parser = argparse.ArgumentParser()
parser.add_argument("--seqs_path", help="Define the sequence datasets for evaluation")
parser.add_argument("--outs_path", help="Output paths for temporary and final outputs")
parser.add_argument("--csv_features", help="Feature information for describe tracks")
parser.add_argument("--mode", help="Mode of the model outputs, e.g., center, center_averaged, total")
args = parser.parse_args()

seqs = open_fa(args.seqs_path)

dp = ${CLASS_NAME}()
pred, df = dp.predict_bash(seqs, args.csv_features, args.mode)
np.save(os.path.join(args.outs_path, "pred_${LNAME}.npy"), pred)
df.to_csv(os.path.join(args.outs_path, "anno_${LNAME}.csv"), index=False)
EOF

  echo "[FILE] Generated $SCRIPT"
}


unified_py() {
  local OUTS="$1"
  local SCRIPT="$OUTS/script_unified.py"

  local npy_list=""
  for name in "${MODEL_LIST[@]}"; do
    local lower="${name,,}"
    npy_list+="\"$lower\", "
  done
  npy_list="${npy_list%, }"

  cat > "$SCRIPT" <<EOF
import os
import numpy as np
import pandas as pd

outs = r"$OUTS"
npy_order = [ $npy_list ]

uni_list = []
uni_df   = []

for name in npy_order:
    arr_path  = os.path.join(outs, f"pred_{name}.npy")
    anno_path = os.path.join(outs, f"anno_{name}.csv")

    if os.path.exists(arr_path):
        print(f"[INFO] Merging {arr_path}")
        uni_list.append(np.load(arr_path))

    if os.path.exists(anno_path):
        print(f"[INFO] Merging {anno_path}")
        df = pd.read_csv(anno_path)
        uni_df.append(df)

if uni_list:
    uni_arr = np.concatenate(uni_list, axis=1)
    np.save(os.path.join(outs, "uni_pred.npy"), uni_arr)
else:
    print("[ERROR] no prediction arrays found!")

if uni_df:
    uni_all = pd.concat(uni_df, axis=0)
    uni_all = uni_all.reset_index(drop=True)
    uni_all.to_csv(os.path.join(outs, "uni_anno.csv"), index=False)
else:
    print("[ERROR] no annotation CSVs found!")
EOF

  echo "[FILE] Generated unified script: $SCRIPT"
}

# Default values
LOC="center"
DEV="0"
SELECTED_MODELS=""
NOHUP=0
CLEAR=0

usage() {
  cat <<EOF
Usage: $0 -i SEQS_PATH -o OUTS_PATH -t TRACKS_PATH [-l LOC] [-d DEV] [-m MODEL_INDEXES] [-n] [-c]
  -i  Path to input sequences file (seqs_path)
  -o  Path to output directory (outs_path)
  -t  Path to tracks/features CSV file (tracks_path)
  -l  Location tag for unified_prediction (default: "center", "full")
  -d  CUDA device index (default: 0)
  -m  Comma-separated model index list (e.g., 0,2,5). See list below.
  -n  Run entire pipeline under nohup (into OUTS_PATH/nohup.out)
  -c  After finish, clear all files in OUTS_PATH except uni_pred.npy and reports.txt
EOF
  echo
  echo "Available models:"
  for i in "${!MODEL_LIST[@]}"; do
    printf "  %2d: %-12s | %s\n" "$i" "${MODEL_LIST[$i]}" "${MODEL_DESCRIPTIONS[$i]}"
  done
  exit 1
}

# ----------------------------------------------------------------------------
# Parse args
# ----------------------------------------------------------------------------

ARGS=()
for arg in "$@"; do
  case "$arg" in
    --nohup) ARGS+=(-n)    ;;
    --clear) ARGS+=(-c)    ;;
    *)       ARGS+=("$arg") ;;
  esac
done
set -- "${ARGS[@]}"

while getopts "i:o:t:l:d:m:nch" opt; do
  case $opt in
    i) SEQS_PATH=$OPTARG ;;
    o) OUTS_PATH=$OPTARG ;;
    t) TRACKS_PATH=$OPTARG ;;
    l) LOC=$OPTARG ;;
    d) DEV=$OPTARG ;;
    m) SELECTED_MODELS=$OPTARG ;;
    n) NOHUP=1 ;;
    c) CLEAR=1 ;;
    h|*) usage ;;
  esac
done

# Ensure required args are provided
if [[ -z "$SEQS_PATH" || -z "$OUTS_PATH" || -z "$TRACKS_PATH" ]]; then
  usage
fi

SEQS_PATH="$(realpath "$SEQS_PATH")"
OUTS_PATH="$(realpath "$OUTS_PATH")"
TRACKS_PATH="$(realpath "$TRACKS_PATH")"

# Prepare output and report
mkdir -p "$OUTS_PATH"

# if --nohup, re‐launch under nohup and exit
if [[ $NOHUP -eq 1 ]]; then
  echo "[INFO] Relaunching under nohup; output will go to $OUTS_PATH/nohup.out"
  NEW_ARGS=()
  for arg in "$@"; do
    if [[ "$arg" == "-n" || "$arg" == "--nohup" ]]; then
      continue
    fi
    NEW_ARGS+=("$arg")
  done

  nohup bash "$0" "${NEW_ARGS[@]}" >"$OUTS_PATH/nohup.out" 2>&1 &
  exit 0
fi

REPORT_FILE="$OUTS_PATH/reports.txt"
REPORT_FILE="$(realpath "$REPORT_FILE")"
{
  echo "Report generated: $(date)"
  echo
} > "$REPORT_FILE"



# Set CUDA device
export CUDA_VISIBLE_DEVICES="$DEV"

# Build indices array
IFS=',' read -ra MODEL_INDICES <<< "$SELECTED_MODELS"
if [[ -z "$SELECTED_MODELS" ]]; then
  MODEL_INDICES=("${!MODEL_LIST[@]}")
fi

# Setting Directory
CURRENT_DIR="$(pwd)"
PROJECT_DIR="$(dirname "$0")/.."


# ----------------------------------------------------------------------------
# Run each model
# ----------------------------------------------------------------------------

for idx in "${MODEL_INDICES[@]}"; do
  NAME="${MODEL_LIST[$idx]}"
  ENV="${ENV_LIST[$idx]}"
  LNAME="${NAME,,}"
  SEPARATOR=$(printf '%0.s#' {1..80})

  # Echo to console and log
  echo "$SEPARATOR" | tee -a "$REPORT_FILE"
  echo "[START] ${NAME}" | tee -a "$REPORT_FILE"
  echo "$SEPARATOR" | tee -a "$REPORT_FILE"

  generate_py "$NAME" "$OUTS_PATH"

  CMD=(python "$OUTS_PATH/script_${LNAME}.py" \
       --seqs_path "$SEQS_PATH" \
       --outs_path "$OUTS_PATH" \
       --csv_features "$TRACKS_PATH" \
       --mode "$LOC")

  echo "[INFO] Running ${NAME} in env Digital_Platform_${ENV}" | tee -a "$REPORT_FILE"
  echo "[INFO] Running on CUDA device ${DEV}" | tee -a "$REPORT_FILE"
  echo "[CMD] ${CMD[*]}" | tee -a "$REPORT_FILE"

  # Activate and launch

  (
  cd "$PROJECT_DIR"
  conda activate "Digital_Platform_${ENV}"
  "${CMD[@]}" 2>&1 | tee -a "$REPORT_FILE"
  ) &

  PID=$!
  cd "$CURRENT_DIR"
  echo "[INFO] PID $PID started for $NAME" | tee -a "$REPORT_FILE"

  # Wait and log status
  wait $PID
  code=$?
  if [[ $code -eq 0 ]]; then
    echo "[INFO] PID $PID ($NAME) succeeded" | tee -a "$REPORT_FILE"
  else
    echo "[ERROR] PID $PID ($NAME) failed (code $code)" | tee -a "$REPORT_FILE"
  fi

  echo "$SEPARATOR" | tee -a "$REPORT_FILE"
  echo "[END] ${NAME}" | tee -a "$REPORT_FILE"
  echo "$SEPARATOR" | tee -a "$REPORT_FILE"
  echo "" | tee -a "$REPORT_FILE"

done

# ----------------------------------------------------------------------------
# Unified prediction & cleanup
# ----------------------------------------------------------------------------

unified_py "$OUTS_PATH"
echo "[INFO] Running unified prediction" | tee -a "$REPORT_FILE"
python "$OUTS_PATH/script_unified.py" 2>&1 | tee -a "$REPORT_FILE"

if [[ $CLEAR -eq 1 ]]; then
  echo "[INFO] Clearing intermediate files" | tee -a "$REPORT_FILE"
  find "$OUTS_PATH" -maxdepth 1 \
    ! -name "uni_pred.npy" \
    ! -name "uni_anno.csv" \
    ! -name "reports.txt" \
    -type f -exec rm -f {} +
fi

# Deactivate environment
conda deactivate