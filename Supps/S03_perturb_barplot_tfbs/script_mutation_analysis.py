import os, sys
import numpy as np
import pandas as pd
import argparse
import json

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap
from scipy.stats import spearmanr, pearsonr, rankdata, wilcoxon, chi2

def get_matched(pred, anno, keywords, top_cols=None, match_mode="soft"):
    """
    Parameters:
    pred : ndarray (M samples x N channels)
    anno : DataFrame with metadata for N channels
    keywords: dict or list - depends on match_mode
    top_cols: int or None
    match_mode: "soft" (default) or "hard"
    """
    if match_mode == "hard":
        if not isinstance(keywords, dict):
            raise ValueError("For hard mode, keywords must be a dictionary")
        # Check for invalid keys
        validkeys = {'celltype', 'motif'}
        inputkeys = set(keywords.keys())
        invalidkeys = inputkeys - validkeys
        if invalidkeys:
            raise ValueError(f"Illegal keys: {invalidkeys}. Only 'celltype' and/or 'motif' allowed")
        # Check if at least one valid key is present
        if not inputkeys & validkeys:
            raise ValueError("At least one valid key required: 'celltype' or 'motif'")
    elif match_mode == "soft":
        if not isinstance(keywords, list):
            raise ValueError("For soft mode, keywords must be a list")
    else:
        raise ValueError("Invalid match_mode. Must be 'hard' or 'soft'")
    if len(keywords) == 0 or anno.empty:
        return np.array([]), pd.DataFrame()
    lower_keywords = {k: [x.lower() for x in v] 
                     for k,v in keywords.items()} if match_mode == "hard" else                     [str(kw).lower() for kw in keywords]
    if match_mode == "hard":
        mask = pd.Series(True, index=anno.index)
        if 'celltype' in lower_keywords:
            cell_mask = anno['celltype'].str.lower().str.contains(
                '|'.join(lower_keywords['celltype']), na=False)
            mask &= cell_mask
        if 'motif' in lower_keywords:
            feat_mask = anno['feature'].str.lower().apply(
                lambda x: any(kw in x for kw in lower_keywords['motif']))
            mask &= feat_mask
        matched_idx = anno[mask].index.values
    else:  # Soft mode
        match_counts = []
        for i, row in anno.iterrows():
            row_text = ' '.join(map(str, row)).lower()
            count = sum(kw in row_text for kw in lower_keywords)
            match_counts.append(count)
        match_counts = np.array(match_counts)
        sorted_idx = np.argsort(-match_counts)
        matched_idx = sorted_idx[match_counts[sorted_idx] > 0]
    if top_cols and len(matched_idx) > top_cols:
        matched_idx = matched_idx[:top_cols]
    if len(matched_idx) == 0:
        return np.array([]), pd.DataFrame()
    matched_pred = pred[:, matched_idx]
    matched_anno = anno.loc[matched_idx].reset_index(drop=False)
    return matched_pred, matched_anno

def generate_pdf(data, anno, pdf_path, title_prefix, row_labels=None):
    """
    Helper function to generate a PDF with heatmap(s) and a single table page.

    Parameters:
    data        : ndarray (M samples x N channels), heatmap data
    anno        : DataFrame, channel annotations (must contain 'index')
    pdf_path    : str, path for the output PDF file
    title_prefix: str, prefix for heatmap titles
    row_labels  : list, optional labels for heatmap rows
    """
    # Input validation
    if not isinstance(data, np.ndarray) or data.size == 0:
        raise ValueError("data must be a non-empty numpy array")
    if not isinstance(anno, pd.DataFrame) or anno.empty:
        raise ValueError("anno must be a non-empty DataFrame")
    if 'index' not in anno.columns:
        raise ValueError("anno must contain an 'index' column")
    if not isinstance(pdf_path, str) or not pdf_path.endswith('.pdf'):
        raise ValueError("pdf_path must be a valid string ending with '.pdf'")
    if row_labels is not None:
        if not isinstance(row_labels, (list, tuple)) or not row_labels:
            raise ValueError("row_labels must be a non-empty list or tuple")
        if len(row_labels) != data.shape[0]:
            raise ValueError("row_labels length must match number of samples")

    # A4 size in inches (8.27 x 11.69 inches)
    a4_width, a4_height = 8.27, 11.69

    n_samples, n_channels = data.shape

    # Initialize PDF
    with PdfPages(pdf_path) as pdf:
        # Split heatmap if n_samples > 25
        max_seqs_per_page = 25
        n_heatmap_pages = (n_samples + max_seqs_per_page - 1) // max_seqs_per_page

        for page in range(n_heatmap_pages):
            start_idx = page * max_seqs_per_page
            end_idx = min((page + 1) * max_seqs_per_page, n_samples)
            page_data = data[start_idx:end_idx, :]

            # Adaptive figure size for heatmap
            heatmap_height = min(7.5, 0.1 * n_channels + 4)  # Scale height, max 7.5 inches
            fig = plt.figure(figsize=(a4_width, a4_height), dpi=300)
            ax = fig.add_axes([0.1, 0.1, 0.8, heatmap_height/a4_height])

            # Create heatmap
            yticklabels = row_labels[start_idx:end_idx] if row_labels else True
            sns.heatmap(data=page_data, cmap="RdBu_r", xticklabels=anno["index"],
                        yticklabels=yticklabels, ax=ax)
            ax.set_xlabel("Tracks (Index)")
            ax.set_ylabel("Seqs (Ids)")
            ax.set_title(f"{title_prefix} Heatmap ({start_idx+1}-{end_idx} of {n_samples} samples)")

            # Save heatmap page
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Prepare table data
        table_columns = ['index', 'model', 'celltype', 'feature', 'source', 'p_value']
        available_columns = [col for col in table_columns if col in anno.columns]
        if not available_columns:
            raise ValueError("No valid table columns found in anno")
        table_data = anno[available_columns].copy()
        
        # Format p-values as scientific notation if present
        if 'p_value' in available_columns:
            table_data['p_value'] = table_data['p_value'].apply(lambda x: f"{x:.2e}" if pd.notnull(x) else "N/A")

        # Convert all entries to strings and truncate long strings
        max_chars = 20
        for col in available_columns:
            table_data[col] = table_data[col].astype(str).apply(
                lambda x: x[:max_chars-3] + '...' if len(x) > max_chars else x)

        table_data = table_data.values.tolist()
        table_headers = ['Index', 'Model', 'Cell Type', 'Feature', 'Source', 'P-Value'][:len(available_columns)]

        # Create figure for table (single page)
        fig = plt.figure(figsize=(a4_width, a4_height), dpi=300)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        # Create table
        table = ax.table(cellText=table_data,
                         colLabels=table_headers,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])

        # Adjust table font size and scaling
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)  # Scale table height for readability

        # Hide table axes
        ax.axis('off')
        ax.set_title("Annotation Table")

        # Save table page
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        print(f"PDF report saved to {pdf_path} with {n_heatmap_pages + 1} pages")

def plot_differ_heatmap(plot_path, pos_pred, neg_pred, anno, top_cols=10, top_rows=None, pi=None, ni=None, normalize=True):
    """
    Parameters:
    pos_pred  : ndarray (M x N), positive samples (paired with neg_pred)
    neg_pred  : ndarray (M x N), negative samples (paired with pos_pred)
    anno      : DataFrame, channel annotations (must contain 'index')
    top_cols  : int, number of top/bottom differing channels to select
    top_rows  : int, optional number of top differing rows to select
    pi        : ndarray (K1 x N), positive control, optional
    ni        : ndarray (K2 x N), negative control, optional
    normalize : bool, whether to normalize columns
    """
    # Input validation
    if not isinstance(pos_pred, np.ndarray) or pos_pred.size == 0:
        raise ValueError("pos_pred must be a non-empty numpy array")
    if not isinstance(neg_pred, np.ndarray) or neg_pred.size == 0:
        raise ValueError("neg_pred must be a non-empty numpy array")
    if not isinstance(anno, pd.DataFrame) or anno.empty:
        raise ValueError("anno must be a non-empty DataFrame")
    if 'index' not in anno.columns:
        raise ValueError("anno must contain an 'index' column")
    if not isinstance(top_cols, int) or top_cols <= 0:
        raise ValueError("top_cols must be a positive integer")
    if top_rows is not None:
        if not isinstance(top_rows, int) or top_rows <= 0:
            raise ValueError("top_rows must be a positive integer")

    n_channels = pos_pred.shape[1]
    for mat in [neg_pred, pi, ni]:
        if mat is not None and mat.shape[1] != n_channels:
            raise ValueError("All matrices must have the same number of channels")

    # Ensure paired samples for Wilcoxon signed-rank test
    if pos_pred.shape[0] != neg_pred.shape[0]:
        raise ValueError("pos_pred and neg_pred must have the same number of samples for paired Wilcoxon test")
    if pos_pred.shape[0] < 1:
        raise ValueError("pos_pred and neg_pred must each have at least one sample")

    # Validate top_rows against available rows
    total_rows = pos_pred.shape[0]
    if pi is not None:
        total_rows += pi.shape[0]
    if ni is not None:
        total_rows += ni.shape[0]
    if top_rows is not None and top_rows > total_rows:
        raise ValueError(f"top_rows ({top_rows}) cannot exceed total available rows ({total_rows})")

    # Check column limit early
    n_channels = pos_pred.shape[1]
    if top_cols and top_cols > 30:
        raise ValueError("Too much top cols to display")
    if not top_cols and n_channels > 30:
        raise ValueError("No top cols defined, and too much channels to display")

    # Column-wise normalization
    def normalize_matrix(matrix):
        if matrix is None:
            return None
        if normalize:
            scaler = MinMaxScaler()
            normalized = np.zeros_like(matrix)
            for j in range(matrix.shape[1]):
                col_data = matrix[:, j].reshape(-1, 1)
                if np.ptp(col_data) == 0:
                    normalized[:, j] = 0.5
                else:
                    normalized[:, j] = scaler.fit_transform(col_data).flatten()
            return normalized
        return matrix

    pos_normalized = normalize_matrix(pos_pred)
    neg_normalized = normalize_matrix(neg_pred)
    pi_normalized = normalize_matrix(pi)
    ni_normalized = normalize_matrix(ni)

    # Calculate Wilcoxon signed-rank p-values and mean differences
    p_values = np.ones(n_channels)
    mean_diff = pos_normalized.mean(axis=0) - neg_normalized.mean(axis=0)
    for j in range(n_channels):
        try:
            if np.ptp(pos_normalized[:, j] - neg_normalized[:, j]) == 0:
                p_values[j] = 1.0  # Constant differences have no significant difference
            else:
                stat, p = wilcoxon(pos_normalized[:, j], neg_normalized[:, j])
                p_values[j] = p
        except ValueError:
            p_values[j] = 1.0  # Handle cases where test fails (e.g., too few samples)

    # Add sign annotation and p-values to anno (create a copy to avoid modifying original)
    plot_anno = anno.copy()
    symbol = ["(+)" if diff > 0 else "(-)" for diff in mean_diff]
    plot_anno["index"] = [f"{index}{sym}" for index, sym in zip(anno["index"], symbol)]
    plot_anno["p_value"] = p_values

    # Sort tracks by p-value (smallest p-value = most significant)
    top_idx = np.argsort(p_values)[:top_cols]  # Smallest p-values (max difference)
    bottom_idx = np.argsort(p_values)[-top_cols:]  # Largest p-values (min difference)
    
    ## Select top_rows by combining p-values across top_cols with Fisher test
    row_indices = None
    if top_rows is not None:
        # Compute local p-values for each row across top_cols
        local_p_values = np.ones((pos_pred.shape[0], len(top_idx)))
        for j_idx, j in enumerate(top_idx):
            for i in range(pos_pred.shape[0]):
                try:
                    diff = pos_normalized[i, j] - neg_normalized[i, j]
                    if diff == 0:
                        local_p_values[i, j_idx] = 1.0
                    else:
                        # Single-sample Wilcoxon test (difference deviates from 0)
                        stat, p = wilcoxon([diff], alternative='two-sided')
                        local_p_values[i, j_idx] = p
                except ValueError:
                    local_p_values[i, j_idx] = 1.0  # Set p-value to 1 if test fails

        # Combine p-values using Fisher’s method
        p_combined = np.ones(pos_pred.shape[0])
        for i in range(pos_pred.shape[0]):
            # Compute -2 * sum(ln(p))
            chi2_stat = -2 * np.sum(np.log(np.maximum(local_p_values[i, :], 1e-10)))  # Avoid ln(0)
            # Degrees of freedom: 2 * number of top_cols
            df = 2 * len(top_idx)
            # Compute combined p-value
            p_combined[i] = 1 - chi2.cdf(chi2_stat, df)
        
        # If p-value combination fails (all p-values are 1), use mean difference as fallback
        if np.all(p_combined == 1.0):
            abs_diff = np.abs(pos_normalized[:, top_idx] - neg_normalized[:, top_idx])
            score_per_row = np.mean(abs_diff, axis=1)
        else:
            score_per_row = -np.log(p_combined + 1e-10)  # Negative log transform, smaller p-values yield higher scores

        # Sort rows by score (highest scores first)
        row_indices = np.argsort(score_per_row)[-min(top_rows, pos_pred.shape[0]):]
        # Allocate remaining rows to pi and ni
        remaining_rows = top_rows - len(row_indices)
        if remaining_rows > 0 and pi_normalized is not None:
            pi_rows = min(pi_normalized.shape[0], remaining_rows)
            row_indices = np.concatenate([row_indices, np.arange(pi_rows)])
            remaining_rows -= pi_rows
        if remaining_rows > 0 and ni_normalized is not None:
            ni_rows = min(ni_normalized.shape[0], remaining_rows)
            row_indices = np.concatenate([row_indices, np.arange(ni_rows)])

    def stack_data(selected_idx):
        """Stack data and create row labels for selected channels."""
        rows = []
        row_labels = []

        if pi_normalized is not None and pi_normalized.shape[0] > 0:
            pi_data = pi_normalized[:, selected_idx]
            if row_indices is not None:
                pi_data = pi_data[:min(pi_data.shape[0], top_rows)]
            rows.append(pi_data)
            row_labels += [f"pi{i+1}" for i in range(pi_data.shape[0])]

        if ni_normalized is not None and ni_normalized.shape[0] > 0:
            ni_data = ni_normalized[:, selected_idx]
            if row_indices is not None:
                ni_data = ni_data[:min(ni_data.shape[0], top_rows - len(rows))]
            rows.append(ni_data)
            row_labels += [f"ni{i+1}" for i in range(ni_data.shape[0])]
        
        pos_data = pos_normalized[:, selected_idx]
        neg_data = neg_normalized[:, selected_idx]
        if row_indices is not None:
            pos_data = pos_data[row_indices[:min(len(row_indices), pos_data.shape[0])], :]
            neg_data = neg_data[row_indices[:min(len(row_indices), neg_data.shape[0])], :]
        
        rows.append(pos_data)
        row_labels += [f"pos{i+1}" for i in range(pos_data.shape[0])]
        rows.append(neg_data)
        row_labels += [f"neg{i+1}" for i in range(neg_data.shape[0])]

        if not rows:
            raise ValueError("No valid data to stack: all input matrices are empty")

        full_matrix = np.vstack(rows)
        if not row_labels:
            raise ValueError("No row labels generated: all input matrices are empty")

        return full_matrix, plot_anno.iloc[selected_idx].reset_index(drop=True), row_labels

    # Generate PDF for max difference (smallest p-values)
    max_data, max_anno, max_labels = stack_data(top_idx)
    max_path = plot_path + '_differ_max.pdf'
    generate_pdf(max_data, max_anno, max_path, "Diff Max", row_labels=max_labels)
    return

parser = argparse.ArgumentParser(description="Mutation analysis script for generating differential heatmaps")
parser.add_argument("--pos_pred", help="Path to positive prediction dataset")
parser.add_argument("--neg_pred", help="Path to negative prediction dataset")
parser.add_argument("--outs_path", help="Output path for temporary and final outputs")
parser.add_argument("--csv_path", help="Feature information CSV for describing tracks")
parser.add_argument("--match_mode", help="Match mode: 'soft' or 'hard'")
parser.add_argument("--keywords", help="Keywords for filtering; for soft: JSON list like '[\"BHLHE40\"]', for hard: JSON dict like '{\"celltype\": [\"HepG2\", \"K562\"], \"motif\": [\"BHLHE40\"]}'")
parser.add_argument("--top_rows", help="Number of top rows to select", type=int, default=None)
parser.add_argument("--top_cols", help="Number of top columns to select", type=int, default=30)

args = parser.parse_args()

# Load predictions and annotations
pos_pred = np.load(args.pos_pred)
neg_pred = np.load(args.neg_pred)
uni_pred = np.concatenate([pos_pred, neg_pred], axis=0)
uni_anno = pd.read_csv(args.csv_path)

# Parse keywords based on match_mode
keywords = json.loads(args.keywords)
if args.match_mode == "soft" and not isinstance(keywords, list):
  raise ValueError("For soft match_mode, keywords must be a list")
if args.match_mode == "hard" and not isinstance(keywords, dict):
  raise ValueError("For hard match_mode, keywords must be a dict")

# Filter predictions based on keywords and match mode
filt_pred, filt_anno = get_matched(uni_pred, uni_anno, keywords=keywords, top_cols=args.top_cols, match_mode=args.match_mode)

# Split filtered predictions
filt_pos_pred = filt_pred[:len(pos_pred)]
filt_neg_pred = filt_pred[len(pos_pred):]

# Generate differential heatmap
plot_differ_heatmap(plot_path=args.outs_path + "/heatmap", pos_pred=filt_pos_pred, neg_pred=filt_neg_pred, anno=filt_anno, top_cols=args.top_cols, top_rows=args.top_rows)
