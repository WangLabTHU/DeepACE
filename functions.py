import os, sys
import numpy as np
import pandas as pd
import numpy as np
import random
from collections import Counter


from scipy.stats import spearmanr, pearsonr, rankdata, wilcoxon, chi2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster import hierarchy
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap

#######################################################################
#---------------------------data in/out-------------------------------#
#######################################################################

def write_txt(file, data):
    f = open(file,'w')
    i = 0
    while i < len(data):
        f.write(data[i] + '\n')
        i = i + 1
    f.close()

def open_fa(file):
    record = []
    f = open(file,'r')
    for item in f:
        if '>' not in item:
            record.append(item[0:-1])
    f.close()
    return record

#######################################################################
#------------------------tracks aggregation---------------------------#
#######################################################################

def get_zscore_cluster(original_matrix, factor=0.8):
    M, N = original_matrix.shape
    if M == 0:
        return np.array([]), np.array([])
    
    zscores = (original_matrix - original_matrix.mean(axis=1, keepdims=True)) / original_matrix.std(axis=1, keepdims=True, ddof=0)
    similarity = 1 - pairwise_distances(zscores, metric='cosine')
    
    adjacency = similarity >= factor
    n_components, labels = connected_components(csr_matrix(adjacency), directed=False)
    
    label_counts = Counter(labels)
    max_label = label_counts.most_common(1)[0][0]
    cluster_indices = np.where(labels == max_label)[0]
    
    return original_matrix[cluster_indices].T, cluster_indices

def get_pearson_cluster(original_matrix, factor=0.8):
    M, N = original_matrix.shape
    if M == 0:
        return np.array([]), np.array([])

    similarity = np.corrcoef(original_matrix)
    adjacency = similarity >= factor
    n_components, labels = connected_components(csr_matrix(adjacency), directed=False)

    label_counts = Counter(labels)
    max_label = label_counts.most_common(1)[0][0]
    cluster_indices = np.where(labels == max_label)[0]
    return original_matrix[cluster_indices].T, cluster_indices

def get_spearman_cluster(original_matrix, factor=0.8):
    M, N = original_matrix.shape
    if M == 0:
        return np.array([]), np.array([])
    ranks = np.apply_along_axis(rankdata, 1, original_matrix)
    
    similarity = np.corrcoef(ranks)
    similarity = np.nan_to_num(similarity, nan=0.0)
    
    adjacency = similarity >= factor
    n_components, labels = connected_components(csr_matrix(adjacency), directed=False)

    label_counts = Counter(labels)
    max_label = label_counts.most_common(1)[0][0]
    cluster_indices = np.where(labels == max_label)[0]
    return original_matrix[cluster_indices].T, cluster_indices

#######################################################################
#---------------------------------------------------------------------#
#######################################################################

def get_cluster(pred, anno, method='zscore', factor=0.9):
    # from (channels, samples) --> (samples, channels)
    pred = pred.T 
    
    M, N = pred.shape
    if M == 0:
        return np.array([]), np.array([])

    if method == 'zscore':
        zscores = (pred - pred.mean(axis=1, keepdims=True)) / \
                  pred.std(axis=1, keepdims=True, ddof=0)
        similarity = 1 - pairwise_distances(zscores, metric='cosine')
    elif method == 'pearson':
        similarity = np.corrcoef(pred)
    elif method == 'spearman':
        ranks = np.apply_along_axis(rankdata, 1, pred)
        similarity = np.corrcoef(ranks)
        similarity = np.nan_to_num(similarity, nan=0.0)
    else:
        raise ValueError(f"Unsupported method: {method}")

    adjacency = similarity >= factor
    n_components, labels = connected_components(csr_matrix(adjacency), directed=False)

    label_counts = Counter(labels)
    max_label = label_counts.most_common(1)[0][0]
    cluster_indices = np.where(labels == max_label)[0]
    
    filt_pred = pred[cluster_indices].T
    filt_anno = anno.iloc[cluster_indices].reset_index(drop=True)
    
    return filt_pred, filt_anno


def get_matched(pred, anno, keywords, top_cols=None, match_mode="soft"):
    """
    Parameters:
    pred : ndarray (M samples x N channels)
    anno : DataFrame with metadata for N channels
    keywords: dict or list - depends on match_mode
    top_cols: int or None
    match_mode: "soft" (default) or "hard"
    """
    
    # Check if keywords type is correct based on match_mode
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

    # Case-insensitive conversion
    lower_keywords = {k: [x.lower() for x in v] 
                     for k,v in keywords.items()} if match_mode == "hard" else \
                    [str(kw).lower() for kw in keywords]

    # Build matching mask
    if match_mode == "hard":
        mask = pd.Series(True, index=anno.index)
        
        # Celltype matching
        if 'celltype' in lower_keywords:
            cell_mask = anno['celltype'].str.lower().str.contains(
                '|'.join(lower_keywords['celltype']), na=False)
            mask &= cell_mask
            
        # Motif matching (using feature column)
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

    # Apply top_cols filtering
    if top_cols and len(matched_idx) > top_cols:
        matched_idx = matched_idx[:top_cols]

    # Final output
    if len(matched_idx) == 0:
        return np.array([]), pd.DataFrame()
    
    matched_pred = pred[:, matched_idx]
    matched_anno = anno.loc[matched_idx].reset_index(drop=True)
    
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


def plot_basic_heatmap(plot_path, pred, anno, top_cols=None, normalize=True):
    """
    Parameters:
    pred     : ndarray (M samples x N channels)
    anno     : DataFrame with metadata for N channels
    top_cols : int, number of top/bottom differing channels to select
    normalize: bool, whether to normalize columns
    """
    # Input validation
    if not isinstance(pred, np.ndarray) or pred.size == 0:
        raise ValueError("pred must be a non-empty numpy array")
    if not isinstance(anno, pd.DataFrame) or anno.empty:
        raise ValueError("anno must be a non-empty DataFrame")
    if 'index' not in anno.columns:
        raise ValueError("anno must contain an 'index' column")

    plot_pred = pred
    plot_anno = anno

    # Check column limit early
    n_channels = plot_pred.shape[1]
    if top_cols and top_cols > 30:
        raise ValueError("Too much top cols to display")
    if not top_cols and n_channels > 30:
        raise ValueError("No top cols defined, and too much channels to display")

    # Column-wise normalization
    if normalize:
        scaler = MinMaxScaler()
        normalized = np.zeros_like(plot_pred)
        for j in range(plot_pred.shape[1]):  # Iterate over columns
            col_data = plot_pred[:, j].reshape(-1, 1)
            if np.ptp(col_data) == 0:  # Handle constant columns
                normalized[:, j] = 0.5
            else:
                normalized[:, j] = scaler.fit_transform(col_data).flatten()
    else:
        normalized = plot_pred

    # Select top dissimilar columns
    if top_cols and top_cols < normalized.shape[1]:
        col_vars = np.var(normalized, axis=0)
        top_cols_idx = np.argsort(col_vars)[-top_cols:]
        dis_normalized = normalized[:, top_cols_idx]
        dis_anno = plot_anno.iloc[top_cols_idx]

        # Generate dissimilar PDF
        dis_path = plot_path.rsplit('.', 1)[0] + '_similar_min.pdf'
        generate_pdf(dis_normalized, dis_anno, dis_path, "Dissimilar")

    # Select top similar columns
    if top_cols and top_cols < normalized.shape[1]:
        corr_matrix = np.corrcoef(normalized, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, 0)
        avg_corr = np.mean(np.abs(corr_matrix), axis=0)
        top_cols_idx = np.argsort(avg_corr)[-top_cols:]
        sim_normalized = normalized[:, top_cols_idx]
        sim_anno = plot_anno.iloc[top_cols_idx]

        # Generate similar PDF
        sim_path = plot_path.rsplit('.', 1)[0] + '_similar_max.pdf'
        generate_pdf(sim_normalized, sim_anno, sim_path, "Similar")
    
    # Otherwise, generate the total channels
    if not top_cols:
        sim_path = plot_path.rsplit('.', 1)[0] + '_similar_max.pdf'
        generate_pdf(normalized, plot_anno, sim_path, "Total")
    return

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

    # Generate PDF for min difference (largest p-values)
    min_data, min_anno, min_labels = stack_data(bottom_idx)
    min_path = plot_path + '_differ_min.pdf'
    generate_pdf(min_data, min_anno, min_path, "Diff Min", row_labels=min_labels)

    return

#######################################################################
#-----------------------------perturb---------------------------------#
#######################################################################

def perturb_seq(ref_seq, start, end, rep=100):
    if start < 0 or end > len(ref_seq) or start >= end:
        raise ValueError("Error settings for start and end regions")
    alt_seqs = []
    for i in range(rep):
        rand_region = ''.join(random.choice('ATCG') for _ in range(end - start))
        rand_seq = ref_seq[:start] + rand_region + ref_seq[end:]
        alt_seqs.append(rand_seq)
    return alt_seqs

def perturb_boxplot(ref_seqs, alt_seqs, model_list, regions_list, annotation, keywords, work_dir):
    def contains_keywords(celltype):
        return any(keyword in celltype for keyword in keywords)
    
    def remove_outliers(group):
        lower = group["delta"].quantile(0.05)
        upper = group["delta"].quantile(0.95)
        return group[(group["delta"] >= lower) & (group["delta"] <= upper)]
    
    ref_models = ["Malinois", "Basset", "MPRALegNet", "SahuCNN", "CLIPNET", "Puffin", "Basenji2", "Expecto", "Sei", "SpliceAI", "Borzoi", 
                  "Enformer", "SegmentNT", "DanQ", "APARENT2", "DeepDNAshape"]
    ref_npy = ["malinois", "basset", "mpralegnet", "sahucnn", "clipnet", "puffin", "basenji2", "expecto", "sei", "spliceai", "borzoi", 
               "enformer", "segmentnt", "danq", "aparent2", "deepdnashape"]
    
    if not all(model in ref_models for model in model_list):
        raise ValueError("Some models are not in the ref_models.")
    
    indices = [ref_models.index(model) for model in model_list]
    npy_list = [ref_npy[i] for i in indices]
    
    ref_pred_list, alt_pred_list = [], []
    for i in range(len(npy_list)):
        ref_pred = np.load( os.path.join(work_dir, f"ref_preds/pred_{npy_list[i]}.npy") )
        alt_pred = np.load( os.path.join(work_dir, f"alt_preds/pred_{npy_list[i]}.npy") )
        ref_pred_list.append(ref_pred)
        alt_pred_list.append(alt_pred)

    ref_pred_list = np.concatenate(ref_pred_list, axis=1)
    alt_pred_list = np.concatenate(alt_pred_list, axis=1)

    model_order = {model: idx for idx, model in enumerate(model_list)}
    anno_df = annotation[ annotation["model"].isin(model_list) ].drop('Unnamed: 0', axis=1).reset_index(drop=True)
    anno_df['sort_key'] = anno_df['model'].map(model_order)
    anno_df = anno_df.sort_values('sort_key', kind='mergesort')
    anno_df = anno_df.drop('sort_key', axis=1).reset_index(drop=True)
    
    ## selecting relevant celltypes
    idx_list = anno_df[anno_df['celltype'].apply(contains_keywords)].index.tolist()
    ref_pred_list = ref_pred_list[:, idx_list]
    alt_pred_list = alt_pred_list[:, idx_list]
    anno_df = anno_df.loc[idx_list, :].reset_index(drop=True)
    
    delta_list = alt_pred_list - ref_pred_list
    plot_dir = os.path.join(work_dir, "boxplot")
    if not os.path.exists( plot_dir ):
        os.makedirs(plot_dir)
    
    ## grouping 
    for i in tqdm(range(delta_list.shape[1])):
        df_plot = pd.DataFrame( {"delta": delta_list[:, i], "region": regions_list} )
        df_plot = df_plot.groupby("region", group_keys=False).apply(remove_outliers)
        plot_path = os.path.join(plot_dir, f"[{anno_df.loc[i, 'model']}]-[{anno_df.loc[i, 'celltype']}]-[{anno_df.loc[i, 'feature']}].png")
        
        sns.set_style("darkgrid")
        font = {'size' : 10}
        matplotlib.rc('font', **font)
        matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
        fig, ax = plt.subplots(figsize = (8,6), dpi = 300)
        
        order = sorted(set(regions_list), key=regions_list.index)
        ax = sns.boxplot( x="region", y="delta", data=df_plot, order=order,
                         boxprops=dict(alpha=.9), fliersize=1, flierprops={"marker": 'x'}, color="tab:blue")
        h,_ = ax.get_legend_handles_labels()

        plt.xticks(rotation=270)
        ax.set_xlabel("", fontsize=10)
        ax.set_ylabel("Relative activtity", fontsize=10)
    
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.title(f"[{anno_df.loc[i, 'model']}]-[{anno_df.loc[i, 'celltype']}]-[{anno_df.loc[i, 'feature']}]")
        plt.show()
        plt.tight_layout()
        plt.savefig(plot_path)




#######################################################################
#-----------------------------PCA analysis----------------------------#
#######################################################################

from scipy.spatial.distance import cosine

def sorting_by_mahalanobis(uni_pred, control_len, dist_ratio=0.1):
    # Calculate the mean of the control data points    
    cov_matrix = np.cov(uni_pred, rowvar=False)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)

    # Lists to store the Mahalanobis distances for blue and red groups
    samples_pred = uni_pred[:-control_len]
    control_pred = uni_pred[-control_len:]
    total_mahalanobis_distances = []
    
    for i in range(len(samples_pred)):
        dist_list = []
        for j in range(len(control_pred)):
            dist_list.append(distance.mahalanobis(samples_pred[i], control_pred[j], inv_cov_matrix))
            # dist_list.append(cosine(samples_pred[i], control_pred[j]))
        # dist = min(dist_list)
        dist = np.mean(dist_list)
        total_mahalanobis_distances.append(dist)
    
    # sorting samples by mahalanobis distances
    sorted_indices = np.argsort(total_mahalanobis_distances)
    proximal_indices = sorted_indices[:int(len(sorted_indices) * dist_ratio)]
    distal_indices = sorted_indices[-int(len(sorted_indices) * dist_ratio):]

    return proximal_indices, distal_indices


def evaluation_labels_on_mahalanobis(samples_val, input_indices, input_tag="proximal", control_tag="negative", pos_ratio=0.2):
    
    # Sort the values of samples and get the indices
    sorted_indices = np.argsort(samples_val)
    num_of_indices = int(len(sorted_indices) * pos_ratio)
    
    # Select the bottom ratio% (blue group) and top ratio% (red group) based on sorted values
    blue_indices = sorted_indices[:num_of_indices]
    red_indices = sorted_indices[-num_of_indices:]
    green_indices = sorted_indices[num_of_indices:-num_of_indices]
    
    # calculating the overlapping of blue_indices and proximal_indices
    
    if input_tag == "proximal":
        if control_tag == 'pos':
            # 未成功检出
            num_neg_samples = sum(1 for label in input_indices if label in blue_indices)
            num_mid_samples = sum(1 for label in input_indices if label in green_indices)
            # 成功检出
            num_pos_samples = sum(1 for label in input_indices if label in red_indices)
        elif control_tag == 'neg':
            # 成功检出
            num_pos_samples = sum(1 for label in input_indices if label in blue_indices)
            num_mid_samples = sum(1 for label in input_indices if label in green_indices)
            # 未成功检出
            num_neg_samples = sum(1 for label in input_indices if label in red_indices)
    elif input_tag == "distal":
        if control_tag == "pos":
            # 成功检出
            num_pos_samples = sum(1 for label in input_indices if label in blue_indices)
            num_mid_samples = sum(1 for label in input_indices if label in green_indices)
            # 未成功检出
            num_neg_samples = sum(1 for label in input_indices if label in red_indices)
        elif control_tag == "neg":
            # 未成功检出
            num_neg_samples = sum(1 for label in input_indices if label in blue_indices)
            num_mid_samples = sum(1 for label in input_indices if label in green_indices)
            # 成功检出
            num_pos_samples = sum(1 for label in input_indices if label in red_indices)
            
    else:
        print('Error!')
    
    return num_pos_samples, num_mid_samples, num_neg_samples

def plot_ratio_barchart(plot_path, nums_ratios_neg, nums_ratios_mid, nums_ratios_pos, title, n_cos):
    
    num_of_all_samples = nums_ratios_neg[0] + nums_ratios_mid[0] + nums_ratios_pos[0]
    fold = 100 / num_of_all_samples
    blue_vals = np.array(nums_ratios_neg) * fold
    green_vals = np.array(nums_ratios_mid) * fold
    red_vals = np.array(nums_ratios_pos) * fold
    indices = np.arange(len(blue_vals))

    # setting colors (Colorblind-friendly palette)
    color_blue = '#4C72B0'
    color_green = '#55A868'
    color_red = '#C44E52'
    plt.figure(figsize=(10, 6), dpi=300)

    # plotting bars
    bar1 = plt.bar(indices, blue_vals, color=color_blue, edgecolor=color_blue, linewidth=1.5, label='FP samples')
    _ = plt.bar(indices, green_vals, bottom=blue_vals, color=color_green, edgecolor=color_green, linewidth=1.5, label='Boundary samples')
    _ = plt.bar(indices, red_vals, bottom=blue_vals + green_vals, color=color_red, edgecolor=color_red, linewidth=1.5, label='TP samples')
    
    # adding texts for blue regions
    for i, rect in enumerate(bar1):
        height = rect.get_height()
        if height > 5:
            plt.text(rect.get_x() + rect.get_width()/2., height/2,
                    f'{int(blue_vals[i])}%', ha='center', va='center',
                    color='white', fontweight='bold', fontsize=10)

    # settings axis and titles
    plt.xticks(indices, n_cos, fontsize=14)
    plt.ylabel('Ratio', fontsize=14)
    plt.ylim(0, 100)
    plt.axhline(20, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)
    plt.axhline(80, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)
    plt.title(title, fontsize=16, fontweight='bold')

    # legends
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=14, frameon=False)

    # saving
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()




# uni_pred = np.load("/home/hyu/Digital_Platform/scripts/tmp/uni_pred.npy")
# uni_anno = pd.read_csv("/home/hyu/Digital_Platform/scripts/tmp/uni_anno.csv")
# filt_pred, filt_anno = get_cluster(uni_pred, uni_anno, method='zscore', factor=0.9)
# filt_pred, filt_anno = get_matched(filt_pred, filt_anno, keywords=["ELF1"], top_cols=200)
# plot_basic_heatmap(plot_path="./test", pred=filt_pred[:50], anno=filt_anno, top_cols=30)

# pos_pred = np.load("/home/hyu/Digital_Platform_analysis/perturbation/pos_BHLHE40/alt_preds/uni_pred.npy")
# neg_pred = np.load("/home/hyu/Digital_Platform_analysis/perturbation/pos_BHLHE40/ref_preds/uni_pred.npy")
# uni_pred = np.concatenate([pos_pred, neg_pred], axis=0)
# uni_anno = pd.read_csv("/home/hyu/Digital_Platform/scripts/tmp/uni_anno.csv")

# keywords = {"celltype": ["HepG2", "K562"], "motif": ["BHLHE40"]}
# filt_pred, filt_anno = get_matched(uni_pred, uni_anno, keywords=keywords, top_cols=200, match_mode="hard")

# filt_pos_pred = filt_pred[:len(pos_pred)]
# filt_neg_pred = filt_pred[len(pos_pred):]

# K = 10
# plot_differ_heatmap(plot_path="./test", pos_pred=filt_pos_pred, neg_pred=filt_neg_pred, anno=filt_anno, top_cols=30, top_rows=10)

# ./mutation_analysis.sh \
# --pos_pred /home/hyu/Digital_Platform_analysis/perturbation/pos_BHLHE40/alt_preds/uni_pred.npy \
# --neg_pred /home/hyu/Digital_Platform_analysis/perturbation/pos_BHLHE40/ref_preds/uni_pred.npy \
# --outs_path /home/hyu/Digital_Platform/scripts/tmp \
# --csv_path /home/hyu/Digital_Platform_analysis/perturbation/pos_BHLHE40/alt_preds/uni_anno.csv \
# --mode hard \
# --keywords '{"celltype": ["HepG2", "K562"], "motif": ["BHLHE40"]}' \
# --top_rows 10 --top_cols 30