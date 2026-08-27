import os
import sys
import warnings

# Suppress harmless warnings and symlink notices
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from sklearn.utils.extmath import fast_logdet

# Register tqdm with pandas
tqdm.pandas()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.eval_utils import (
    ROC_AUROC,
    compute_pearsonr,
    get_calibrate_ece,
    get_tpr_at_fpr,
    compute_aurac_from_image_df,
    df_to_markdown_bold,
    split_balanced_data,
)
from modules.logdet_utils import (
    normalize_embedding,
    get_generation_embeddings,
    get_quad_entropy,
    get_normalized_entropy,
    compute_eigenscore,
)
from modules.semantic_entropy import (
    get_semantic_ids,
    logsumexp_by_id,
    predictive_entropy_rao,
    EntailmentDeberta,
)

def get_adaptive_alpha_dev_set(image_df, logdet_col, quality_col, dev_ratio=0.1, random_seed=10):
    # Randomly get dev set, and compute the ratio of median logdet and median quality as the adaptive alpha. 
    # Note that we do not use balanced split here as we want the dev set to reflect the real distribution of the data.
    dev_df, test_df = split_balanced_data(image_df, dev_ratio, random_seed, balanced=False)
    quality_values = dev_df[quality_col]
    logdet_values = dev_df[logdet_col]
    denom = quality_values.median()
    if denom == 0 or np.isnan(denom):
        denom = quality_values.mean()
    if denom == 0 or np.isnan(denom):
        denom = 1e-5
    num = np.abs(logdet_values.median())
    if np.isnan(num):
        num = np.abs(logdet_values.mean())
    return float(num / denom)

###### UMPIRE implementation ######
def get_logdet_term(sample, jitter=1e-8):
    embeddings = get_generation_embeddings(sample) # shape: (k, embedding_dim)
    k = embeddings.shape[0]
    kernel = np.dot(embeddings, embeddings.T) # shape: (k, k)
    evals = np.maximum(np.linalg.eigvalsh(kernel), 0.0) + jitter
    return float(1/(2 * k) * np.sum(np.log(evals)))

def compute_umpire(sample, jitter=1e-8, alpha=1, length_normalize=False):
    # get embedding and log-likelihoods
    raw_llh = sample['generations_log_likelihood'] # [[<token1_prob>, <token2_prob>, ...], ...]
    k = len(raw_llh) # number of generations
    # Logdet term
    logdet = get_logdet_term(sample, jitter=jitter)
    # Quadratic entropy term
    # Sequence probs - shape: (k,), length normalized if needed.
    if length_normalize: 
        seq_prob = np.array([np.exp(np.sum(i)/len(i)) for i in raw_llh])
    else:
        seq_prob = np.array([np.exp(np.sum(i)) for i in raw_llh]) 
    incoherence_scores = 1 - seq_prob
    quad_entropy = 1/k * np.sum(incoherence_scores)
    # UMPIRE
    UMPIRE =  logdet + alpha * quad_entropy
    return UMPIRE

###### Semantic Entropy ######
# Adapted from https://github.com/lorenzkuhn/semantic_uncertainty
def compute_semantic_entropy_from_scratch(sample, entailment_model):
    # Get semantic clusters
    cluster_ids = get_semantic_ids(
        strings_list=sample['generations_text'], 
        model=entailment_model, 
        strict_entailment=True,
        example=None # as Deberta model don't need this.
    )
    # Sum log-likelihoods for each token sequence
    llh_sums = [np.sum(llh) for llh in sample['generations_log_likelihood']]
    # Aggregate by cluster
    log_likelihood_by_cluster = logsumexp_by_id(cluster_ids, llh_sums)
    # Compute Semantic Entropy
    semantic_entropy_score = predictive_entropy_rao(log_likelihood_by_cluster)
    return semantic_entropy_score

def compute_semantic_entropy_from_cluster_ids(sample):
    # Get semantic clusters from pre-computed cluster ids in the sample
    cluster_ids = sample['cluster_ids']
    # Sum log-likelihoods for each token sequence
    llh_sums = [np.sum(llh) for llh in sample['generations_log_likelihood']]
    # Aggregate by cluster
    log_likelihood_by_cluster = logsumexp_by_id(cluster_ids, llh_sums)
    # Compute Semantic Entropy
    semantic_entropy_score = predictive_entropy_rao(log_likelihood_by_cluster)
    return semantic_entropy_score

def update_result_based_on_df(image_df, cpc_num_bins=50, ece_num_bins=15, eval_col='is_correct', eval_thresold=0.8, conf_col_to_eval_list=[], unc_col_to_eval_list=[]):
    if 'is_correct' in image_df.columns:
        image_correct_df = image_df.loc[image_df['is_correct'] == True]
        image_wrong_df = image_df.loc[image_df['is_correct'] == False]
        eval_col = 'is_correct'
    elif eval_col == "exact_match":
        image_correct_df = image_df.loc[image_df[eval_col] == 1]
        image_wrong_df = image_df.loc[image_df[eval_col] == 0]
    else:
        image_df['is_correct'] = image_df[eval_col] >= eval_thresold
        image_correct_df = image_df.loc[image_df['is_correct'] == True]
        image_wrong_df = image_df.loc[image_df['is_correct'] == False]
        eval_col = 'is_correct'

    # Sanitize any inf/nan in evaluated columns
    for c in conf_col_to_eval_list + unc_col_to_eval_list:
        if c in image_df.columns:
            clean_s = image_df[c].replace([np.inf, -np.inf], np.nan)
            med = clean_s.median()
            if np.isnan(med):
                med = 0.0
            image_df[c] = clean_s.fillna(med)

    if 'is_correct' in image_df.columns:
        image_correct_df = image_df.loc[image_df['is_correct'] == True]
        image_wrong_df = image_df.loc[image_df['is_correct'] == False]

    result_dict = {}
    for col in conf_col_to_eval_list + unc_col_to_eval_list:
        if col in conf_col_to_eval_list:
            auc = ROC_AUROC(image_wrong_df[col], image_correct_df[col])[-1]
            cece = get_calibrate_ece(image_df, col, eval_col=eval_col, num_bins=ece_num_bins, random_seed=10, calibration_ratio=0.05, model_type='minmax', ece_mode='ece', is_uncertainty=False)
            tpr_at_10_fpr = get_tpr_at_fpr(image_wrong_df[col], image_correct_df[col], 0.1)
            tpr_at_1_fpr = get_tpr_at_fpr(image_wrong_df[col], image_correct_df[col], 0.01)
            aurac = compute_aurac_from_image_df(image_df, col, uncertainty=False, eval_col=eval_col)
            pearsonr = -compute_pearsonr(image_df[col], image_df[eval_col], num_bins=cpc_num_bins)[0]
        else:
            auc = ROC_AUROC(image_correct_df[col], image_wrong_df[col])[-1]
            cece = get_calibrate_ece(image_df, col, eval_col=eval_col, num_bins=ece_num_bins, random_seed=10, calibration_ratio=0.05, model_type='minmax', ece_mode='ece')
            tpr_at_10_fpr = get_tpr_at_fpr(image_correct_df[col], image_wrong_df[col], 0.1)
            tpr_at_1_fpr = get_tpr_at_fpr(image_correct_df[col], image_wrong_df[col], 0.01)
            aurac = compute_aurac_from_image_df(image_df, col, uncertainty=True, eval_col=eval_col)
            pearsonr = compute_pearsonr(image_df[col], image_df[eval_col], num_bins=cpc_num_bins)[0]
            
        result_dict[col] = {
            'auc': auc,
            'cece': cece,
            'pearsonr': pearsonr, 
            'tpr_at_0.1_fpr': tpr_at_10_fpr,
            'tpr_at_0.01_fpr': tpr_at_1_fpr,
            'aurac': aurac
        }
    return result_dict

def evaluate_single_dataset(file_path, output_dir, jitter=1e-8, re_cluster_semantic_entropy=True, entailment_model=None):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Generation file {file_path} not found.")

    print(f"\nLoading generation file: {file_path} ...", flush=True)
    if file_path.endswith(".pt"):
        import torch
        llava_results = torch.load(file_path, map_location="cpu", weights_only=False)
    else:
        with open(file_path, 'rb') as r:
            llava_results = pickle.load(r)

    image_df = pd.DataFrame().from_dict(llava_results)
    print(f"Generation file loaded. Number of samples: {len(image_df)}", flush=True)

    # Column mapping for multipass schema
    if 'rollout_embeddings' in image_df.columns and 'embedding' not in image_df.columns:
        image_df = image_df.rename(columns={
            'rollout_embeddings': 'embedding',
            'rollout_token_logprobs': 'generations_log_likelihood',
            'rollout_texts': 'generations_text',
        })
    if 'is_correct' not in image_df.columns and 'vqa_accuracy' in image_df.columns:
        image_df['is_correct'] = image_df['vqa_accuracy'] >= 0.5

    if 'internal_embedding' in image_df.columns:
        image_df = image_df.rename(columns={'internal_embedding': 'embedding'})
    if 'embedding' not in image_df.columns:
        raise ValueError("The 'embedding' column is missing from the DataFrame.")

    # 1. Normalize embeddings
    print("Normalizing embeddings...", flush=True)
    image_df['norm_embedding'] = image_df['embedding'].apply(normalize_embedding)

    # 2. Compute logdet and quad_entropy for adaptive alpha
    print("Computing logdet and quadratic entropy...", flush=True)
    image_df['logdet'] = image_df.apply(lambda x: get_logdet_term(x, jitter=jitter), axis=1)
    image_df['quad_entropy'] = image_df['generations_log_likelihood'].apply(lambda llh: get_quad_entropy(llh))
    adaptive_alpha = get_adaptive_alpha_dev_set(image_df, logdet_col='logdet', quality_col='quad_entropy', dev_ratio=0.1, random_seed=10)
    print(f"Adaptive alpha: {adaptive_alpha:.4f}", flush=True)

    # 3. Compute UMPIRE
    tqdm.pandas(desc="Computing UMPIRE")
    image_df['umpire'] = image_df.progress_apply(lambda x: compute_umpire(x, alpha=adaptive_alpha, jitter=jitter), axis=1)

    # 4. Length-normalized entropy & EigenScore
    tqdm.pandas(desc="Computing length-normalized entropy")
    image_df['ln_entropy'] = image_df['generations_log_likelihood'].progress_apply(get_normalized_entropy)
    tqdm.pandas(desc="Computing eigenscore")
    image_df['eigen_score'] = image_df.progress_apply(lambda x: compute_eigenscore(x, jitter=jitter), axis=1)

    # 5. Semantic Entropy with DeBERTa
    if re_cluster_semantic_entropy or 'cluster_ids' not in image_df.columns:
        if entailment_model is None:
            print("Initializing EntailmentDeberta model on GPU...", flush=True)
            from modules.semantic_entropy import EntailmentDeberta
            entailment_model = EntailmentDeberta()
        print("Computing Semantic Entropy with EntailmentDeberta...", flush=True)
        tqdm.pandas(desc="Computing semantic entropy")
        image_df['semantic_entropy'] = image_df.progress_apply(lambda x: compute_semantic_entropy_from_scratch(x, entailment_model), axis=1)
    else:
        print("Using pre-computed cluster ids for semantic entropy...", flush=True)
        tqdm.pandas(desc="Computing semantic entropy from pre-computed cluster ids")
        image_df['semantic_entropy'] = image_df.progress_apply(lambda x: compute_semantic_entropy_from_cluster_ids(x), axis=1)

    # 6. Evaluate all metrics
    unc_metrics = ['ln_entropy', 'semantic_entropy', 'eigen_score', 'umpire']
    result_dict = update_result_based_on_df(image_df, cpc_num_bins=50, ece_num_bins=50, unc_col_to_eval_list=unc_metrics)
    result_df = pd.DataFrame().from_dict(result_dict, orient='index')
    result_df = result_df.map(lambda x: round(x, 3) if isinstance(x, (float, int)) else x)

    print("\n" + "=" * 60, flush=True)
    print(f" RESULTS: {os.path.basename(file_path)}", flush=True)
    print("=" * 60, flush=True)
    print(df_to_markdown_bold(result_df), flush=True)
    print("=" * 60 + "\n", flush=True)

    # 7. Save immediately to output directory
    os.makedirs(output_dir, exist_ok=True)
    result_json_file = os.path.join(output_dir, 'umpire_results.json')
    result_csv_file = os.path.join(output_dir, 'umpire_results.csv')
    result_df.to_json(result_json_file, orient='index', indent=4)
    result_df.to_csv(result_csv_file)
    print(f"Saved dataset results to:\n  - {result_csv_file}\n  - {result_json_file}\n", flush=True)

    return result_df, entailment_model


if __name__ == "__main__":
    import time
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Compute UMPIRE and baseline uncertainty metrics.")
    parser.add_argument('--generation_file', type=str, default=None,
                        help='Path to a single generation file (.pt or .pkl)')
    parser.add_argument('--features_dir', type=str, default=None,
                        help='Directory containing multi-pass feature files (e.g. data/features_multipass)')
    parser.add_argument('--arch', type=str, default='m3_llava', choices=['m3', 'mqt', 'm3_llava', 'mqt_llava'],
                        help='Model architecture for features_dir search')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to run. Defaults to all 14 datasets if features_dir is given.')
    parser.add_argument('--output_dir', type=str, default='results/umpire_eval',
                        help='Directory to save output results')
    parser.add_argument('--jitter', type=float, default=1e-8,
                        help='Jitter value for numerical stability in logdet computation')
    parser.add_argument('--re_cluster_semantic_entropy', action='store_true', default=True,
                        help='Whether to compute Semantic Entropy using EntailmentDeberta')
    parser.add_argument('--no_resume', action='store_true',
                        help='If set, re-compute even if umpire_results.csv already exists')
    args = parser.parse_args()

    arch_folder = 'm3_llava' if 'm3' in args.arch.lower() else 'mqt_llava'

    # Single File Mode
    if args.generation_file is not None:
        out_dir = args.output_dir
        evaluate_single_dataset(
            file_path=args.generation_file,
            output_dir=out_dir,
            jitter=args.jitter,
            re_cluster_semantic_entropy=args.re_cluster_semantic_entropy,
        )

    # Multi-Dataset Loop Mode (One by One with Progress Tracking & Auto-Save)
    elif args.features_dir is not None:
        base_dir = os.path.join(args.features_dir, arch_folder, 'temp_0.5')
        if not os.path.exists(base_dir):
            base_dir = os.path.join(args.features_dir, arch_folder)
        if not os.path.exists(base_dir):
            base_dir = args.features_dir

        all_14 = [
            'ai2d', 'chartqa', 'docvqa', 'gqa', 'infographicvqa', 'lego-puzzles',
            'mmbench', 'mmmu', 'pope', 'scienceqa', 'seedbench', 'textvqa',
            'vizwiz-vqa', 'vqav2'
        ]
        target_datasets = args.datasets if args.datasets else all_14
        total_ds = len(target_datasets)

        print("\n" + "#" * 80, flush=True)
        print(f" UMPIRE BASELINE BENCHMARK RUNNER ({arch_folder.upper()})", flush=True)
        print(f" Total Datasets: {total_ds}", flush=True)
        print(f" Feature Base Dir: {base_dir}", flush=True)
        print(f" Output Base Dir: {args.output_dir}/{arch_folder}", flush=True)
        print(f" Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print("#" * 80 + "\n", flush=True)

        entailment_model = None
        cumulative_rows = []
        cumulative_csv_path = os.path.join(args.output_dir, f"{arch_folder}_cumulative_summary.csv")

        # Load existing cumulative rows if available
        if os.path.exists(cumulative_csv_path) and not args.no_resume:
            existing_cum_df = pd.read_csv(cumulative_csv_path)
            cumulative_rows = existing_cum_df.to_dict('records')

        for idx, ds_name in enumerate(target_datasets):
            ds_file = os.path.join(base_dir, f"{ds_name}.pt")
            if not os.path.exists(ds_file):
                ds_file = os.path.join(base_dir, f"{ds_name}.pkl")
            if not os.path.exists(ds_file):
                print(f"[WARN] File for dataset '{ds_name}' not found at {ds_file}. Skipping.", flush=True)
                continue

            ds_out_dir = os.path.join(args.output_dir, arch_folder, ds_name)
            csv_path = os.path.join(ds_out_dir, 'umpire_results.csv')

            if os.path.exists(csv_path) and not args.no_resume:
                print(f"[{idx+1}/{total_ds}] [SKIPPED - ALREADY COMPLETED] {ds_name} (found {csv_path})", flush=True)
                continue

            print("\n" + "=" * 80, flush=True)
            print(f" [{idx+1}/{total_ds}] STARTING DATASET: {ds_name.upper()} at {datetime.now().strftime('%H:%M:%S')}", flush=True)
            print(f" File: {ds_file}", flush=True)
            print("=" * 80, flush=True)

            t_start = time.time()
            try:
                res_df, entailment_model = evaluate_single_dataset(
                    file_path=ds_file,
                    output_dir=ds_out_dir,
                    jitter=args.jitter,
                    re_cluster_semantic_entropy=args.re_cluster_semantic_entropy,
                    entailment_model=entailment_model,
                )
                elapsed = time.time() - t_start
                print(f"[{idx+1}/{total_ds}] [COMPLETED] {ds_name} in {elapsed:.1f}s ({elapsed/60.0:.2f}m) at {datetime.now().strftime('%H:%M:%S')}", flush=True)

                # Record into cumulative summary
                for method_name, row in res_df.iterrows():
                    entry = {'dataset': ds_name, 'method': method_name}
                    entry.update(row.to_dict())
                    cumulative_rows.append(entry)

                # Update cumulative CSV immediately
                pd.DataFrame(cumulative_rows).to_csv(cumulative_csv_path, index=False)
                print(f"Updated cumulative summary: {cumulative_csv_path}\n", flush=True)

            except Exception as e:
                print(f"[{idx+1}/{total_ds}] [ERROR] Failed on {ds_name}: {e}", flush=True)

        print("\n" + "#" * 80, flush=True)
        print(f" ALL DATASETS COMPLETED FOR {arch_folder.upper()}!", flush=True)
        print(f" Cumulative Summary: {cumulative_csv_path}", flush=True)
        print("#" * 80 + "\n", flush=True)

    else:
        print("Please provide either --generation_file <path> or --features_dir <path>", flush=True)

    # # Save the updated DataFrame with uncertainty metrics
    # image_df.to_pickle(os.path.join(args.output_dir, 'image_df_with_uncertainty.pkl'))
