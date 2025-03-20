import json
import matplotlib.patches
import matplotlib.transforms
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from typing import Dict, List
from math import sqrt, ceil
from statistics import median, stdev, quantiles
from argparse import ArgumentParser
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
import hashlib


# sns.set_theme(font="monospace", font_scale=1.5)
sns.set_theme(font_scale=2.75)


def _hash_single_file(json_file: Path) -> bytes:
    """Hash a single file"""
    hash_md5 = hashlib.md5()
    with open(json_file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.digest()

def get_experiment_hash(experiment_path: Path) -> str:
    """Get a hash of all json files in the experiment directory to detect changes"""
    import hashlib
    
    # Sort files to ensure deterministic order
    json_files = sorted(experiment_path.glob("*.json"))
    
    # Compute hashes in parallel
    file_hashes = Parallel(n_jobs=32)(
        delayed(_hash_single_file)(json_file)
        for json_file in json_files
    )
    
    # Combine all hashes deterministically
    final_hash = hashlib.md5()
    for file_hash in file_hashes:
        final_hash.update(file_hash)
        
    return final_hash.hexdigest()

def load_cache(cache_file: Path) -> tuple[dict, dict]:
    """Load cached results and experiment hashes"""
    if not cache_file.exists():
        return {}, {}
    
    with open(cache_file, "r") as f:
        cache = json.load(f)
    return cache.get("results", {}), cache.get("hashes", {})

def save_cache(cache_file: Path, results: dict, hashes: dict):
    """Save results and experiment hashes to cache file"""
    with open(cache_file, "w") as f:
        json.dump({
            "results": results,
            "hashes": hashes
        }, f)


def read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _extract_name_from_path(path: str) -> str:
    """Example of expected path:
        'metadataholistic_textdisabled_rankscore_task158_search_model_Qwen2.5-7B-Instruct_nshot_5_numnodes_9_numsamples_1000.json'
    Expected output: task158.
    """
    return [part for part in path.split("_") if "task" in part][0]

def _get_model_size(model_name: str) -> float:
    if "1B" in model_name:
        return 1    
    elif "1.5B" in model_name:
        return 1.5
    elif "2b" in model_name or "2B" in model_name:
        return 2
    elif "3B" in model_name:
        return 3
    elif "7B" in model_name:
        return 7
    elif "8B" in model_name:
        return 8
    elif "9b" in model_name or "9B" in model_name:
        return 9

def plot_and_save_hist(values: np.ndarray, experiment_name: str, savepath: str | Path):
    plt.hist(values, bins=ceil(sqrt(len(values))))
    plt.title(f"{experiment_name} spreads\non {len(values)} tasks from Natural Instructions")
    plt.xlabel("Spread (max accuracy - min accuracy)")
    plt.savefig(savepath, dpi=550, bbox_inches="tight")
    plt.close()


def _compute_unbalanced_metrics(node: List) -> Dict[str, float]:
    entries = node[1]
    y_true = [e["answer"] for e in entries]
    y_pred = [e["generation"] for e in entries]
    
    metrics = {}
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["matthews_corrcoef"] = matthews_corrcoef(y_true, y_pred)
    return metrics


def _process_single_path(path: str) -> Dict:
    result_json = read_json(path)

    # Check if metrics are already computed
    if "matthews_corrcoefs" in result_json and "balanced_accuracies" in result_json:
        matthews_corrs = result_json["matthews_corrcoefs"]
        balanced_accuracies = result_json["balanced_accuracies"]
    else:
        # Compute metrics
        metadata = result_json["metadata"]
        nodes = metadata["nodes"].values() if "nodes" in metadata else metadata["ensembles"].values()
        nodes_metrics = []
        for node in nodes:
            nodes_metrics.append(_compute_unbalanced_metrics(node))
        
        balanced_accuracies = [node["balanced_accuracy"] for node in nodes_metrics]
        matthews_corrs = [node["matthews_corrcoef"] for node in nodes_metrics]
        
        # Save metrics in result_json
        result_json["matthews_corrcoefs"] = matthews_corrs
        result_json["balanced_accuracies"] = balanced_accuracies
        
        # Write back to file
        with open(path, "w") as f:
            json.dump(result_json, f)

    format_to_acc = result_json["all_structured_prompt_formats_accuracies"]

    best_format = None
    best_acc = None 
    best_n = None
    worst_format = None
    worst_acc = None
    worst_n = None

    for format, acc_tuple in format_to_acc.items():
        acc, one_minus_acc, n = acc_tuple

        if best_acc is None or acc > best_acc:
            best_format = format
            best_acc = acc
            best_n = n
        
        if worst_acc is None or acc < worst_acc:
            worst_format = format
            worst_acc = acc
            worst_n = n

    median_accuracy = median([acc_tuple[0] for acc_tuple in format_to_acc.values()])
    accuracy_std = stdev([acc_tuple[0] for acc_tuple in format_to_acc.values()])

    # first format in dict is assumed to be the default format
    default_format = next(iter(format_to_acc))
    default_accuracy = format_to_acc[default_format][0]
    changes = [default_accuracy - acc_tuple[0] for acc_tuple in format_to_acc.values()]
    changes.pop(0) # remove first 0

    matt25, matt50, matt75 = quantiles(matthews_corrs, n=4)

    return {
        "task": _extract_name_from_path(str(path)),
        "spread": best_acc - worst_acc,
        "std": accuracy_std,
        "best_accuracy": best_acc,
        "worst_accuracy": worst_acc,
        "median_accuracy": median_accuracy,
        "default_accuracy": default_accuracy,
        "mean_drop": sum(changes) / len(changes) if len(changes) > 0 else 0,
        "median_balanced_accuracy": median(balanced_accuracies),
        "spread_balanced": max(balanced_accuracies) - min(balanced_accuracies),
        "std_balanced": stdev(balanced_accuracies),
        "median_matthews_corrcoef": matt50,
        "upper_error_matthews": matt75 - matt50,
        "lower_error_matthews": matt50 - matt25,
        "spread_matthews": max(matthews_corrs) - min(matthews_corrs),
        "std_matthews": stdev(matthews_corrs),
        "best_format": best_format,
        "worst_format": worst_format,
        "best_n": best_n,
        "worst_n": worst_n,
    }

def collect_spreads(paths: List[str]) -> pd.DataFrame:
    
    records = Parallel(n_jobs=32)(
        delayed(_process_single_path)(path) 
        for path in paths
        # for path in tqdm(paths, desc="paths")
    )
    
    return pd.DataFrame.from_records(records)


def plot_gaussian(mean, cov, ax, color, n_std=3.0):
    """
    Plots a Gaussian ellipse with given mean and covariance.
    Input:
    - mean: vector of means (mean_accuracy, mean_spread)
    - cov: covariance matrix
    - ax: ax object to add the ellipse to
    - color: color of the ellipse
    - n_std: number of standard deviations to scale the ellipse
    Output:
    - ax: with added ellipse
    """
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Calculate the angle of the ellipse
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    
    # Calculate the width and height of the ellipse
    width, height = 2 * np.sqrt(eigenvalues)
    
    # Create the ellipse
    ellipse = matplotlib.patches.Ellipse(xy=mean, width=n_std * width, height=n_std * height, angle=angle, 
                    edgecolor="none", fc=color, lw=2, alpha=0.3)

    # Add the ellipse to the plot
    ax.add_patch(ellipse)
    return ax


def find_pareto_front(df, col1, col2):
    """
    Find the Pareto front in a DataFrame with two columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    col1 (str): The name of the first column (higher is better).
    col2 (str): The name of the second column (lower is better).
    
    Returns:
    pd.DataFrame: A DataFrame containing the rows that form the Pareto front.
    """
    # Sort the DataFrame by col1 in descending order and col2 in ascending order
    df_sorted = df.sort_values(by=[col1, col2], ascending=[False, True])
    
    pareto_front = []
    min_col2 = float('inf')
    
    for _, row in df_sorted.iterrows():
        if row[col2] < min_col2:
            pareto_front.append(row)
            min_col2 = row[col2]
    
    # Convert the list of rows back to a DataFrame
    pareto_front_df = pd.DataFrame(pareto_front)
    
    return pareto_front_df


def create_ranking_table(df, metric, higher_is_better=True) -> str:
    # For each model and setting combination, rank the methods
    df = df.copy()

    # Convert rankings so that 1 is best (whether higher or lower values are better)
    df["rank"] = df.groupby(["model", "setting", "task"])[[metric]].rank(ascending=not higher_is_better)
    
    # Average ranks across models for each setting and method
    avg_ranks = df.groupby(['setting', 'method'])['rank'].mean().unstack()
    
    # Convert to LaTeX table
    latex_table = avg_ranks.to_latex(
        float_format=lambda x: '{:.2f}'.format(x),
        bold_rows=True,
        caption=f"Average rankings of methods across models (1 is best)\nMetric: {metric}",
        label="tab:method_rankings"
    )
    
    return latex_table


def create_accuracy_std_table(df):
    # Calculate mean of median_accuracy and std for each combination
    stats = df.groupby(["setting", "model", "method"], observed=True)[["median_accuracy", "std"]].mean()
    
    # Format the accuracy ± std string
    stats["formatted"] = stats.apply(
        lambda x: f"${x['median_accuracy']:.4f} \pm {2*x['std']:.4f}$", 
        axis=1
    )
    
    # Pivot to get desired table format
    table = stats["formatted"].unstack(level=[1, 2])
    
    # Convert to LaTeX table with custom formatting
    latex_table = table.to_latex(
        multicolumn=True,
        multicolumn_format='c',
        bold_rows=False,
        caption="Accuracy (mean ± 2*std) for each setting, model and method combination",
        label="tab:accuracy_std"
    )
    
    return latex_table

def create_accuracy_std_table_single_method(df):
    # Calculate mean of median_accuracy and std for each combination
    stats = df.groupby(["setting", "model"], observed=True)[["median_accuracy", "std"]].mean()
    
    # Format the accuracy ± std string
    stats["formatted"] = stats.apply(
        lambda x: f"${x['median_accuracy']:.4f} \pm {2*x['std']:.4f}$", 
        axis=1
    )
    
    # Pivot to get desired table format
    table = stats["formatted"].unstack()
    
    # Convert to LaTeX table with custom formatting
    latex_table = table.to_latex(
        bold_rows=False,
        caption="Accuracy (mean ± 2*std) for each setting and model.",
        label="tab:accuracy_std"
    )
    
    return latex_table



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root-dir", default="exp")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--num-nodes", type=int, default=10)
    parser.add_argument("-e", "--experiment-names", nargs="+")

    args = parser.parse_args()
    return args


N_SELECTED_TASKS = 52

def main():
    args = parse_args()

    total_df = []
    cache_file = Path(args.root_dir) / "analysis_cache.json"
    cached_results, cached_hashes = load_cache(cache_file)
    
    Path(args.image_dir).mkdir()

    experiment_names = args.experiment_names if args.experiment_names else \
        sorted([filename.name for filename in Path(args.root_dir).iterdir() if filename.is_dir()])

    experiment_names = [name for name in experiment_names if 
        # not "1ksteps" in name and
        # not "superclear" in name and
        # not "iid" in name and
        # not "simpleanswers-batch-calibration" in name
        not "zeroshot" in name and 
        not "test" in name and 
        not "augs" in name and 
        not "debug" in name and 
        not "batch-calibration-probs" in name and 
        not "rank1" in name and
        not "default" in name and 
        not "ensemble" in name and 
        not "sensitivity-aware-decoding" in name and
        not "batch-calibration" in name and
        # ("lora" in name  or "iidx2-no-chat" in name) and 
        not "exact-match" in name and 
        not "compositional" in name and
        # "lora" in name and
        not "cross" in name and
        not "unbalanced" in name
        # "consistency" in name
    ]
    experiment_names = [name for name in experiment_names if ("2-shot" in name and "lora" not in name) or ("0-shot" in name and "lora" in name)]
    experiment_names = [name for name in experiment_names if "no-chat-template" in name]
    experiment_names = [name for name in experiment_names if ("instruct" in name.lower() or "it" in name)]
    # experiment_names = [name for name in experiment_names if "Llama" in name or "Qwen2.5" in name or "gemma-2" in name]
    experiment_names = [name for name in experiment_names if "Llama-3.2-3B" in name]
    # experiment_names = [name for name in experiment_names if "Llama-3.2" in name or "Qwen2.5-1.5B" in name or "Qwen2.5-3B" in name]
    # experiment_names = [name for name in experiment_names if "3B" in name or "1.5B" in name or "1B" in name]

    for n in experiment_names:
        print("\t", n)

    results = {}
    current_hashes = {}
    
    for experiment_name in tqdm(experiment_names, desc="models"):
        subdir = Path(args.root_dir) / experiment_name
        print(experiment_name)
        
        # Check if experiment needs recomputation
        current_hash = get_experiment_hash(subdir)
        current_hashes[experiment_name] = current_hash
        
        if False and (experiment_name in cached_results and 
                experiment_name in cached_hashes and 
                cached_hashes[experiment_name] == current_hash):
            # Use cached results
            df = pd.DataFrame.from_dict(cached_results[experiment_name])
            total_df.append(df)
            results[experiment_name] = cached_results[experiment_name]
            continue

        model_name, tail = experiment_name.split("---")

        num_nodes = 5 * (args.num_nodes + 1) - 1 if "ensemble" in experiment_name else args.num_nodes
        pattern = f"metadataholistic*{model_name}*.json"
        evaluated_tasks_result_paths = list(subdir.glob(pattern))
        if len(evaluated_tasks_result_paths) != N_SELECTED_TASKS:
            print(f"{model_name=}")
            print(f"ATTENTION: {experiment_name} is only evaluated on {len(evaluated_tasks_result_paths)} tasks")
        df = collect_spreads(evaluated_tasks_result_paths)

        if "exact-match" in tail:
            df["setting"] = "exact-match"
        elif "unbalanced" in tail:
            df["setting"] = "unbalanced"
        elif "cross" in tail:
            df["setting"] = "cross-domain"
        elif "compositional" in tail:
            df["setting"] = "compositional"
        else:
            df["setting"] = "uniform"

        # Process experiment handle
        tail = tail.replace("no-chat-template-", "")
        tail = tail.replace("unbalanced-", "")
        tail = tail.replace("exact-match-", "")
            
        if "lora" in model_name:
            tail = tail.replace("0-shot", "")
            tail = tail.replace("response-only-", "")
            tail = tail.replace("iidx2-consistency-", " (consistency)")
            tail = tail.replace("iidx2-", "")
            # tail = tail.replace("iidx2-", " (uniform)")
            tail = tail.replace("cross-", " (cross-domain)")
            tail = tail.replace("iid-", " (cross-domain)")
            tail = tail.replace("separator-space-", "")
            tail = tail.replace("compositional-", " (compositional)")

            tail = tail.replace("32augs-", " (32 augs)")
            tail = tail.replace("2augs-", " (2 augs)")
            tail = tail.replace("8augs-", " (8 augs)")
            tail = tail.replace("16augs-", " (16 augs)")

            tail = "LoRA" + tail 

            # if tail == "LoRA":
            #     tail = "LoRA (4 augs)"
            model_name = model_name.replace("_lora", "")
        else:
            tail = tail.replace("2-shot", "")
            tail = tail.replace("iidx2-split-", "")
            tail = tail.replace("iidx2-", "")
            tail = tail.replace("iid-", "(new) ")
            tail = tail.replace("batch-calibration-", "BC")
            tail = tail.replace("sensitivity-aware-decoding-", "SAD")
            tail = tail.replace("template-ensembles-", "TE")
            if tail == "" or tail == "-":
                tail = "FS"

        # Process model name
        model_name = model_name.replace("-Instruct", "")\
            .replace("-it", "")\
            .replace("Meta-", "")\
            .replace("2.5", " 2.5")\
            .replace("-", " ")\
            .title()

        df["experiment"] = f"{model_name:>30}{tail:>40}"
        df["method"] = tail
        df["model"] = model_name
        df["size"] = _get_model_size(model_name)

        df.to_csv(subdir / "spreads.csv")
        plot_and_save_hist(df["spread"].values, experiment_name, subdir / "spreads.png")
        total_df.append(df)
        
        # Cache the results
        results[experiment_name] = df.to_dict('records')
    
    # Save updated cache
    save_cache(cache_file, results, current_hashes)
    
    total_df = pd.concat(total_df)

    # upper error = 75th percentile - 50th percentile
    # lower error = 50th percentile - 25th percentile
    # upper error + lower error = 75th percentile - 25th percentile = IQR
    total_df["iqr_matthews"] = total_df["upper_error_matthews"] + total_df["lower_error_matthews"]

# Create latex tables
    for family in ("Llama", "Qwen", "Gemma"):
        latex_table = create_accuracy_std_table_single_method(total_df[total_df["model"].str.contains(family)])
        with open(f"{args.image_dir}/accuracy_std_table_{family}.tex", "w") as f:
            f.write(latex_table)

    metrics = ("median_matthews_corrcoef", "iqr_matthews")
    directions = (True, False)
    for metric, higher_is_better in zip(metrics, directions):
        latex_table = create_ranking_table(total_df, metric, higher_is_better)
        with open(f"{args.image_dir}/rankings_{metric}.tex", "w") as f:
            f.write(latex_table)
    # all_methods = ["FS", "BC", "SAD", "TE", "LoRA (uniform)", "LoRA (compositional)", "LoRA (cross-domain)", "LoRA (consistency)"]
    # all_methods = ["FS", "BC", "SAD", "TE", "LoRA", "LoRA (compositional)", "LoRA (cross-domain)", "LoRA (consistency)", "LoRA (consistency)-beta30.0"]
    all_methods = ["FS", "BC", "SAD", "TE", "LoRA", "LoRA (consistency)", "LoRA (consistency)-beta30.0", "LoRA (consistency)-beta100.0"]
    # all_methods = ["FS", "LoRA (2 augs)", "LoRA (4 augs)", "LoRA (8 augs)", "LoRA (16 augs)", "LoRA (32 augs)"]

    unique_methods = total_df["method"].unique()
    print(f"{unique_methods=}")
    for m in unique_methods:
        if m not in all_methods:
            print(f"{m} is not in the list of methods")
    method2color = dict(zip(all_methods, sns.color_palette("Set2")))

    total_df["method"] = pd.Categorical(total_df["method"], categories=all_methods)
    total_df = total_df.sort_values(["method", "size"])
    total_df["method"] = total_df["method"].cat.remove_unused_categories()

    # total_df = total_df[total_df["setting"] == "exact-match"]

    default_figsize = (22, 9)
    default_legend_loc = (1.0, 0.5)
    default_ncol = 1
    default_corner = "center left"

    # default_legend_loc = (0.5, -0.3)
    # default_ncol = 5
    # default_corner = "center"

# Spread
    x_axis = "model"
    plt.figure(figsize=default_figsize)
    ax = sns.barplot(data=total_df, x=x_axis, y="spread", hue="method" if x_axis == "model" else None, errorbar=None, palette=method2color)
    plt.xticks(rotation=15, ha="right")
    plt.xlabel("")
    plt.ylabel("Spread over prompts", labelpad=25)
    plt.title("Methods' spread over prompts on different models\n(lower is better)")
    if ax.legend_ is not None:
        sns.move_legend(ax, default_corner, bbox_to_anchor=default_legend_loc, ncol=default_ncol, title="Method")
    plt.savefig(f"{args.image_dir}/clustered_spread_barplot.png", dpi=550, bbox_inches="tight")
    plt.close()

# Clustered barplot of accuracy with errorbars
    plt.figure(figsize=default_figsize)
    ax = sns.barplot(data=total_df, x=x_axis, y="median_accuracy", hue="method" if x_axis == "model" else None, errorbar=None, palette=method2color)
    plt.xticks(rotation=15, ha="right")

    mean_stds = total_df.groupby(["model", "method"], observed=True)["std"].mean()
    for p, mean_std in zip(ax.patches, mean_stds):
        x = p.get_x()
        w = p.get_width()
        h = p.get_height()
        plt.errorbar(x + w / 2, h, yerr=2 * mean_std, fmt="none", linewidth=2, color="black", capsize=4)

    plt.xlabel("")
    plt.ylabel("Accuracy", labelpad=25)
    plt.title("Methods' performance on different models")
    if ax.legend_ is not None:
        sns.move_legend(ax, default_corner, bbox_to_anchor=default_legend_loc, ncol=default_ncol, title="Method")
    plt.savefig(f"{args.image_dir}/clustered_barplot.png", dpi=550, bbox_inches="tight")
    plt.close()

# Paired spread and accuracy plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15), sharex=True)
    
    # Top plot (spread)
    sns.barplot(data=total_df, x=x_axis, y="spread", hue="method" if x_axis == "model" else None, errorbar=None, palette=method2color, ax=ax1)
    # ax1.set_xticklabels([])  # Remove x tick labels from top plot
    ax1.tick_params(axis="x", which="both", length=0.)
    ax1.set_xlabel("")
    ax1.set_ylabel("Spread over prompts", labelpad=25)
    ax1.set_title("Methods' spread over prompts on different models\n(lower is better)", fontsize=36, pad=30)
    ax1.get_legend().remove()  # Remove legend from top plot
    
    # Bottom plot (accuracy with errorbars)
    sns.barplot(data=total_df, x=x_axis, y="median_accuracy", hue="method" if x_axis == "model" else None, errorbar=None, palette=method2color, ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=15, ha="right")
    ax2.set_xlabel("")
    ax2.set_ylabel("Accuracy", labelpad=25)
    ax2.set_title("Methods' performance on different models", fontsize=36, pad=30)
    
    # Add error bars to bottom plot
    mean_stds = total_df.groupby(["model", "method"], observed=True)["std"].mean()
    for p, mean_std in zip(ax2.patches, mean_stds):
        x = p.get_x()
        w = p.get_width()
        h = p.get_height()
        ax2.errorbar(x + w / 2, h, yerr=2 * mean_std, fmt="none", linewidth=2, color="black", capsize=4)
    
    # Move legend to the right center
    legend = ax2.get_legend()
    fig.legend(
        legend.legend_handles, 
        [t.get_text() for t in legend.get_texts()],
        title="Method",
        bbox_to_anchor=(1.0, 0.5),
        loc="center left"
    )
    legend.remove()  # Remove the original legend
    
    plt.tight_layout()
    plt.savefig(f"{args.image_dir}/paired_clustered_spread_and_accuracy_with_errorbars.png", 
                dpi=550, 
                bbox_inches="tight")
    plt.close()

# Paired spread and Matthews correlation coefficient plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 15), sharex=True)
    
    # Top plot (spread Matthews)
    sns.barplot(data=total_df, x=x_axis, y="spread_matthews", hue="method" if x_axis == "model" else None, errorbar=None, palette=method2color, ax=ax1)
    ax1.tick_params(axis="x", which="both", length=0.)
    ax1.set_xlabel("")
    ax1.set_ylabel("Spread (Matthews)", labelpad=25)
    ax1.set_title("Methods' spread (Matthews) over prompts on different models\n(lower is better)", fontsize=36, pad=30)
    ax1.get_legend().remove()
    
    # Bottom plot (Matthews correlation coefficient with errorbars)
    sns.barplot(data=total_df, x=x_axis, y="median_matthews_corrcoef", hue="method" if x_axis == "model" else None, errorbar=None, palette=method2color, ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=15, ha="right")
    ax2.set_xlabel("")
    ax2.set_ylabel("Matthews correlation", labelpad=25)
    ax2.set_title("Methods' performance on different models", fontsize=36, pad=30)
    
    # Add asymmetric error bars to bottom plot
    mean_upper_error = total_df.groupby(["model", "method"], observed=True)["upper_error_matthews"].mean()
    mean_lower_error = total_df.groupby(["model", "method"], observed=True)["lower_error_matthews"].mean()
    for p, upper, lower in zip(ax2.patches, mean_upper_error, mean_lower_error):
        x = p.get_x()
        w = p.get_width()
        h = p.get_height()
        ax2.errorbar(x + w / 2, h, yerr=[[lower], [upper]], fmt="none", linewidth=2, color="black", capsize=4)
    
    # Move legend to the right center
    legend = ax2.get_legend()
    fig.legend(
        legend.legend_handles, 
        [t.get_text() for t in legend.get_texts()],
        title="Method",
        bbox_to_anchor=(1.0, 0.5),
        loc="center left"
    )
    legend.remove()
    
    plt.tight_layout()
    plt.savefig(f"{args.image_dir}/paired_clustered_spread_and_matthews_with_errorbars.png", 
                dpi=550, 
                bbox_inches="tight")
    plt.close()


# Std
    plt.figure(figsize=default_figsize)
    ax = sns.barplot(data=total_df, x=x_axis, y="std", hue="method" if x_axis == "model" else None, errorbar=None, palette=method2color)
    plt.xticks(rotation=15, ha="right")
    plt.xlabel("")
    plt.ylabel("Standard deviation over prompts", labelpad=25)
    plt.title("Methods' standard deviation over prompts on different models\n(lower is better)")
    if ax.legend_ is not None:
        sns.move_legend(ax, default_corner, bbox_to_anchor=default_legend_loc, ncol=default_ncol, title="Method")
    plt.savefig(f"{args.image_dir}/clustered_std_barplot.png", dpi=550, bbox_inches="tight")
    plt.close()

# Pareto: accuracy/spread
    plt.figure(figsize=(20, 9))
    averaged_by_tasks = pd.concat([
        total_df.groupby(["method", "model"], observed=True)[["median_accuracy", "spread"]].mean(),
        total_df.groupby(["method", "model"], observed=True)[["method", "model", "size"]].first()
    ], axis=1)

    ax = sns.scatterplot(
        data=averaged_by_tasks,
        y="median_accuracy",
        x="spread",
        hue="method",
        style="model",
        size="size",
        palette=method2color,
        sizes=(400, 1000),
        edgecolor="black",
        linewidth=1,
    )
    plt.xlabel("Spread over prompts", labelpad=10)
    plt.ylabel("Accuracy", labelpad=15)
    plt.title("Accuracy and spread trade-off")

    for method in total_df['method'].unique():
        method_data = averaged_by_tasks[averaged_by_tasks['method'] == method]
        mean = method_data[['spread', 'median_accuracy',]].mean().values
        cov = method_data[['spread', 'median_accuracy', ]].cov().values
        
        plot_gaussian(mean, cov, ax, method2color[method], n_std=1.5)

    front = find_pareto_front(averaged_by_tasks, "median_accuracy", "spread")

    ax.plot(front["spread"], front["median_accuracy"], linewidth=2, linestyle="--", color="black")

    h, l = ax.get_legend_handles_labels()
    method_border = total_df.method.nunique() + 1
    size_border = total_df["method"].nunique() + 1 + total_df["size"].nunique() + 1
    # legend1 = plt.legend(h[1:method_border], l[1:method_border], loc="center", bbox_to_anchor=(0.5, -0.3), ncol=6, title="Method", markerscale=4)
    legend1 = plt.legend(h[1:method_border], l[1:method_border], loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=1, title="Method", markerscale=4, fontsize=28, title_fontsize=32)
    legend2 = plt.legend(h[size_border:], l[size_border:], loc="upper right", bbox_to_anchor=(1.3, 1.0), ncols=1, title="Model", markerscale=4, fontsize=28, title_fontsize=32)
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    plt.savefig(f"{args.image_dir}/pareto_accuracy_spread.png", dpi=550, bbox_inches="tight", bbox_extra_artists=[legend1, legend2])
    plt.close()


# Pareto: accuracy/spread
    plt.figure(figsize=(20, 9))
    averaged_by_tasks = pd.concat([
        total_df.groupby(["method", "model"], observed=True)[["median_accuracy", "spread"]].mean(),
        total_df.groupby(["method", "model"], observed=True)[["method", "model", "size"]].first()
    ], axis=1)

    ax = sns.scatterplot(
        data=averaged_by_tasks,
        y="median_accuracy",
        x="spread",
        hue="method",
        style="model",
        size="size",
        palette=method2color,
        sizes=(400, 1000),
        edgecolor="black",
        linewidth=1,
    )
    plt.xlabel("Spread over prompts", labelpad=10)
    plt.ylabel("Accuracy", labelpad=15)
    plt.title("Accuracy and spread trade-off")

    for method in total_df['method'].unique():
        method_data = averaged_by_tasks[averaged_by_tasks['method'] == method]
        mean = method_data[['spread', 'median_accuracy',]].mean().values
        cov = method_data[['spread', 'median_accuracy', ]].cov().values
        
        plot_gaussian(mean, cov, ax, method2color[method], n_std=1.5)

    front = find_pareto_front(averaged_by_tasks, "median_accuracy", "spread")

    ax.plot(front["spread"], front["median_accuracy"], linewidth=2, linestyle="--", color="black")

    h, l = ax.get_legend_handles_labels()
    method_border = total_df.method.nunique() + 1
    size_border = total_df["method"].nunique() + 1 + total_df["size"].nunique() + 1
    # legend1 = plt.legend(h[1:method_border], l[1:method_border], loc="center", bbox_to_anchor=(0.5, -0.3), ncol=6, title="Method", markerscale=4)
    legend1 = plt.legend(h[1:method_border], l[1:method_border], loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=1, title="Method", markerscale=4, fontsize=28, title_fontsize=32)
    legend2 = plt.legend(h[size_border:], l[size_border:], loc="upper right", bbox_to_anchor=(1.3, 1.0), ncols=1, title="Model", markerscale=4, fontsize=28, title_fontsize=32)
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    plt.savefig(f"{args.image_dir}/pareto_matthews_spread.png", dpi=550, bbox_inches="tight", bbox_extra_artists=[legend1, legend2])
    plt.close()


# Clustered barplot of BALANCED accuracy with errorbars
    plt.figure(figsize=default_figsize)
    ax = sns.barplot(data=total_df, x=x_axis, y="median_balanced_accuracy", hue="method" if x_axis == "model" else None, errorbar=None, palette=method2color)
    plt.xticks(rotation=15, ha="right")

    mean_stds = total_df.groupby(["model", "method"], observed=True)["std_balanced"].mean()
    for p, mean_std in zip(ax.patches, mean_stds):
        x = p.get_x()
        w = p.get_width()
        h = p.get_height()
        plt.errorbar(x + w / 2, h, yerr=2 * mean_std, fmt="none", linewidth=2, color="black", capsize=4)

    plt.xlabel("")
    plt.ylabel("Balanced accuracy", labelpad=25)
    plt.title("Methods' performance on different models")
    if ax.legend_ is not None:
        sns.move_legend(ax, default_corner, bbox_to_anchor=default_legend_loc, ncol=default_ncol, title="Method")
    plt.savefig(f"{args.image_dir}/clustered_barplot_balanced.png", dpi=550, bbox_inches="tight")
    plt.close()

# BALANCED Spread
    plt.figure(figsize=default_figsize)
    ax = sns.barplot(data=total_df, x=x_axis, y="spread", hue="method" if x_axis == "model" else None, errorbar=None, palette=method2color)
    plt.xticks(rotation=15, ha="right")
    plt.xlabel("")
    plt.ylabel("Spread of balanced accuracy over prompts", labelpad=25)
    plt.title("Methods' balanced accuracy spread over prompts on different models\n(lower is better)")
    if ax.legend_ is not None:
        sns.move_legend(ax, default_corner, bbox_to_anchor=default_legend_loc, ncol=default_ncol, title="Method")
    plt.savefig(f"{args.image_dir}/clustered_spread_barplot_balanced.png", dpi=550, bbox_inches="tight")
    plt.close()


# Clustered barplot of MATTHEWS CORRELATION with errorbars
    plt.figure(figsize=default_figsize)
    ax = sns.barplot(data=total_df, x=x_axis, y="median_matthews_corrcoef", hue="method" if x_axis == "model" else None, errorbar=None, palette=method2color)
    plt.xticks(rotation=15, ha="right")

    mean_upper_error = total_df.groupby(["model", "method"], observed=True)["upper_error_matthews"].mean()
    mean_lower_error = total_df.groupby(["model", "method"], observed=True)["lower_error_matthews"].mean()
    for p, mean_up, mean_low in zip(ax.patches, mean_upper_error, mean_lower_error):
        x = p.get_x()
        w = p.get_width()
        h = p.get_height()
        # print(f"DEBUG === {mean_low=}, {mean_up=}")
        plt.errorbar(x + w / 2, h, yerr=[[mean_low], [mean_up]], fmt="none", linewidth=2, color="black", capsize=4)

    plt.xlabel("")
    plt.ylabel("Matthews correlation", labelpad=25)
    plt.title("Methods' performance on different models")
    if ax.legend_ is not None:
        sns.move_legend(ax, default_corner, bbox_to_anchor=default_legend_loc, ncol=default_ncol, title="Method")
    plt.savefig(f"{args.image_dir}/clustered_barplot_matthews.png", dpi=550, bbox_inches="tight")
    plt.close()

# Side-by-side MATTHEWS CORRELATION with errorbars
    print(total_df)
    plt.figure(figsize=default_figsize)
    ax = sns.barplot(data=total_df, x="setting", y="median_matthews_corrcoef", hue="method" if x_axis == "model" else None, errorbar=None, palette=method2color)
    # plt.xticks(rotation=15, ha="right")

    mean_upper_error = total_df.groupby(["setting", "method"], observed=True)["upper_error_matthews"].mean()
    mean_lower_error = total_df.groupby(["setting", "method"], observed=True)["lower_error_matthews"].mean()
    for p, mean_up, mean_low in zip(ax.patches, mean_upper_error, mean_lower_error):
        x = p.get_x()
        w = p.get_width()
        h = p.get_height()
        # print(f"DEBUG === {mean_low=}, {mean_up=}")
        plt.errorbar(x + w / 2, h, yerr=[[mean_low], [mean_up]], fmt="none", linewidth=2, color="black", capsize=4)

    plt.xlabel("")
    plt.ylabel("Matthews correlation", labelpad=25)
    plt.title("Uniform vs. unbalanced setting, Llama 3.1 8B")
    if ax.legend_ is not None:
        sns.move_legend(ax, default_corner, bbox_to_anchor=default_legend_loc, ncol=default_ncol, title="Method")
    plt.savefig(f"{args.image_dir}/clustered_barplot_matthews_setting.png", dpi=550, bbox_inches="tight")
    plt.close()


    # plt.figure(figsize=(12, 8))
    # sns.boxplot(data=total_df, x="mean_drop", y="experiment", hue="model", whis=(5, 95), palette=model2color, legend=False)
    # plt.xlabel(f"Whiskers denote 5th and 95th percentiles\n{N_SELECTED_TASKS} tasks from Natural Instructions")
    # plt.ylabel("")
    # plt.title("Mean performance change across prompt formats compared to default format\niid split")
    # plt.savefig(f"{args.image_dir}/mean_drop_boxplot.png", dpi=550, bbox_inches="tight")
    # plt.close()

    # unique_models = total_df["model"].unique()
    # model2color = dict(zip(unique_models, sns.color_palette("colorblind")))

    # for model in unique_models:
    #     subset = total_df[total_df["model"] == model]

    #     plt.figure(figsize=(12, 8))
    #     # sns.barplot(data=subset, x="spread", y="experiment", errorbar=("pi", 90), color=model2color[model], legend=False)
    #     sns.boxplot(data=subset, x="spread", y="experiment", color=model2color[model], legend=False)
    #     plt.xlabel(f"{N_SELECTED_TASKS} tasks from Natural Instructions")
    #     plt.ylabel("")
    #     plt.title(f"Spread across prompt formats\niid split")
    #     # plt.savefig(f"{args.image_dir}/barplot_{model}.png", dpi=550, bbox_inches="tight")
    #     plt.savefig(f"{args.image_dir}/boxplot_{model}.png", dpi=550, bbox_inches="tight")
    #     plt.close()

    #     plt.figure(figsize=(12, 8))
    #     sns.boxplot(data=subset, x="std", y="experiment", color=model2color[model], legend=False)
    #     plt.xlabel(f"{N_SELECTED_TASKS} tasks from Natural Instructions")
    #     plt.ylabel("")
    #     plt.title(f"Standard deviation of accuracy across prompt formats\niid split")
    #     plt.savefig(f"{args.image_dir}/std_boxplot_{model}.png", dpi=550, bbox_inches="tight")
    #     plt.close()

    #     plt.figure(figsize=(12, 8))
    #     sns.boxplot(data=subset, x="median_accuracy", y="experiment", color=model2color[model], legend=False)
    #     plt.xlabel(f"{N_SELECTED_TASKS} tasks from Natural Instructions")
    #     plt.ylabel("")
    #     plt.title(f"Median accuracy across prompt formats\niid split")
    #     plt.savefig(f"{args.image_dir}/median_accuracy_boxplot_{model}.png", dpi=550, bbox_inches="tight")
    #     plt.close()

    #     # plt.figure(figsize=(12, 8))
    #     # sns.boxplot(data=subset, x="mean_drop", y="experiment", color=model2color[model], whis=(5, 95), legend=False)
    #     # plt.xlabel(f"Whiskers denote 5th and 95th percentiles\n{N_SELECTED_TASKS} tasks from Natural Instructions")
    #     # plt.ylabel("")
    #     # plt.title(f"Mean performance change across prompt formats compared to default format\niid split")
    #     # plt.savefig(f"{args.image_dir}/mean_drop_boxplot_{model}.png", dpi=550, bbox_inches="tight")
    #     # plt.close()

    # plt.figure(figsize=(12, 8))
    # sns.barplot(data=total_df, x="spread", y="experiment", hue="model", errorbar=None, palette=model2color)
    # plt.xlabel(f"Spread across prompt formats\n{N_SELECTED_TASKS} tasks from Natural Instructions")
    # plt.ylabel("")
    # plt.title("Spread across prompt formats\niid split")
    # plt.savefig(f"{args.image_dir}/spread.png", dpi=550, bbox_inches="tight")
    # plt.close()

    # plt.figure(figsize=(12, 8))
    # sns.barplot(data=total_df, x="std", y="experiment", hue="model", errorbar=None, palette=model2color)
    # plt.xlabel(f"Standard deviation of accuracy across prompt formats\n{N_SELECTED_TASKS} tasks from Natural Instructions")
    # plt.ylabel("")
    # plt.title("Standard deviation of accuracy across prompt formats\niid split")
    # plt.savefig(f"{args.image_dir}/std.png", dpi=550, bbox_inches="tight")
    # plt.close()

    # plt.figure(figsize=(12, 8))
    # sns.barplot(data=total_df, x="spread", y="experiment", hue="model", errorbar=("pi", 90), palette=model2color, legend=False)
    # plt.xlabel(f"{N_SELECTED_TASKS} tasks from Natural Instructions")
    # plt.ylabel("")
    # plt.title("Spread across prompt formats\niid split")
    # plt.savefig(f"{args.image_dir}/all_barplot.png", dpi=550, bbox_inches="tight")
    # plt.close()

# Barplot with errorbars
    # plt.figure(figsize=(12, 8))
    # sns.barplot(data=total_df, x="median_accuracy", y="experiment", hue="model", errorbar=None, palette=model2color)
    # plt.errorbar(
    #     x=total_df.groupby("experiment")["median_accuracy"].mean(),
    #     y=total_df.groupby("experiment")["experiment"].first(),
    #     xerr=2 * total_df.groupby("experiment")["std"].mean(),
    #     fmt="none",
    #     linewidth=2,
    #     color="black",
    #     capsize=4,
    # )
    # plt.title(f"{N_SELECTED_TASKS} tasks from Natural Instructions")
    # plt.ylabel("")
    # plt.xlabel(f"Barplot -- accuracy: median over formats, mean over tasks.\nErrorbars -- variation of accuracy: 2 * (std over formats, mean over tasks).")
    # plt.savefig(f"{args.image_dir}/median_accuracy_all_boxplot.png", dpi=550, bbox_inches="tight")
    # plt.close()


if __name__ == "__main__":
    main()