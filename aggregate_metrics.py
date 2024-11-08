import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List
from math import sqrt, ceil
from statistics import median
from argparse import ArgumentParser

sns.set_theme()


def read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _extract_name_from_path(path: str) -> str:
    """Example of expected path:
        'metadataholistic_textdisabled_rankscore_task158_search_model_Qwen2.5-7B-Instruct_nshot_5_numnodes_9_numsamples_1000.json'
    We want to extract task name, and task name
    """
    return path.split("_")[3]


def plot_and_save_hist(values: np.ndarray, model_name: str, savepath: str | Path):
    plt.hist(values, bins=ceil(sqrt(len(values))))
    plt.title(f"{model_name} spreads\non {len(values)} tasks from Natural Instructions")
    plt.xlabel("Spread (max accuracy - min accuracy)")
    plt.savefig(savepath, dpi=350, bbox_inches="tight")
    plt.close()


def collect_spreads(paths: List[str]) -> pd.DataFrame:
    records = []

    for path in tqdm(paths, desc="paths"):
        result_json = read_json(path)
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

        record = {
            "task": _extract_name_from_path(str(path)),
            "spread": best_acc - worst_acc,
            "best_accuracy": best_acc,
            "worst_accuracy": worst_acc,
            "median_accuracy": median_accuracy,
            "best_format": best_format,
            "worst_format": worst_format,
            "best_n": best_n,
            "worst_n": worst_n,
        }

        records.append(record)

    return pd.DataFrame.from_records(records)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root-dir", default="exp")
    parser.add_argument("--num-nodes", type=int, default=9)
    parser.add_argument("-m", "--model-names", nargs="+")

    args = parser.parse_args()
    return args


N_SELECTED_TASKS = 53

if __name__ == "__main__":
    args = parse_args()

    total_df = []

    model_names = args.model_names if args.model_names else \
        sorted([filename.name for filename in Path(args.root_dir).iterdir() if filename.is_dir()])
    print(model_names)

    for model_name in model_names:
        subdir = Path(args.root_dir) / model_name
        pattern = f"metadataholistic*{model_name[:-len('-chattemplate')] if model_name.endswith('-chattemplate') else model_name}*_numnodes_{args.num_nodes}*.json"
        df = collect_spreads(list(subdir.glob(pattern)))
        assert len(df) == N_SELECTED_TASKS, f"{len(df)=}"
        df["model"] = model_name

        df.to_csv(subdir / "spreads.csv")
        plot_and_save_hist(df["spread"].values, model_name, subdir / "spreads.png")
        total_df.append(df)
    
    total_df = pd.concat(total_df)

    sns.boxplot(data=total_df, x="spread", y="model", hue="model", palette="colorblind", notch=True)
    plt.xlabel(f"Performance spread across prompt formats\n5-shot, {N_SELECTED_TASKS} tasks from Natural Instructions")
    plt.ylabel("")
    plt.savefig(f"{args.root_dir}/all_boxplot.png", dpi=350, bbox_inches="tight")
    plt.close()

    sns.boxplot(data=total_df, x="median_accuracy", y="model", hue="model", palette="colorblind")
    plt.xlabel(f"Median accuracy across prompt formats\n5-shot, {N_SELECTED_TASKS} tasks from Natural Instructions")
    plt.ylabel("")
    plt.savefig(f"{args.root_dir}/median_accuracy_all_boxplot.png", dpi=350, bbox_inches="tight")
    plt.close()
