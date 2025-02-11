import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from typing import Dict, List
from math import sqrt, ceil
from statistics import median, stdev
from argparse import ArgumentParser

# sns.set_theme(font="monospace", font_scale=1.5)
sns.set_theme(font_scale=1.5)

def read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _extract_name_from_path(path: str) -> str:
    """Example of expected path:
        'metadataholistic_textdisabled_rankscore_task158_search_model_Qwen2.5-7B-Instruct_nshot_5_numnodes_9_numsamples_1000.json'
    We want to extract task name, and task name
    """
    return path.split("_")[3]


def plot_and_save_hist(values: np.ndarray, experiment_name: str, savepath: str | Path):
    plt.hist(values, bins=ceil(sqrt(len(values))))
    plt.title(f"{experiment_name} spreads\non {len(values)} tasks from Natural Instructions")
    plt.xlabel("Spread (max accuracy - min accuracy)")
    plt.savefig(savepath, dpi=350, bbox_inches="tight")
    plt.close()


def _process_single_path(path: str) -> Dict:
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
    accuracy_std = stdev([acc_tuple[0] for acc_tuple in format_to_acc.values()])

    # first format in dict is assumed to be the default format
    default_format = next(iter(format_to_acc))
    default_accuracy = format_to_acc[default_format][0]
    changes = [default_accuracy - acc_tuple[0] for acc_tuple in format_to_acc.values()]
    changes.pop(0) # remove first 0

    return {
        "task": _extract_name_from_path(str(path)),
        "spread": best_acc - worst_acc,
        "std": accuracy_std,
        "best_accuracy": best_acc,
        "worst_accuracy": worst_acc,
        "median_accuracy": median_accuracy,
        "default_accuracy": default_accuracy,
        "mean_drop": sum(changes) / len(changes) if len(changes) > 0 else 0,
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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root-dir", default="exp")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--num-nodes", type=int, default=10)
    parser.add_argument("-e", "--experiment-names", nargs="+")

    args = parser.parse_args()
    return args


N_SELECTED_TASKS = 52

if __name__ == "__main__":
    args = parse_args()

    total_df = []

    Path(args.image_dir).mkdir()

    experiment_names = args.experiment_names if args.experiment_names else \
        sorted([filename.name for filename in Path(args.root_dir).iterdir() if filename.is_dir()])

    experiment_names = [name for name in experiment_names if 
        not "zeroshot" in name and 
        not "test" in name and 
        # not "ensemble" in name and 
        not "debug" in name and 
        not "batch-calibration-probs" in name and 
        # not "sensitivity-aware-decoding" in name and
        not "rank1" in name and
        # not "batch-calibration" in name and
        not "default" in name and
        not "augs" in name and
        not "exact-match" in name and
        "compositional" in name and
        not "unbalanced" in name
        # not "1ksteps" in name and
        # not "superclear" in name and
        # not "iid" in name and
        # not "simpleanswers-batch-calibration" in name
    ]
    # experiment_names = [name for name in experiment_names if "2-shot" in name or "0-shot" in name]
    experiment_names = [name for name in experiment_names if ("2-shot" in name and "lora" not in name) or ("0-shot" in name and "lora" in name)]
    experiment_names = [name for name in experiment_names if "no-chat-template" in name]
    experiment_names = [name for name in experiment_names if ("instruct" in name.lower() or "it" in name)]
    experiment_names = [name for name in experiment_names if "Llama-3.2" in name or "Qwen2.5" in name or "gemma-2" in name]
    # experiment_names = [name for name in experiment_names if "Llama-3.1-8B" in name or "Qwen2.5-7B" in name or "gemma-2-9b" in name]
    # experiment_names = [name for name in experiment_names if ("lora" not in name) or ("lora" in name and "iid" in name)]

    for n in experiment_names:
        print("\t", n)

    for experiment_name in tqdm(experiment_names, desc="models"):
        subdir = Path(args.root_dir) / experiment_name

        model_name, tail = experiment_name.split("---")

        num_nodes = 5 * (args.num_nodes + 1) - 1 if "ensemble" in experiment_name else args.num_nodes
        pattern = f"metadataholistic*{model_name}*.json"
        evaluated_tasks_result_paths = list(subdir.glob(pattern))
        if len(evaluated_tasks_result_paths) != N_SELECTED_TASKS:
            print(f"{model_name=}")
            print(f"ATTENTION: {experiment_name} is only evaluated on {len(evaluated_tasks_result_paths)} tasks")
        df = collect_spreads(evaluated_tasks_result_paths)

        tail = tail.replace("no-chat-template-", "")
        tail = tail.replace("compositional-separator-space-", "")
        tail = tail.replace("compositional-", "")


        if "lora" in model_name:
            tail = tail.replace("0-shot", "")
            tail = tail.replace("response-only-", "")
            tail = tail.replace("iidx2", "")
            tail = tail.replace("iid", "-ood")
            tail = "LoRA" + tail 
            model_name = model_name[:-len("_lora")]
        else:
            tail = tail.replace("2-shot", "")
            tail = tail.replace("iid-split-", "")
            tail = tail.replace("iidx2-", "")
            tail = tail.replace("iid", "")
            tail = tail.replace("batch-calibration-", "BC")
            tail = tail.replace("sensitivity-aware-decoding-", "SAD")
            tail = tail.replace("template-ensembles-", "TE")
            if tail == "" or tail == "-":
                tail = "FS"

        df["experiment"] = f"{model_name:>30}{tail:>40}"
        df["method"] = tail
        df["model"] = model_name

        df.to_csv(subdir / "spreads.csv")
        plot_and_save_hist(df["spread"].values, experiment_name, subdir / "spreads.png")
        total_df.append(df)
    
    total_df = pd.concat(total_df)

    all_methods = ["FS", "BC", "SAD", "TE", "LoRA", "LoRA-ood", "LoRA-JS"]

    # unique_methods = total_df["method"].unique()
    # ordered_methods = [m for m in all_methods if m in unique_methods]
    # print(ordered_methods)
    method2color = dict(zip(all_methods, sns.color_palette("Set2")))

    total_df["method"] = pd.Categorical(total_df["method"], categories=all_methods)
    total_df.sort_values("method")
    total_df["method"] = total_df["method"].cat.remove_unused_categories()

    default_figsize = (17, 5)

# Clustered barplot of accuracy with errorbars
    plt.figure(figsize=default_figsize)
    ax = sns.barplot(data=total_df, x="model", y="median_accuracy", hue="method", errorbar=None, palette=method2color)
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
    sns.move_legend(ax, "center left", bbox_to_anchor=(1.0, 0.5), ncol=1, title="Method")
    plt.savefig(f"{args.image_dir}/clustered_barplot.png", dpi=350, bbox_inches="tight")
    plt.close()

# Std
    plt.figure(figsize=default_figsize)
    ax = sns.barplot(data=total_df, x="model", y="std", hue="method", errorbar=None, palette=method2color)
    plt.xticks(rotation=15, ha="right")
    plt.xlabel("")
    plt.ylabel("Standard deviation over prompts", labelpad=25)
    plt.title("Methods' standard deviation over prompts on different models\n(lower is better)")
    sns.move_legend(ax, "center left", bbox_to_anchor=(1.0, 0.5), ncol=1, title="Method")
    plt.savefig(f"{args.image_dir}/clustered_std_barplot.png", dpi=350, bbox_inches="tight")
    plt.close()

# Spread
    plt.figure(figsize=default_figsize)
    ax = sns.barplot(data=total_df, x="model", y="spread", hue="method", errorbar=None, palette=method2color)
    plt.xticks(rotation=15, ha="right")
    plt.xlabel("")
    plt.ylabel("Spread over prompts", labelpad=25)
    plt.title("Methods' spread over prompts on different models\n(lower is better)")
    sns.move_legend(ax, "center left", bbox_to_anchor=(1.0, 0.5), ncol=1, title="Method")
    plt.savefig(f"{args.image_dir}/clustered_spread_barplot.png", dpi=350, bbox_inches="tight")
    plt.close()

    # plt.figure(figsize=(12, 8))
    # sns.boxplot(data=total_df, x="mean_drop", y="experiment", hue="model", whis=(5, 95), palette=model2color, legend=False)
    # plt.xlabel(f"Whiskers denote 5th and 95th percentiles\n{N_SELECTED_TASKS} tasks from Natural Instructions")
    # plt.ylabel("")
    # plt.title("Mean performance change across prompt formats compared to default format\niid split")
    # plt.savefig(f"{args.image_dir}/mean_drop_boxplot.png", dpi=350, bbox_inches="tight")
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
    #     # plt.savefig(f"{args.image_dir}/barplot_{model}.png", dpi=350, bbox_inches="tight")
    #     plt.savefig(f"{args.image_dir}/boxplot_{model}.png", dpi=350, bbox_inches="tight")
    #     plt.close()

    #     plt.figure(figsize=(12, 8))
    #     sns.boxplot(data=subset, x="std", y="experiment", color=model2color[model], legend=False)
    #     plt.xlabel(f"{N_SELECTED_TASKS} tasks from Natural Instructions")
    #     plt.ylabel("")
    #     plt.title(f"Standard deviation of accuracy across prompt formats\niid split")
    #     plt.savefig(f"{args.image_dir}/std_boxplot_{model}.png", dpi=350, bbox_inches="tight")
    #     plt.close()

    #     plt.figure(figsize=(12, 8))
    #     sns.boxplot(data=subset, x="median_accuracy", y="experiment", color=model2color[model], legend=False)
    #     plt.xlabel(f"{N_SELECTED_TASKS} tasks from Natural Instructions")
    #     plt.ylabel("")
    #     plt.title(f"Median accuracy across prompt formats\niid split")
    #     plt.savefig(f"{args.image_dir}/median_accuracy_boxplot_{model}.png", dpi=350, bbox_inches="tight")
    #     plt.close()

    #     # plt.figure(figsize=(12, 8))
    #     # sns.boxplot(data=subset, x="mean_drop", y="experiment", color=model2color[model], whis=(5, 95), legend=False)
    #     # plt.xlabel(f"Whiskers denote 5th and 95th percentiles\n{N_SELECTED_TASKS} tasks from Natural Instructions")
    #     # plt.ylabel("")
    #     # plt.title(f"Mean performance change across prompt formats compared to default format\niid split")
    #     # plt.savefig(f"{args.image_dir}/mean_drop_boxplot_{model}.png", dpi=350, bbox_inches="tight")
    #     # plt.close()

    # plt.figure(figsize=(12, 8))
    # sns.barplot(data=total_df, x="spread", y="experiment", hue="model", errorbar=None, palette=model2color)
    # plt.xlabel(f"Spread across prompt formats\n{N_SELECTED_TASKS} tasks from Natural Instructions")
    # plt.ylabel("")
    # plt.title("Spread across prompt formats\niid split")
    # plt.savefig(f"{args.image_dir}/spread.png", dpi=350, bbox_inches="tight")
    # plt.close()

    # plt.figure(figsize=(12, 8))
    # sns.barplot(data=total_df, x="std", y="experiment", hue="model", errorbar=None, palette=model2color)
    # plt.xlabel(f"Standard deviation of accuracy across prompt formats\n{N_SELECTED_TASKS} tasks from Natural Instructions")
    # plt.ylabel("")
    # plt.title("Standard deviation of accuracy across prompt formats\niid split")
    # plt.savefig(f"{args.image_dir}/std.png", dpi=350, bbox_inches="tight")
    # plt.close()

    # plt.figure(figsize=(12, 8))
    # sns.barplot(data=total_df, x="spread", y="experiment", hue="model", errorbar=("pi", 90), palette=model2color, legend=False)
    # plt.xlabel(f"{N_SELECTED_TASKS} tasks from Natural Instructions")
    # plt.ylabel("")
    # plt.title("Spread across prompt formats\niid split")
    # plt.savefig(f"{args.image_dir}/all_barplot.png", dpi=350, bbox_inches="tight")
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
    # plt.savefig(f"{args.image_dir}/median_accuracy_all_boxplot.png", dpi=350, bbox_inches="tight")
    # plt.close()
