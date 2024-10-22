import json
import pandas as pd
from pathlib import Path
from typing import Dict, List


def read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _extract_name_from_path(path: str) -> str:
    """Example of expected path:
        'metadataholistic_textdisabled_rankscore_task158_search_model_Qwen2.5-7B-Instruct_nshot_5_numnodes_9_numsamples_1000.json'
    We want to extract task name, and task name
    """
    return path.split("_")[3]


def collect_spreads(paths: List[str]) -> pd.DataFrame:
    records = []

    for path in paths:
        print(path)
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

        spread = best_acc - worst_acc

        record = {
            "task": _extract_name_from_path(str(path)),
            "spread": spread,
            "best_accuracy": best_acc,
            "worst_accuracy": worst_acc,
            "best_format": best_format,
            "worst_format": worst_format,
            "best_n": best_n,
            "worst_n": worst_n,
        }

        records.append(record)

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    root_dir = "data"

    num_nodes = 9
    model_name = "Qwen2.5-7B"
    pattern = f"metadataholistic*{model_name}*_numnodes_{num_nodes}*.json"

    paths = Path(root_dir).glob(pattern)
    df = collect_spreads(paths)

    df["model"] = model_name

    print(df)