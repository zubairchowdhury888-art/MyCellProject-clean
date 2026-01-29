"""
Merge entropy and grammar metrics into a single summary CSV.

Inputs (03_results/):
  - grammar_complexity_summary.csv  (entropy per dataset)
  - grammar_metrics.json            (Sequitur grammar metrics per dataset)

Output (03_results/):
  - combined_grammar_summary.csv
"""

import os
import json
import pandas as pd

# Base paths (relative to this script in 02_analysis/)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "03_results")

ENTROPY_CSV = os.path.join(RESULTS_DIR, "grammar_complexity_summary.csv")
GRAMMAR_JSON = os.path.join(RESULTS_DIR, "grammar_metrics.json")
OUT_CSV = os.path.join(RESULTS_DIR, "combined_grammar_summary.csv")


def load_entropy_table(path: str) -> pd.DataFrame:
    """
    Load entropy summary CSV and normalise the dataset name column.

    Expected columns (at minimum):
      - dataset or dataset_name
      - entropy_bits (or similar)
    """
    df = pd.read_csv(path)

    # Normalise column names
    cols = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # Try to identify dataset and entropy columns
    if "dataset" in df.columns:
        dataset_col = "dataset"
    elif "dataset_name" in df.columns:
        dataset_col = "dataset_name"
    else:
        raise ValueError("Could not find dataset column in entropy CSV.")

    # Heuristic for entropy column name
    entropy_candidates = [c for c in df.columns if "entropy" in c]
    if not entropy_candidates:
        raise ValueError("Could not find an entropy column in entropy CSV.")
    entropy_col = entropy_candidates[0]

    df = df[[dataset_col, entropy_col]].copy()
    df.rename(columns={dataset_col: "dataset", entropy_col: "entropy_bits"}, inplace=True)

    # Ensure dataset names align with grammar_metrics keys where needed
    # If your CSV uses names without the '_symseq' suffix, they will still match
    # because we will strip that suffix from the JSON keys below.
    return df


def load_grammar_metrics(path: str) -> pd.DataFrame:
    """
    Load grammar_metrics.json and return as a DataFrame with a 'dataset' column.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for key, metrics in data.items():
        # Strip trailing '_symseq' if present to match entropy dataset names
        if key.endswith("_symseq"):
            dataset = key.replace("_symseq", "")
        else:
            dataset = key
        row = {"dataset": dataset}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def merge_entropy_and_grammar():
    ent_df = load_entropy_table(ENTROPY_CSV)
    gram_df = load_grammar_metrics(GRAMMAR_JSON)

    # Inner join on dataset name; assert we have the same 5 datasets
    merged = pd.merge(ent_df, gram_df, on="dataset", how="inner")

    merged.to_csv(OUT_CSV, index=False)
    print(f"Wrote combined summary for {len(merged)} datasets to {OUT_CSV}")


if __name__ == "__main__":
    merge_entropy_and_grammar()