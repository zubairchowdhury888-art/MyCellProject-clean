"""
Comparative grammar-ready preprocessing for fungal electrical datasets
Author: Zubair Chowdhury
Date: January 2, 2026

Step 1: Load 5 datasets
Step 2: Normalise & resample
Step 3: Convert to symbolic sequences ready for grammar extraction
Step 4: Compute simple syntactic complexity proxies
"""

import os
import pandas as pd
import numpy as np

RAW_PATH = "../00_raw_data"

DATASETS = {
    "Cordyceps_militari_Adamatzky_2021":
        f"{RAW_PATH}/zenodo_adamatzky_2021/Cordyceps_militari.txt",
    "Enoki_Flammulina_Adamatzky_2021":
        f"{RAW_PATH}/zenodo_adamatzky_2021/Enoki_Flammulina_velutipes.txt",
    "GhostFungi_Omphalotus_Adamatzky_2021":
        f"{RAW_PATH}/zenodo_adamatzky_2021/Ghost_Fungi_Omphalotus_nidiformis.txt",
    "Schizophyllum_Adamatzky_2021":
        f"{RAW_PATH}/zenodo_adamatzky_2021/Schizophyllum_commune.txt",
    "Schizophyllum_Chowdhury_2025_fast":
        f"{RAW_PATH}/chowdhury_2025/fast_spike_window.csv",
}

OUTPUT_SYM_PATH = "../03_results/grammar_sequences"
os.makedirs(OUTPUT_SYM_PATH, exist_ok=True)

def load_dataset(name, path):
    if not os.path.exists(path):
        print(f"[ERROR] Missing: {name} -> {path}")
        return None

    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep="\t")

    # Use only numeric columns
    num_df = df.select_dtypes(include="number")
    return num_df

def zscore_normalise(df):
    return (df - df.mean()) / df.std(ddof=0)

def resample_to_length(series, target_len=50000):
    """Resample a 1D series to a fixed length for fair comparison."""
    orig_len = len(series)
    if orig_len == target_len:
        return series.values

    x_old = np.linspace(0, 1, orig_len)
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, series.values)

def quantise_to_symbols(series, n_bins=5):
    """
    Map continuous z-scored values to discrete symbols 'A'.. based on quantiles.
    """
    q = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(series, q)
    # ensure strictly increasing
    bins[0] -= 1e-9
    bins[-1] += 1e-9

    labels = [chr(ord("A") + i) for i in range(n_bins)]
    digitised = np.digitize(series, bins[1:-1], right=False)
    symbols = [labels[i] for i in digitised]
    return "".join(symbols)

def shannon_entropy(symbol_string):
    """Simple syntactic complexity proxy: symbol entropy."""
    if not symbol_string:
        return 0.0
    values, counts = np.unique(list(symbol_string), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())

def process_all_datasets():
    rows = []

    for name, path in DATASETS.items():
        print("\n" + "="*70)
        print(f"Processing: {name}")
        print("="*70)

        df = load_dataset(name, path)
        if df is None or df.empty:
            rows.append({
                "dataset": name,
                "channels_used": 0,
                "sequence_length": 0,
                "entropy": np.nan,
            })
            continue

        # For now: pick first numeric channel
        channel = df.iloc[:, 0]
        print(f"Original length: {len(channel)} samples")

        # Normalise and resample
        channel_z = zscore_normalise(channel)
        resampled = resample_to_length(channel_z, target_len=50000)

        # Quantise -> symbol string
        sym_string = quantise_to_symbols(pd.Series(resampled), n_bins=5)

        # Save symbol sequence to file
        out_file = os.path.join(OUTPUT_SYM_PATH, f"{name}_symseq.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(sym_string)

        # Complexity proxy
        H = shannon_entropy(sym_string)

        rows.append({
            "dataset": name,
            "channels_used": 1,
            "sequence_length": len(sym_string),
            "entropy_bits": H,
        })

        print(f"Saved symbol sequence: {out_file}")
        print(f"Entropy (bits): {H:.4f}")

    summary = pd.DataFrame(rows)
    summary_path = "../03_results/grammar_complexity_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nSUMMARY TABLE:")
    print(summary)
    print(f"\nSaved summary to {summary_path}")

if __name__ == "__main__":
    process_all_datasets()