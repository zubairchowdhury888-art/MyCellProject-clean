"""
Plot entropy vs number of nonterminals for all datasets.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "03_results")
COMBINED_CSV = os.path.join(RESULTS_DIR, "combined_grammar_summary.csv")
OUT_PNG = os.path.join(RESULTS_DIR, "plot_entropy_vs_nonterminals.png")

def main():
    df = pd.read_csv(COMBINED_CSV)

    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    x = df["entropy_bits"]
    y = df["n_nonterminals"]

    ax.scatter(x, y, s=80, color="black", alpha=0.7)

    # Label each point with shortened dataset name
    for _, row in df.iterrows():
        label = row["dataset"].replace("_Adamatzky_2021", "").replace("_Chowdhury_2025_fast", " (fast)")
        ax.text(row["entropy_bits"] + 0.05,
                row["n_nonterminals"] + 1.5,
                label,
                fontsize=9)

    ax.set_xlabel("Symbol entropy (bits)", fontsize=11)
    ax.set_ylabel("Number of nonterminals", fontsize=11)
    ax.set_title("Entropy vs Grammar Size (Nonterminals)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()
    print(f"Saved {OUT_PNG}")

if __name__ == "__main__":
    main()