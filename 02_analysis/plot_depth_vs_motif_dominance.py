"""
Plot maximum rule depth vs motif dominance (usage_max / seq_length).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "03_results")
COMBINED_CSV = os.path.join(RESULTS_DIR, "combined_grammar_summary.csv")
OUT_PNG = os.path.join(RESULTS_DIR, "plot_depth_vs_motif_dominance.png")

def main():
    df = pd.read_csv(COMBINED_CSV)

    df["motif_dominance"] = df["usage_max"] / df["seq_length"]

    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    x = df["depth_max"]
    y = df["motif_dominance"]

    ax.scatter(x, y, s=80, color="black", alpha=0.7)

    for _, row in df.iterrows():
        label = row["dataset"].replace("_Adamatzky_2021", "").replace("_Chowdhury_2025_fast", " (fast)")
        ax.text(row["depth_max"] + 0.05,
                row["motif_dominance"] + 0.005,
                label,
                fontsize=9)

    ax.set_xlabel("Maximum rule depth", fontsize=11)
    ax.set_ylabel("Motif dominance (usage_max / seq_length)", fontsize=11)
    ax.set_title("Hierarchy vs Motif Dominance", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()
    print(f"Saved {OUT_PNG}")

if __name__ == "__main__":
    main()