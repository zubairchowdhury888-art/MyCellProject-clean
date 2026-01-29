#!/usr/bin/env python3
"""
markov_baseline.py - Generate Markov baseline and compare
Part of: Syntactic Information Processing in Fungal Networks (Chowdhury, 2025)

Usage:
    python markov_baseline.py detected_spikes.csv --duration 3600 --n_trials 20

Requirements:
    numpy, pandas, scipy
"""

import numpy as np
import pandas as pd
import argparse
from scipy import stats
import zlib

def calculate_entropy_simple(sequence, window_size):
    """Simplified entropy calculation"""
    n_windows = len(sequence) - window_size + 1
    if n_windows <= 0:
        return 0

    patterns = {}
    for i in range(n_windows):
        pattern = tuple(sequence[i:i+window_size])
        patterns[pattern] = patterns.get(pattern, 0) + 1

    probs = np.array(list(patterns.values())) / n_windows
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(2**window_size)
    return entropy / max_entropy if max_entropy > 0 else 0

def estimate_complexity_simple(sequence):
    """Simplified complexity estimation"""
    seq_bytes = sequence.astype(np.uint8).tobytes()
    compressed = zlib.compress(seq_bytes, level=9)
    return len(compressed) / len(seq_bytes)

def generate_markov_sequence(spike_binary, length):
    """Generate Markov sequence from empirical transitions"""
    # Calculate transition probabilities
    transitions = np.zeros((2, 2))
    for i in range(len(spike_binary) - 1):
        from_state = int(spike_binary[i])
        to_state = int(spike_binary[i+1])
        transitions[from_state, to_state] += 1

    trans_probs = transitions / transitions.sum(axis=1, keepdims=True)

    # Generate sequence
    markov_seq = np.zeros(length)
    markov_seq[0] = spike_binary[0]

    for i in range(1, length):
        current_state = int(markov_seq[i-1])
        markov_seq[i] = np.random.choice([0, 1], p=trans_probs[current_state])

    return markov_seq, trans_probs

def main():
    parser = argparse.ArgumentParser(description='Generate Markov baseline comparison')
    parser.add_argument('spike_file', help='CSV file with detected spikes')
    parser.add_argument('--duration', type=int, required=True, help='Recording duration (seconds)')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Markov simulations')
    parser.add_argument('--output', default='markov_baseline_comparison.csv', help='Output filename')

    args = parser.parse_args()

    print(f"Loading spikes from: {args.spike_file}")

    try:
        spikes_df = pd.read_csv(args.spike_file)
        spike_times = spikes_df['Time_seconds'].values

        # Create binary spike train
        spike_binary = np.zeros(args.duration)
        spike_times_int = spike_times.astype(int)
        spike_times_int = spike_times_int[spike_times_int < args.duration]
        spike_binary[spike_times_int] = 1

        print(f"Analyzing {len(spike_times)} spikes over {args.duration} seconds")

        # Calculate empirical metrics
        print("\nCalculating empirical metrics...")
        empirical_entropy = calculate_entropy_simple(spike_binary, 10)
        empirical_complexity = estimate_complexity_simple(spike_binary)

        print(f"  Empirical entropy (10s): {empirical_entropy:.4f}")
        print(f"  Empirical complexity: {empirical_complexity:.4f}")

        # Generate Markov baselines
        print(f"\nGenerating {args.n_trials} Markov baseline sequences...")

        markov_entropies = []
        markov_complexities = []

        for trial in range(args.n_trials):
            markov_seq, trans_probs = generate_markov_sequence(spike_binary, len(spike_binary))

            ent = calculate_entropy_simple(markov_seq, 10)
            comp = estimate_complexity_simple(markov_seq)

            markov_entropies.append(ent)
            markov_complexities.append(comp)

            if (trial + 1) % 5 == 0:
                print(f"  Completed {trial + 1}/{args.n_trials} trials")

        # Statistical comparison
        mean_markov_ent = np.mean(markov_entropies)
        std_markov_ent = np.std(markov_entropies)
        mean_markov_comp = np.mean(markov_complexities)
        std_markov_comp = np.std(markov_complexities)

        t_ent, p_ent = stats.ttest_1samp(markov_entropies, empirical_entropy)
        t_comp, p_comp = stats.ttest_1samp(markov_complexities, empirical_complexity)

        print("\n" + "="*60)
        print("STATISTICAL COMPARISON")
        print("="*60)
        print(f"\nEntropy (10s window):")
        print(f"  Empirical: {empirical_entropy:.4f}")
        print(f"  Markov:    {mean_markov_ent:.4f} ± {std_markov_ent:.4f}")
        print(f"  t = {t_ent:.4f}, p = {p_ent:.6f}")
        print(f"  Result: {'SIGNIFICANT' if p_ent < 0.05 else 'NOT SIGNIFICANT'} (α=0.05)")

        print(f"\nComplexity:")
        print(f"  Empirical: {empirical_complexity:.4f}")
        print(f"  Markov:    {mean_markov_comp:.4f} ± {std_markov_comp:.4f}")
        print(f"  t = {t_comp:.4f}, p = {p_comp:.6f}")
        print(f"  Result: {'SIGNIFICANT' if p_comp < 0.05 else 'NOT SIGNIFICANT'} (α=0.05)")

        # Save results
        results = pd.DataFrame({
            'Metric': ['Entropy', 'Entropy', 'Complexity', 'Complexity'],
            'Type': ['Empirical', 'Markov_Mean', 'Empirical', 'Markov_Mean'],
            'Value': [empirical_entropy, mean_markov_ent, empirical_complexity, mean_markov_comp],
            'Std': [np.nan, std_markov_ent, np.nan, std_markov_comp],
            'T_Statistic': [t_ent, np.nan, t_comp, np.nan],
            'P_Value': [p_ent, np.nan, p_comp, np.nan]
        })

        results.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
