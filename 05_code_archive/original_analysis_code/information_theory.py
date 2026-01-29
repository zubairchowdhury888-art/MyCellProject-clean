#!/usr/bin/env python3
"""
information_theory.py - Calculate entropy and complexity metrics
Part of: Syntactic Information Processing in Fungal Networks (Chowdhury, 2025)

Usage:
    python information_theory.py detected_spikes.csv --duration 3600

Requirements:
    numpy, pandas, scipy
"""

import numpy as np
import pandas as pd
import argparse
import zlib

def calculate_shannon_entropy(sequence, window_size):
    """Calculate Shannon entropy for binary sequence"""
    n_windows = len(sequence) - window_size + 1
    if n_windows <= 0:
        return 0, 0

    patterns = {}
    for i in range(n_windows):
        pattern = tuple(sequence[i:i+window_size])
        patterns[pattern] = patterns.get(pattern, 0) + 1

    probs = np.array(list(patterns.values())) / n_windows
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(2**window_size)
    normalized = entropy / max_entropy if max_entropy > 0 else 0

    return entropy, normalized

def estimate_kolmogorov_complexity(sequence):
    """Estimate Kolmogorov complexity via compression"""
    seq_bytes = sequence.astype(np.uint8).tobytes()
    compressed = zlib.compress(seq_bytes, level=9)
    return len(compressed), len(seq_bytes), len(compressed) / len(seq_bytes)

def calculate_isi_entropy(isis):
    """Calculate entropy of inter-spike interval distribution"""
    hist, edges = np.histogram(isis, bins='auto')
    probs = hist / len(isis)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def main():
    parser = argparse.ArgumentParser(description='Calculate information theory metrics')
    parser.add_argument('spike_file', help='CSV file with detected spikes')
    parser.add_argument('--duration', type=int, required=True, help='Total recording duration (seconds)')
    parser.add_argument('--output', default='information_theory_metrics.csv', help='Output filename')

    args = parser.parse_args()

    print(f"Loading spikes from: {args.spike_file}")

    try:
        spikes_df = pd.read_csv(args.spike_file)
        spike_times = spikes_df['Time_seconds'].values
        isis = spikes_df['ISI_seconds'].dropna().values

        print(f"Loaded {len(spike_times)} spikes")
        print(f"Recording duration: {args.duration} seconds ({args.duration/3600:.1f} hours)")

        # Create binary spike train
        spike_binary = np.zeros(args.duration)
        spike_times_int = spike_times.astype(int)
        spike_times_int = spike_times_int[spike_times_int < args.duration]
        spike_binary[spike_times_int] = 1

        print("\nCalculating Shannon entropy at multiple scales...")

        window_sizes = [5, 10, 30, 60]
        results = []

        for w in window_sizes:
            ent, norm = calculate_shannon_entropy(spike_binary, w)
            print(f"  Window {w}s: H = {ent:.4f} bits (normalized: {norm:.4f})")
            results.append({
                'Metric': f'Shannon_Entropy_{w}s',
                'Value': ent,
                'Normalized': norm
            })

        # Kolmogorov complexity
        print("\nEstimating Kolmogorov complexity...")
        comp_len, orig_len, complexity = estimate_kolmogorov_complexity(spike_binary)
        print(f"  Original: {orig_len} bytes")
        print(f"  Compressed: {comp_len} bytes")
        print(f"  K = {complexity:.4f}")
        print(f"  Compressibility: {(1-complexity)*100:.1f}%")

        results.append({
            'Metric': 'Kolmogorov_Complexity',
            'Value': complexity,
            'Normalized': 1 - complexity  # compressibility
        })

        # ISI entropy
        print("\nCalculating ISI entropy...")
        isi_entropy = calculate_isi_entropy(isis)
        print(f"  H_ISI = {isi_entropy:.4f} bits")

        results.append({
            'Metric': 'ISI_Entropy',
            'Value': isi_entropy,
            'Normalized': np.nan
        })

        # Redundancy
        norm_entropy_10s = [r['Normalized'] for r in results if 'Shannon_Entropy_10s' in r['Metric']][0]
        redundancy = 1 - norm_entropy_10s

        results.append({
            'Metric': 'Redundancy',
            'Value': redundancy,
            'Normalized': np.nan
        })

        print(f"  Redundancy: {redundancy:.1%}")

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
