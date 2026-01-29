#!/usr/bin/env python3
"""
example_usage.py - Simple example of using the analysis scripts

This demonstrates how to use the scripts programmatically instead of command-line
"""

import pandas as pd
import numpy as np

# Example 1: Load and examine spike data
print("Example 1: Loading spike detection results")
print("="*60)

spikes = pd.read_csv('detected_spikes.csv')
print(f"Loaded {len(spikes)} spikes")
print(f"\nFirst 5 spikes:")
print(spikes.head())

# Calculate basic statistics
isis = spikes['ISI_seconds'].dropna()
print(f"\nMean ISI: {isis.mean():.2f} seconds")
print(f"Median ISI: {isis.median():.2f} seconds")

# Example 2: Load information theory metrics
print("\n\nExample 2: Information theory metrics")
print("="*60)

metrics = pd.read_csv('information_theory_metrics.csv')
print(metrics.to_string(index=False))

# Example 3: Load CA results
print("\n\nExample 3: Cellular automaton results")
print("="*60)

ca_results = pd.read_csv('ca_noise_results.csv')

# Calculate mean MI by model and noise level
summary = ca_results.groupby(['Model', 'Noise_Level'])['Mutual_Information'].mean()
print("\nMean Mutual Information by Model and Noise Level:")
print(summary.unstack())

print("\n\nAnalysis complete!")
