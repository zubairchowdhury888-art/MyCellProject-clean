"""
Dataset Inspector for Fungal Electrical Activity Comparative Study
Author: Zubair Chowdhury
Date: January 2, 2026
"""

import pandas as pd
import os
import sys

print("="*70)
print("FUNGAL ELECTRICAL ACTIVITY - DATASET INSPECTION")
print("Comparative Analysis: 5 Datasets")
print("="*70)

raw_data_path = "../00_raw_data"

datasets = {
    "Adamatzky 2021 - Cordyceps militari": 
        f"{raw_data_path}/zenodo_adamatzky_2021/Cordyceps_militari.txt",
    "Adamatzky 2021 - Enoki (Flammulina velutipes)": 
        f"{raw_data_path}/zenodo_adamatzky_2021/Enoki_Flammulina_velutipes.txt",
    "Adamatzky 2021 - Ghost Fungi (Omphalotus nidiformis)": 
        f"{raw_data_path}/zenodo_adamatzky_2021/Ghost_Fungi_Omphalotus_nidiformis.txt",
    "Adamatzky 2021 - Schizophyllum commune": 
        f"{raw_data_path}/zenodo_adamatzky_2021/Schizophyllum_commune.txt",
    "Chowdhury 2025 - Schizophyllum commune (Fast Spikes)": 
        f"{raw_data_path}/chowdhury_2025/fast_spike_window.csv"
}

results = []

for name, filepath in datasets.items():
    print(f"\n{'='*70}")
    print(f"Dataset: {name}")
    print('='*70)
    
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        continue
    
    # Get file size
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    try:
        # Read first 1000 rows to inspect
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, nrows=1000)
        else:
            # Try tab-separated first (Adamatzky format)
            df = pd.read_csv(filepath, sep='\t', nrows=1000)
        
        # Get basic stats
        n_channels = df.shape[1] - 1 if 'Time' in df.columns[0] else df.shape[1]
        
        print(f"Columns detected: {df.shape[1]}")
        print(f"Electrode channels: {n_channels}")
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        
        # Voltage statistics (skip time column if present)
        voltage_cols = df.select_dtypes(include='number').columns
        print(f"\nVoltage range (mV):")
        print(f"  Min: {df[voltage_cols].min().min():.6f}")
        print(f"  Max: {df[voltage_cols].max().max():.6f}")
        print(f"  Mean: {df[voltage_cols].mean().mean():.6f}")
        
        results.append({
            'Dataset': name,
            'Size_MB': file_size_mb,
            'Channels': n_channels,
            'Status': 'LOADED'
        })
        
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append({
            'Dataset': name,
            'Size_MB': file_size_mb,
            'Channels': 0,
            'Status': f'ERROR: {str(e)[:50]}'
        })

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))

print(f"\n[SUCCESS] Inspected {len([r for r in results if 'LOADED' in r['Status']])}/{len(datasets)} datasets")
print(f"Total data size: {sum([r['Size_MB'] for r in results]):.2f} MB")
print("\n" + "="*70)
print("Next step: Run grammar extraction analysis!")
print("="*70)