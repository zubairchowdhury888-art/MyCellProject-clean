#!/usr/bin/env python3
"""
spike_detection.py - Detect electrical spikes in fungal recordings
Part of: Syntactic Information Processing in Fungal Networks (Chowdhury, 2025)

Usage:
    python spike_detection.py input_data.csv --output detected_spikes.csv

Arguments:
    input_data.csv: CSV with columns [Time, Channel_1, Channel_3, Channel_5, Channel_7, Channel_9]
    --output: Output filename for detected spikes (optional, default: detected_spikes.csv)
    --threshold: Detection threshold in mV/s (default: -0.15)
    --min_interval: Minimum spike separation in seconds (default: 30)
    --channel: Which channel to analyze (default: Channel_7)

Requirements:
    numpy, pandas
"""

import numpy as np
import pandas as pd
import argparse
import sys

def detect_spikes(voltage_data, time_data, threshold=-0.15, min_interval=30):
    """
    Detect spikes using derivative thresholding

    Parameters:
    -----------
    voltage_data : array-like
        Voltage time series (mV)
    time_data : array-like
        Time stamps (seconds)
    threshold : float
        dV/dt threshold for spike detection (mV/s), default -0.15
    min_interval : int
        Minimum time between spikes (seconds), default 30

    Returns:
    --------
    spike_indices : array
        Indices of detected spikes
    spike_times : array
        Time stamps of spikes
    spike_amplitudes : array
        Voltage amplitudes at spike peaks
    """

    # Calculate derivative
    dv_dt = np.diff(voltage_data)

    # Detect crossings
    spike_candidates = np.where(dv_dt < threshold)[0]

    if len(spike_candidates) == 0:
        return np.array([]), np.array([]), np.array([])

    # Filter by minimum interval
    filtered_spikes = [spike_candidates[0]]

    for spike_idx in spike_candidates[1:]:
        time_since_last = time_data[spike_idx] - time_data[filtered_spikes[-1]]
        if time_since_last >= min_interval:
            filtered_spikes.append(spike_idx)

    spike_indices = np.array(filtered_spikes)
    spike_times = time_data[spike_indices]
    spike_amplitudes = voltage_data[spike_indices]

    return spike_indices, spike_times, spike_amplitudes

def main():
    parser = argparse.ArgumentParser(description='Detect spikes in fungal electrical recordings')
    parser.add_argument('input_file', help='Input CSV file with voltage data')
    parser.add_argument('--output', default='detected_spikes.csv', help='Output CSV filename')
    parser.add_argument('--threshold', type=float, default=-0.15, help='Detection threshold (mV/s)')
    parser.add_argument('--min_interval', type=int, default=30, help='Minimum spike interval (seconds)')
    parser.add_argument('--channel', default='Channel_7', help='Channel to analyze')

    args = parser.parse_args()

    print(f"Loading data from: {args.input_file}")

    try:
        # Load data
        df = pd.read_csv(args.input_file, skiprows=1)

        # Check if channel exists
        if args.channel not in df.columns:
            print(f"Error: Channel '{args.channel}' not found in data")
            print(f"Available channels: {df.columns.tolist()}")
            sys.exit(1)

        time = df['Time'].values
        voltage = df[args.channel].values

        print(f"Data loaded: {len(time)} samples, {(time[-1]-time[0])/3600:.1f} hours")
        print(f"Analyzing channel: {args.channel}")
        print(f"Detection threshold: {args.threshold} mV/s")
        print(f"Minimum interval: {args.min_interval} seconds")

        # Detect spikes
        spike_indices, spike_times, spike_amplitudes = detect_spikes(
            voltage, time, args.threshold, args.min_interval
        )

        print(f"\nDetected {len(spike_times)} spikes")

        if len(spike_times) > 0:
            # Calculate inter-spike intervals
            isis = np.diff(spike_times)

            # Create output dataframe
            results = pd.DataFrame({
                'Spike_Number': range(1, len(spike_times) + 1),
                'Time_seconds': spike_times,
                'Amplitude_mV': spike_amplitudes,
                'ISI_seconds': np.concatenate([[np.nan], isis])
            })

            # Save results
            results.to_csv(args.output, index=False)
            print(f"Results saved to: {args.output}")

            # Print summary statistics
            print(f"\nSummary Statistics:")
            print(f"  Mean ISI: {np.mean(isis):.2f} seconds")
            print(f"  Median ISI: {np.median(isis):.2f} seconds")
            print(f"  Min ISI: {np.min(isis):.2f} seconds")
            print(f"  Max ISI: {np.max(isis):.2f} seconds")
            print(f"  Mean amplitude: {np.mean(spike_amplitudes):.4f} mV")
        else:
            print("No spikes detected. Try lowering the threshold.")

    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
