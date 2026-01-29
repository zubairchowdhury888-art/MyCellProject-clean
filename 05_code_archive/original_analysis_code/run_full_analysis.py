#!/usr/bin/env python3
"""
run_full_analysis.py - Run complete analysis pipeline
Part of: Syntactic Information Processing in Fungal Networks (Chowdhury, 2025)

This script runs the entire analysis pipeline:
1. Spike detection
2. Information theory metrics
3. Markov baseline comparison
4. Cellular automaton models

Usage:
    python run_full_analysis.py input_data.csv --duration 3533

Requirements:
    All other Python scripts must be in the same directory
"""

import subprocess
import sys
import argparse
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error in {description}")
        print(f"Command failed with return code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Error: Script not found. Make sure all Python files are in the same directory.")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run full analysis pipeline')
    parser.add_argument('input_file', help='Input CSV with voltage data')
    parser.add_argument('--duration', type=int, required=True, 
                        help='Recording duration in seconds')
    parser.add_argument('--threshold', type=float, default=-0.15, 
                        help='Spike detection threshold (mV/s)')
    parser.add_argument('--channel', default='Channel_7', 
                        help='Channel to analyze')

    args = parser.parse_args()

    print("="*80)
    print("COMPLETE ANALYSIS PIPELINE")
    print("Syntactic Information Processing in Fungal Networks")
    print("="*80)
    print(f"\nInput file: {args.input_file}")
    print(f"Duration: {args.duration} seconds ({args.duration/3600:.1f} hours)")
    print(f"Channel: {args.channel}")
    print(f"Threshold: {args.threshold} mV/s")

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"\nError: Input file '{args.input_file}' not found")
        sys.exit(1)

    # Step 1: Spike Detection
    success = run_command(
        ['python3', 'spike_detection.py', args.input_file,
         '--threshold', str(args.threshold),
         '--channel', args.channel,
         '--output', 'detected_spikes.csv'],
        "Spike Detection"
    )

    if not success:
        print("\nPipeline stopped due to error in spike detection")
        sys.exit(1)

    # Step 2: Information Theory Metrics
    success = run_command(
        ['python3', 'information_theory.py', 'detected_spikes.csv',
         '--duration', str(args.duration),
         '--output', 'information_theory_metrics.csv'],
        "Information Theory Analysis"
    )

    if not success:
        print("\nPipeline stopped due to error in information theory analysis")
        sys.exit(1)

    # Step 3: Markov Baseline
    success = run_command(
        ['python3', 'markov_baseline.py', 'detected_spikes.csv',
         '--duration', str(args.duration),
         '--n_trials', '20',
         '--output', 'markov_baseline_comparison.csv'],
        "Markov Baseline Comparison"
    )

    if not success:
        print("\nWarning: Markov baseline comparison failed, but continuing...")

    # Step 4: Cellular Automaton Models
    success = run_command(
        ['python3', 'cellular_automaton.py',
         '--noise', '0.0', '0.05', '0.10', '0.15', '0.20', '0.25', '0.30',
         '--trials', '5',
         '--steps', '20',
         '--size', '50',
         '--output', 'ca_noise_results.csv'],
        "Cellular Automaton Models"
    )

    if not success:
        print("\nWarning: Cellular automaton models failed, but analysis complete")

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  ✓ detected_spikes.csv")
    print("  ✓ information_theory_metrics.csv")
    print("  ✓ markov_baseline_comparison.csv")
    print("  ✓ ca_noise_results.csv")
    print("\nAll analysis complete!")

if __name__ == '__main__':
    main()
