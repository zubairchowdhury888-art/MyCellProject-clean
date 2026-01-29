
# Analysis Scripts for Syntactic Information Processing in Fungal Networks

**Author:** Zubair E Chowdhury  
**Paper:** Syntactic Information Processing in Fungal Electrical Networks: Evidence from Schizophyllum commune and Computational Modeling  
**Date:** December 2025  
**License:** MIT

## Overview

This package contains the complete Python analysis pipeline used in the paper. All scripts can be run independently or via the master script.

## Files

1. **spike_detection.py** - Detect spikes using derivative thresholding
2. **information_theory.py** - Calculate Shannon entropy, Kolmogorov complexity, ISI metrics
3. **markov_baseline.py** - Generate Markov chains and compare to empirical data
4. **cellular_automaton.py** - Test 3 CA models under progressive noise
5. **run_full_analysis.py** - Master script that runs entire pipeline

## Requirements

```bash
pip install numpy pandas scipy matplotlib
```

Or use the provided requirements.txt:

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Run complete pipeline

```bash
python run_full_analysis.py your_data.csv --duration 3533 --channel Channel_7
```

### Option 2: Run individual scripts

```bash
# Step 1: Detect spikes
python spike_detection.py your_data.csv --threshold -0.15 --channel Channel_7

# Step 2: Calculate information metrics
python information_theory.py detected_spikes.csv --duration 3533

# Step 3: Generate Markov baseline
python markov_baseline.py detected_spikes.csv --duration 3533 --n_trials 20

# Step 4: Run cellular automaton models
python cellular_automaton.py --noise 0.0 0.05 0.10 0.15 0.20 0.25 0.30
```

## Input Data Format

CSV file with columns:
- `Time` (seconds)
- `Channel_1`, `Channel_3`, `Channel_5`, `Channel_7`, `Channel_9` (voltages in mV)

Example:
```
Time,Channel_1,Channel_3,Channel_5,Channel_7,Channel_9
0,-0.0105,0.0176,0.0038,-0.1472,0.0606
1,-0.0108,0.0180,0.0040,-0.1475,0.0608
...
```

## Output Files

- `detected_spikes.csv` - List of detected spike events
- `information_theory_metrics.csv` - Entropy and complexity metrics
- `markov_baseline_comparison.csv` - Statistical comparison results
- `ca_noise_results.csv` - Cellular automaton simulation results

## Reproducing Paper Results

To reproduce the results from the paper:

```bash
# Using the fast spike window (59-minute segment)
python run_full_analysis.py fast_spike_window.csv --duration 3533 --channel Channel_7
```

Expected results:
- 395 spikes detected
- Shannon entropy (10s): ~4.84 bits
- Kolmogorov complexity: ~0.098 (90.2% compressible)
- Markov comparison: p = 0.56 (entropy), p = 0.007 (complexity)
- Syntactic CA advantage: ~419× capacity

## Customization

### Adjust spike detection sensitivity

```bash
python spike_detection.py data.csv --threshold -0.10  # More sensitive
python spike_detection.py data.csv --threshold -0.20  # Less sensitive
```

### Change Markov trials

```bash
python markov_baseline.py data.csv --duration 3533 --n_trials 50
```

### Test different noise levels

```bash
python cellular_automaton.py --noise 0.0 0.1 0.2 0.3 0.4 0.5
```

## Troubleshooting

**"No spikes detected"**
- Try lowering the threshold: `--threshold -0.10`
- Check your channel selection: `--channel Channel_X`
- Verify data format matches expected structure

**"File not found"**
- Ensure all .py files are in the same directory
- Check input file path is correct
- Use absolute paths if needed

**Memory errors with large files**
- Process data in chunks
- Reduce grid size for CA: `--size 30`
- Reduce CA trials: `--trials 3`

## Citation

If you use these scripts, please cite:

```
Chowdhury, Z.E. (2025). Syntactic Information Processing in Fungal Electrical 
Networks: Evidence from Schizophyllum commune and Computational Modeling. 
Zenodo. https://doi.org/[DOI]
```

## Contact

Questions? Email: zubairchowdhury888@gmail.com  
Website: https://gomaa.uk  
Substack: https://grumpyjournalist.substack.com

## License

MIT License - see LICENSE file for details
