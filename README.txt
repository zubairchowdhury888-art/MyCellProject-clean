# Distinguishing Protocomputational Substrates in Mycelial Networks
## Syntactic Coupling Predicts Nonlocal Correlation

Author: Zubair Chowdhury
Date: December 28, 2025
Manuscript ID: BIOSYS-S-25-01456 (Submitted to BioSystems)

## Contents

This repository contains all simulation code and data for the paper.

### Code Files:
- model_a_classical.py - Classical local chemotaxis (null model)
- model_b_syntactic.py - CTMU syntactic coupling model
- model_c_ruliad.py - Wolfram ruliad sampling model
- test_correlation.py - Cross-colony correlation analysis

### Data Files:
- paper6_simulation_data.json - Complete simulation results

## Installation

Requirements:
- Python 3.14 or higher
- NumPy 2.4.0 or higher

Install dependencies:
```
python -m pip install numpy
```

## Usage

Run each model individually:
```
python model_a_classical.py
python model_b_syntactic.py
python model_c_ruliad.py
```

Run cross-colony correlation test:
```
python test_correlation.py
```

## Key Results

- Model A (Classical): Corr_AB = 0.040
- Model B (Syntactic): Corr_AB = 0.955
- Model C (Ruliad): Corr_AB = 0.085

Only syntactic coupling predicts strong nonlocal correlation (>0.7).

## Citation

If you use this code or data, please cite:

Chowdhury, Z.E. (2025). Distinguishing Protocomputational Substrates in Mycelial Networks: 
Syntactic Coupling Predicts Nonlocal Correlation. Preprint submitted to BioSystems.
Manuscript ID: BIOSYS-S-25-01456
DOI: [Will be added after Zenodo publication]

## License

Code: MIT License
Paper: CC BY 4.0

## Contact

Email: zubairchowdhury888@gmail.com
