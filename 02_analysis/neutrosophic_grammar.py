import pandas as pd
import numpy as np
df = pd.read_csv('03_results/combined_grammar_summary.csv')
print("Raw head:\n", df[['dataset', 'n_nonterminals', 'depth_std', 'compression_ratio']].head())
df['nonterminals'] = df['n_nonterminals']
def neutrosophic_metrics(row):
    T = row['nonterminals'] / 89.0  # Truth: syntactic capacity
    I = row['depth_std'] / 50000    # Indeterminacy: hierarchy variance
    F = 1 - (row['compression_ratio'] / 0.902)  # Falsity: vs. optimal
    H_N = -(T*np.log2(T+1e-10) + I*np.log2(I+1e-10) + F*np.log2(F+1e-10))
    return pd.Series({'T':T, 'I':I, 'F':F, 'H_N':H_N, 'hyper_truth':(T+I+F > 1)})
df = df.join(df.apply(neutrosophic_metrics, axis=1))
df.to_csv('03_results/H_N.csv', index=False)
print(df[['dataset', 'n_nonterminals', 'H_N', 'hyper_truth', 'T', 'I', 'F']])