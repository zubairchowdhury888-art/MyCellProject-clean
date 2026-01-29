import numpy as np
import json
from datetime import datetime

# Test cross-colony correlation for all three models
# Two isolated colonies with NO physical connection

def run_colony(model_class, seed_offset=0, **kwargs):
    """Run a single colony and return spike pattern"""
    np.random.seed(42 + seed_offset)
    if model_class == "classical":
        from model_a_classical import ClassicalMycelium as Model
    elif model_class == "syntactic":
        from model_b_syntactic import SyntacticMycelium as Model
    elif model_class == "ruliad":
        from model_c_ruliad import RuliadMycelium as Model
    
    colony = Model(**kwargs)
    colony.run()
    return np.array(colony.spikes)

def compute_correlation(spikes_a, spikes_b):
    """Compute Pearson correlation between two spike trains"""
    min_len = min(len(spikes_a), len(spikes_b))
    spikes_a = spikes_a[:min_len]
    spikes_b = spikes_b[:min_len]
    
    corr = np.corrcoef(spikes_a, spikes_b)[0, 1]
    return corr

# Test all three models
models = ["classical", "syntactic", "ruliad"]
results = {}

print("Testing Cross-Colony Correlation (Isolated Colonies)")
print("=" * 70)

for model_name in models:
    print(f"\nTesting {model_name.upper()} model...")
    
    # Run two isolated colonies
    spikes_colony_a = run_colony(model_name, seed_offset=0, 
                                  grid_size=50, n_hyphae=100, steps=500)
    spikes_colony_b = run_colony(model_name, seed_offset=100,  # Different seed = isolated
                                  grid_size=50, n_hyphae=100, steps=500)
    
    corr_ab = compute_correlation(spikes_colony_a, spikes_colony_b)
    
    results[model_name] = {
        "model": model_name,
        "Corr_AB": float(corr_ab),
        "interpretation": "nonlocal coupling" if abs(corr_ab) > 0.7 else 
                         "weak field" if abs(corr_ab) > 0.5 else "local only"
    }
    
    print(f"  Colony A spikes (first 10): {spikes_colony_a[:10]}")
    print(f"  Colony B spikes (first 10): {spikes_colony_b[:10]}")
    print(f"  Corr_AB = {corr_ab:.4f}")
    print(f"  → {results[model_name]['interpretation']}")

print("\n" + "=" * 70)
print("\nFINAL CORRELATION SUMMARY:")
print(json.dumps(results, indent=2))
print("\n" + "=" * 70)
print("\nThresholds (pre-registered):")
print("  Corr_AB > 0.7  → Strong nonlocal coupling (ruliad access)")
print("  Corr_AB > 0.5  → Weak field coupling (syntactic substrate)")
print("  Corr_AB < 0.5  → Local dynamics only (classical)")