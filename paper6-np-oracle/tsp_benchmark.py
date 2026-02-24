import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import time
import random
import itertools
from tqdm import tqdm

# ─────────────────────────────────────────────
# SEED FOR REPRODUCIBILITY
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ─────────────────────────────────────────────
# 1. GENERATE RANDOM TSP INSTANCES
# ─────────────────────────────────────────────
def generate_cities(n, seed=None):
    rng = np.random.default_rng(seed)
    return rng.random((n, 2))  # coordinates in [0,1] x [0,1]

def tour_length(tour, cities):
    total = 0.0
    n = len(tour)
    for i in range(n):
        a = cities[tour[i]]
        b = cities[tour[(i + 1) % n]]
        total += np.linalg.norm(a - b)
    return total

# ─────────────────────────────────────────────
# 2. BRUTE FORCE (exact NP search, n <= 11)
# ─────────────────────────────────────────────
def brute_force_tsp(cities):
    n = len(cities)
    best_length = float('inf')
    best_tour = None
    for perm in itertools.permutations(range(1, n)):
        tour = [0] + list(perm)
        length = tour_length(tour, cities)
        if length < best_length:
            best_length = length
            best_tour = tour
    return best_tour, best_length

# ─────────────────────────────────────────────
# 3. NEAREST NEIGHBOUR HEURISTIC (classical P)
# ─────────────────────────────────────────────
def nearest_neighbour_tsp(cities):
    n = len(cities)
    unvisited = set(range(n))
    tour = [0]
    unvisited.remove(0)
    while unvisited:
        current = tour[-1]
        nearest = min(unvisited, key=lambda x: np.linalg.norm(cities[current] - cities[x]))
        tour.append(nearest)
        unvisited.remove(nearest)
    return tour, tour_length(tour, cities)

# ─────────────────────────────────────────────
# 4. MYCELIAL GROWTH SIMULATION
#    Models fungal network growth as a 
#    pheromone-diffusion / nutrient-flow process
#    analogous to Physarum and S. commune 
#    travelling-wave optimization
# ─────────────────────────────────────────────
def mycelial_tsp(cities, iterations=300, decay=0.92, reinforce=2.5, alpha=1.2, beta=2.0):
    """
    Bio-inspired mycelial approximation of TSP.
    
    Models:
      - Nutrient gradient diffusion (pheromone trails)
      - Parallel hyphal growth to multiple nodes
      - Decay of inactive pathways (apoptosis analogue)
      - Reinforcement of high-flow corridors
        (analogous to betweenness-centrality trunks
         observed in Oyarte Galvez et al., Nature 2025)
    """
    n = len(cities)
    
    # Distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = np.linalg.norm(cities[i] - cities[j])
    
    # Pheromone matrix (mycelium trail strength)
    pheromone = np.ones((n, n)) * 0.1
    
    best_tour = None
    best_length = float('inf')
    
    for iteration in range(iterations):
        # Each iteration = one hyphal growth wave
        tours = []
        lengths = []
        
        # Run multiple parallel hyphal explorers (parallelism of mycelium)
        num_explorers = max(5, n // 2)
        for _ in range(num_explorers):
            start = random.randint(0, n - 1)
            visited = [start]
            unvisited = set(range(n)) - {start}
            
            while unvisited:
                current = visited[-1]
                # Probability based on pheromone (alpha) + distance heuristic (beta)
                scores = {}
                for j in unvisited:
                    trail = pheromone[current][j] ** alpha
                    visibility = (1.0 / dist[current][j]) ** beta
                    scores[j] = trail * visibility
                
                total = sum(scores.values())
                probs = {j: scores[j] / total for j in scores}
                
                # Stochastic selection (non-deterministic, like hyphal tip growth)
                r = random.random()
                cumulative = 0.0
                chosen = list(unvisited)[-1]
                for j, p in probs.items():
                    cumulative += p
                    if r <= cumulative:
                        chosen = j
                        break
                
                visited.append(chosen)
                unvisited.remove(chosen)
            
            length = tour_length(visited, cities)
            tours.append(visited)
            lengths.append(length)
            
            if length < best_length:
                best_length = length
                best_tour = visited[:]
        
        # Pheromone decay (pathway pruning - apoptosis analogue)
        pheromone *= decay
        
        # Reinforce trails used by best tours this iteration
        best_idx = np.argmin(lengths)
        best_this_iter = tours[best_idx]
        for i in range(n):
            a = best_this_iter[i]
            b = best_this_iter[(i + 1) % n]
            deposit = reinforce / lengths[best_idx]
            pheromone[a][b] += deposit
            pheromone[b][a] += deposit
        
        # Clamp pheromone to prevent overflow
        pheromone = np.clip(pheromone, 0.01, 20.0)
    
    return best_tour, best_length


# ─────────────────────────────────────────────
# 5. RUN FULL BENCHMARK
# ─────────────────────────────────────────────
def run_benchmark():
    sizes = [6, 8, 10, 20, 50, 100]
    results = []
    
    print("\n" + "="*70)
    print("  MYCELIUM vs P (Nearest Neighbour) vs BRUTE FORCE TSP BENCHMARK")
    print("  Paper 6 — Chowdhury 2026 — Biological NP Oracle Experiment")
    print("="*70 + "\n")
    
    for n in tqdm(sizes, desc="Running benchmark"):
        cities = generate_cities(n, seed=SEED + n)
        row = {'n': n}
        
        # --- BRUTE FORCE (only for small n) ---
        if n <= 10:
            t0 = time.perf_counter()
            _, bf_len = brute_force_tsp(cities)
            t1 = time.perf_counter()
            row['brute_force_length'] = bf_len
            row['brute_force_time_s'] = t1 - t0
            row['optimal'] = bf_len
        else:
            row['brute_force_length'] = None
            row['brute_force_time_s'] = None
            row['optimal'] = None
        
        # --- NEAREST NEIGHBOUR ---
        t0 = time.perf_counter()
        _, nn_len = nearest_neighbour_tsp(cities)
        t1 = time.perf_counter()
        row['nn_length'] = nn_len
        row['nn_time_s'] = t1 - t0
        
        # --- MYCELIAL SIMULATION ---
        iters = 200 if n <= 20 else 400
        t0 = time.perf_counter()
        _, myc_len = mycelial_tsp(cities, iterations=iters)
        t1 = time.perf_counter()
        row['mycelial_length'] = myc_len
        row['mycelial_time_s'] = t1 - t0
        
        # --- EFFICIENCY RATIOS ---
        if row['optimal']:
            row['nn_vs_optimal_%'] = round((nn_len / row['optimal'] - 1) * 100, 2)
            row['mycelial_vs_optimal_%'] = round((myc_len / row['optimal'] - 1) * 100, 2)
        else:
            row['nn_vs_optimal_%'] = None
            row['mycelial_vs_optimal_%'] = None
        
        results.append(row)
        
        print(f"\n  n={n:>3} cities")
        print(f"    Brute force:      {row['brute_force_length']:.4f}" if row['brute_force_length'] else f"    Brute force:      N/A (n>{10})")
        print(f"    Nearest neighbour:{row['nn_length']:.4f}  [{row['nn_time_s']*1000:.2f} ms]")
        print(f"    Mycelial model:   {row['mycelial_length']:.4f}  [{row['mycelial_time_s']:.3f} s]")
        if row['mycelial_vs_optimal_%'] is not None:
            print(f"    Mycelial excess vs optimal: +{row['mycelial_vs_optimal_%']}%")
            print(f"    NN excess vs optimal:       +{row['nn_vs_optimal_%']}%")
    
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 6. PLOT RESULTS
# ─────────────────────────────────────────────
def plot_results(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Mycelium Network as Physical NP Oracle — TSP Benchmark\n"
        "Paper 6: Chowdhury 2026",
        fontsize=13, fontweight='bold'
    )
    
    # --- Plot 1: Tour length comparison ---
    ax1 = axes[0]
    ax1.plot(df['n'], df['nn_length'], 'bs--', label='Nearest Neighbour (classical P)', linewidth=1.5)
    ax1.plot(df['n'], df['mycelial_length'], 'g^-', label='Mycelial Model (biological)', linewidth=2)
    bf = df[df['brute_force_length'].notna()]
    ax1.plot(bf['n'], bf['brute_force_length'], 'r*-', label='Brute Force (exact NP)', linewidth=1.5, markersize=10)
    ax1.set_xlabel('Number of cities (n)', fontsize=11)
    ax1.set_ylabel('Tour length (normalised units)', fontsize=11)
    ax1.set_title('Tour Quality: Mycelial vs Classical vs Exact', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Computation time ---
    ax2 = axes[1]
    ax2.plot(df['n'], df['nn_time_s'] * 1000, 'bs--', label='Nearest Neighbour (ms)', linewidth=1.5)
    ax2.plot(df['n'], df['mycelial_time_s'] * 1000, 'g^-', label='Mycelial Model (ms)', linewidth=2)
    bf_t = df[df['brute_force_time_s'].notna()]
    ax2.plot(bf_t['n'], bf_t['brute_force_time_s'] * 1000, 'r*-', label='Brute Force (ms)', linewidth=1.5, markersize=10)
    ax2.set_xlabel('Number of cities (n)', fontsize=11)
    ax2.set_ylabel('Computation time (ms)', fontsize=11)
    ax2.set_title('Computation Time Scaling', fontsize=11)
    ax2.set_yscale('log')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('tsp_benchmark_results.png', dpi=150, bbox_inches='tight')
    print("\n  [SAVED] tsp_benchmark_results.png")
    plt.show()


# ─────────────────────────────────────────────
# 7. SAVE CSV
# ─────────────────────────────────────────────
def save_csv(df):
    df.to_csv('tsp_benchmark_data.csv', index=False)
    print("  [SAVED] tsp_benchmark_data.csv")
    print("\n  Full results table:")
    print(df.to_string(index=False))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = run_benchmark()
    save_csv(df)
    plot_results(df)
    print("\n  BENCHMARK COMPLETE. Two files saved:")
    print("  - tsp_benchmark_results.png  (for Paper 6 figures)")
    print("  - tsp_benchmark_data.csv     (for Paper 6 Results table)")