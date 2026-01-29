#!/usr/bin/env python3
"""
cellular_automaton.py - Test syntactic vs Boolean information processing
Part of: Syntactic Information Processing in Fungal Networks (Chowdhury, 2025)

Usage:
    python cellular_automaton.py --noise 0.0 0.05 0.10 0.15 0.20 0.25 0.30

Requirements:
    numpy, pandas
"""

import numpy as np
import pandas as pd
import argparse

class CellularAutomaton:
    """Base class for 2D cellular automaton"""

    def __init__(self, size=50, initial_density=0.3):
        self.size = size
        self.grid = np.random.random((size, size)) < initial_density
        self.history = [self.grid.copy()]

    def get_neighborhood(self, i, j):
        """Get Moore neighborhood (8 neighbors)"""
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = (i + di) % self.size, (j + dj) % self.size
                neighbors.append(self.grid[ni, nj])
        return neighbors

    def step(self):
        """Override in subclasses"""
        pass

    def add_noise(self, noise_level):
        """Flip random bits with probability noise_level"""
        noise_mask = np.random.random(self.grid.shape) < noise_level
        self.grid = np.logical_xor(self.grid, noise_mask)


class RandomModel(CellularAutomaton):
    """Model 1: Random spiking (baseline)"""

    def step(self):
        self.grid = np.random.random(self.grid.shape) < 0.1
        self.history.append(self.grid.copy())


class BooleanLogicModel(CellularAutomaton):
    """Model 2: Boolean logic gates (standard computation)"""

    def step(self):
        new_grid = np.zeros_like(self.grid)

        for i in range(self.size):
            for j in range(self.size):
                neighbors = self.get_neighborhood(i, j)
                n_active = sum(neighbors)

                # Conway-like rules: (current AND ≥2 neighbors) OR (≥3 neighbors)
                if self.grid[i, j] and n_active >= 2:
                    new_grid[i, j] = True
                elif n_active >= 3:
                    new_grid[i, j] = True

        self.grid = new_grid
        self.history.append(self.grid.copy())


class SyntacticModel(CellularAutomaton):
    """Model 3: Error-correcting syntax (quantum-inspired)"""

    def _decode_triplet(self, triplet):
        """Majority vote decoding"""
        return sum(triplet) >= 2

    def step(self):
        new_grid = np.zeros_like(self.grid)

        for i in range(self.size):
            for j in range(self.size):
                neighbors = self.get_neighborhood(i, j)

                if len(neighbors) >= 3:
                    # Sample 3 random neighbors for triplet decoding
                    triplet_indices = np.random.choice(len(neighbors), 3, replace=False)
                    triplet = [neighbors[idx] for idx in triplet_indices]

                    # Decode using majority vote
                    consensus = self._decode_triplet(triplet)
                    neighbor_count = sum(neighbors)

                    # Update based on consensus
                    if consensus and neighbor_count >= 4:
                        new_grid[i, j] = True
                    elif not consensus and neighbor_count <= 2:
                        new_grid[i, j] = False
                    else:
                        # Maintain current state (error correction)
                        new_grid[i, j] = self.grid[i, j]
                else:
                    new_grid[i, j] = self.grid[i, j]

        self.grid = new_grid
        self.history.append(self.grid.copy())


def calculate_mutual_information(grid1, grid2):
    """Calculate mutual information between two states"""
    x = grid1.flatten()
    y = grid2.flatten()

    # Calculate joint probabilities
    p_00 = np.mean((x == 0) & (y == 0))
    p_01 = np.mean((x == 0) & (y == 1))
    p_10 = np.mean((x == 1) & (y == 0))
    p_11 = np.mean((x == 1) & (y == 1))

    p_x0 = p_00 + p_01
    p_x1 = p_10 + p_11
    p_y0 = p_00 + p_10
    p_y1 = p_01 + p_11

    mi = 0
    for p_xy, p_x, p_y in [(p_00, p_x0, p_y0), (p_01, p_x0, p_y1),
                            (p_10, p_x1, p_y0), (p_11, p_x1, p_y1)]:
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * np.log2(p_xy / (p_x * p_y))

    return mi


def run_experiment(model_name, ModelClass, noise_levels, n_trials, n_steps, grid_size):
    """Run noise resistance experiment for one model"""
    results = []

    for noise in noise_levels:
        for trial in range(n_trials):
            # Initialize model
            model = ModelClass(size=grid_size, initial_density=0.3)
            reference_state = model.grid.copy()

            # Run evolution with noise
            for step in range(n_steps):
                model.step()
                if noise > 0:
                    model.add_noise(noise)

            # Measure final state
            final_mi = calculate_mutual_information(reference_state, model.grid)
            final_activity = np.mean(model.grid)

            results.append({
                'Model': model_name,
                'Noise_Level': noise,
                'Trial': trial + 1,
                'Mutual_Information': final_mi,
                'Activity_Level': final_activity
            })

    return results


def main():
    parser = argparse.ArgumentParser(description='Run cellular automaton noise experiments')
    parser.add_argument('--noise', nargs='+', type=float, 
                        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                        help='Noise levels to test')
    parser.add_argument('--trials', type=int, default=5, help='Trials per condition')
    parser.add_argument('--steps', type=int, default=20, help='Evolution steps per trial')
    parser.add_argument('--size', type=int, default=50, help='Grid size (N×N)')
    parser.add_argument('--output', default='cellular_automaton_results.csv', help='Output filename')

    args = parser.parse_args()

    print("="*80)
    print("CELLULAR AUTOMATON NOISE RESISTANCE EXPERIMENT")
    print("="*80)
    print(f"\nParameters:")
    print(f"  Grid size: {args.size}×{args.size}")
    print(f"  Evolution steps: {args.steps}")
    print(f"  Trials per condition: {args.trials}")
    print(f"  Noise levels: {args.noise}")
    print(f"  Total simulations: {3 * len(args.noise) * args.trials}")

    all_results = []

    # Run experiments for all three models
    for model_name, ModelClass in [('Random', RandomModel), 
                                     ('Boolean', BooleanLogicModel),
                                     ('Syntactic', SyntacticModel)]:

        print(f"\nRunning {model_name} model...")
        results = run_experiment(
            model_name, ModelClass, args.noise, 
            args.trials, args.steps, args.size
        )
        all_results.extend(results)
        print(f"  ✓ Completed {len(results)} simulations")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output, index=False)

    print(f"\n✓ Results saved to: {args.output}")

    # Calculate and display summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    summary = results_df.groupby(['Model', 'Noise_Level']).agg({
        'Mutual_Information': ['mean', 'std'],
        'Activity_Level': 'mean'
    }).reset_index()

    print("\nMutual Information (mean ± std):")
    print(f"{'Noise':<10} {'Random':<20} {'Boolean':<20} {'Syntactic':<20}")
    print("-"*70)

    for noise in args.noise:
        row_data = summary[summary['Noise_Level'] == noise]

        random_mi = row_data[row_data['Model'] == 'Random'][('Mutual_Information', 'mean')].values[0]
        boolean_mi = row_data[row_data['Model'] == 'Boolean'][('Mutual_Information', 'mean')].values[0]
        syntactic_mi = row_data[row_data['Model'] == 'Syntactic'][('Mutual_Information', 'mean')].values[0]

        print(f"{noise:<10.2f} {random_mi:<20.6f} {boolean_mi:<20.6f} {syntactic_mi:<20.6f}")

    # Calculate capacity advantage
    syntactic_initial = results_df[(results_df['Model'] == 'Syntactic') & 
                                   (results_df['Noise_Level'] == 0.0)]['Mutual_Information'].mean()
    boolean_initial = results_df[(results_df['Model'] == 'Boolean') & 
                                 (results_df['Noise_Level'] == 0.0)]['Mutual_Information'].mean()

    if boolean_initial > 0:
        capacity_ratio = syntactic_initial / boolean_initial
        print(f"\nSyntactic capacity advantage: {capacity_ratio:.1f}×")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
