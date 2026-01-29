import numpy as np
import json
from datetime import datetime

# Model C: Ruliad Sampling (Wolfram-inspired)
# Multiple computational threads run simultaneously
# Mycelium samples different threads based on context

class RuliadMycelium:
    def __init__(self, grid_size=50, n_hyphae=100, steps=500, n_threads=5):
        self.grid_size = grid_size
        self.n_hyphae = n_hyphae
        self.steps = steps
        self.n_threads = n_threads  # Number of parallel computational threads
        self.nutrient_map = self.init_nutrients()
        self.hyphae_positions = [(grid_size//2, grid_size//2)]
        self.spikes = []
        self.thread_states = [np.random.rand(grid_size, grid_size) for _ in range(n_threads)]
        
    def init_nutrients(self):
        x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        source1 = np.exp(-((x-10)**2 + (y-10)**2)/50)
        source2 = np.exp(-((x-40)**2 + (y-40)**2)/50)
        return source1 + source2
    
    def sample_ruliad(self, position):
        """Sample multiple computational threads and select based on context"""
        x, y = position
        # Each thread represents a different "slice" of the ruliad
        thread_samples = [thread[y, x] for thread in self.thread_states]
        
        # Context-dependent sampling: choose thread with highest local coherence
        selected_thread_idx = np.argmax(thread_samples)
        return selected_thread_idx, thread_samples[selected_thread_idx]
    
    def update_threads(self):
        """Threads evolve independently (multiway branching)"""
        for i, thread in enumerate(self.thread_states):
            # Simple CA-like update: each thread evolves
            self.thread_states[i] = np.roll(thread, shift=i-2, axis=0) * 0.99 + np.random.rand(self.grid_size, self.grid_size) * 0.01
    
    def grow_step(self):
        self.update_threads()
        new_positions = []
        thread_selections = []
        
        for pos in self.hyphae_positions[-self.n_hyphae:]:
            x, y = pos
            neighbors = [
                (x+1,y), (x-1,y), (x,y+1), (x,y-1),
                (x+1,y+1), (x+1,y-1), (x-1,y+1), (x-1,y-1)
            ]
            valid = [(nx,ny) for nx,ny in neighbors 
                     if 0<=nx<self.grid_size and 0<=ny<self.grid_size]
            
            if valid:
                # Sample ruliad for each neighbor
                ruliad_weights = []
                for nx, ny in valid:
                    thread_idx, thread_val = self.sample_ruliad((nx, ny))
                    nutrient_val = self.nutrient_map[ny, nx]
                    # Combine nutrient + ruliad thread value
                    combined = 0.5 * nutrient_val + 0.5 * thread_val
                    ruliad_weights.append(combined)
                    thread_selections.append(thread_idx)
                
                probs = np.array(ruliad_weights) / np.sum(ruliad_weights)
                choice = np.random.choice(len(valid), p=probs)
                new_positions.append(valid[choice])
        
        self.hyphae_positions.extend(new_positions)
        
        # Spikes reflect cross-thread coherence (nonlocal correlation signature)
        if len(thread_selections) > 0:
            thread_diversity = len(set(thread_selections)) / self.n_threads
            spike_amplitude = (1 - thread_diversity) * 20  # Low diversity = high coherence = strong spike
        else:
            spike_amplitude = 0
        
        self.spikes.append(spike_amplitude + np.random.normal(0, 1))
    
    def run(self):
        for _ in range(self.steps):
            self.grow_step()
        return self.compute_metrics()
    
    def compute_metrics(self):
        unique_pos = len(set(self.hyphae_positions))
        branching_ratio = unique_pos / len(self.hyphae_positions)
        
        spike_array = np.array(self.spikes)
        fidelity = np.mean(np.abs(spike_array))
        
        positions = np.array(list(set(self.hyphae_positions)))
        if len(positions) > 1:
            distances = np.linalg.norm(positions[:, None] - positions, axis=2)
            topology_score = np.mean(distances)
        else:
            topology_score = 0.0
        
        # Coherence metric: how often do spikes show low-variance periods?
        coherence_score = 1.0 / (np.var(spike_array) + 1e-6)
        
        return {
            "model": "Ruliad_Sampling",
            "n_threads": self.n_threads,
            "branching_ratio": float(branching_ratio),
            "fidelity": float(fidelity),
            "spike_variance": float(np.var(spike_array)),
            "topology_score": float(topology_score),
            "coherence_score": float(coherence_score),
            "n_hyphae_final": len(self.hyphae_positions),
            "timestamp": datetime.now().isoformat()
        }

# Run Model C
print("Running Model C: Ruliad Sampling (Wolfram)...")
print("=" * 60)
np.random.seed(42)
model_c = RuliadMycelium(grid_size=50, n_hyphae=100, steps=500, n_threads=5)
results_c = model_c.run()

print("\nModel C (Ruliad Sampling) Results:")
print(json.dumps(results_c, indent=2))
print("\n" + "=" * 60)