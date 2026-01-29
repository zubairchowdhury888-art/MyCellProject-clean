import numpy as np
import json
from datetime import datetime

# Model A: Classical Local Chemotaxis
# Hyphal growth driven purely by nutrient gradients + noise

class ClassicalMycelium:
    def __init__(self, grid_size=50, n_hyphae=100, steps=500):
        self.grid_size = grid_size
        self.n_hyphae = n_hyphae
        self.steps = steps
        self.grid = np.zeros((grid_size, grid_size))
        self.nutrient_map = self.init_nutrients()
        self.hyphae_positions = [(grid_size//2, grid_size//2)]
        self.spikes = []
        
    def init_nutrients(self):
        # Gaussian nutrient sources
        x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        source1 = np.exp(-((x-10)**2 + (y-10)**2)/50)
        source2 = np.exp(-((x-40)**2 + (y-40)**2)/50)
        return source1 + source2
    
    def grow_step(self):
        # Each hypha extends toward highest local nutrient gradient
        new_positions = []
        for pos in self.hyphae_positions[-self.n_hyphae:]:
            x, y = pos
            # Sample 8 neighbors
            neighbors = [
                (x+1,y), (x-1,y), (x,y+1), (x,y-1),
                (x+1,y+1), (x+1,y-1), (x-1,y+1), (x-1,y-1)
            ]
            valid = [(nx,ny) for nx,ny in neighbors 
                     if 0<=nx<self.grid_size and 0<=ny<self.grid_size]
            
            if valid:
                # Probabilistic choice weighted by nutrient concentration
                nutrients = np.array([self.nutrient_map[ny,nx] for nx,ny in valid])
                probs = nutrients / np.sum(nutrients)  # Fixed: normalize properly
                choice = np.random.choice(len(valid), p=probs)
                new_positions.append(valid[choice])
        
        self.hyphae_positions.extend(new_positions)
        
        # Electrical spike follows growth (lagged)
        spike_amplitude = len(new_positions) * np.random.normal(0.1, 0.05)
        self.spikes.append(spike_amplitude)
    
    def run(self):
        for _ in range(self.steps):
            self.grow_step()
        return self.compute_metrics()
    
    def compute_metrics(self):
        # Topology: measure branching ratio
        unique_pos = len(set(self.hyphae_positions))
        branching_ratio = unique_pos / len(self.hyphae_positions)
        
        # Spike statistics
        spike_array = np.array(self.spikes)
        fidelity = np.mean(np.abs(spike_array))
        
        return {
            "model": "Classical_Local",
            "branching_ratio": float(branching_ratio),
            "fidelity": float(fidelity),
            "spike_variance": float(np.var(spike_array)),
            "n_hyphae_final": len(self.hyphae_positions),
            "timestamp": datetime.now().isoformat()
        }

# Run Model A baseline
print("Running Model A: Classical Local Chemotaxis...")
print("=" * 60)
np.random.seed(42)
model_a = ClassicalMycelium(grid_size=50, n_hyphae=100, steps=500)
results_a = model_a.run()

print("\nModel A (Classical Local) Results:")
print(json.dumps(results_a, indent=2))
print("\n" + "=" * 60)
print("Baseline established. Ready for Models B & C comparison.")