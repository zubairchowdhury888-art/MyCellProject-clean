import numpy as np
import json
from datetime import datetime

# Model B: Syntactic Coupling (CTMU-inspired)
# Hyphae optimize for INFORMATION TOPOLOGY, not just nutrients
# Global network state influences local branching decisions

class SyntacticMycelium:
    def __init__(self, grid_size=50, n_hyphae=100, steps=500, coupling_strength=0.3):
        self.grid_size = grid_size
        self.n_hyphae = n_hyphae
        self.steps = steps
        self.coupling_strength = coupling_strength  # How much global field influences local choice
        self.grid = np.zeros((grid_size, grid_size))
        self.nutrient_map = self.init_nutrients()
        self.hyphae_positions = [(grid_size//2, grid_size//2)]
        self.spikes = []
        self.global_info_field = np.zeros((grid_size, grid_size))  # Information topology
        
    def init_nutrients(self):
        x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        source1 = np.exp(-((x-10)**2 + (y-10)**2)/50)
        source2 = np.exp(-((x-40)**2 + (y-40)**2)/50)
        return source1 + source2
    
    def update_info_field(self):
        # Global information field: regions with LOW hyphal density get HIGHER info value
        # (encourages exploration, error-correction topology)
        for pos in self.hyphae_positions[-self.n_hyphae:]:
            x, y = pos
            self.grid[y, x] += 1
        
        # Invert density to create exploration gradient
        self.global_info_field = 1.0 / (self.grid + 0.1)
    
    def grow_step(self):
        self.update_info_field()
        new_positions = []
        
        for pos in self.hyphae_positions[-self.n_hyphae:]:
            x, y = pos
            neighbors = [
                (x+1,y), (x-1,y), (x,y+1), (x,y-1),
                (x+1,y+1), (x+1,y-1), (x-1,y+1), (x-1,y-1)
            ]
            valid = [(nx,ny) for nx,ny in neighbors 
                     if 0<=nx<self.grid_size and 0<=ny<self.grid_size]
            
            if valid:
                # Combine nutrient gradient + global information field
                nutrients = np.array([self.nutrient_map[ny,nx] for nx,ny in valid])
                info_values = np.array([self.global_info_field[ny,nx] for nx,ny in valid])
                
                # Weighted combination: local nutrients + global info topology
                combined = (1 - self.coupling_strength) * nutrients + self.coupling_strength * info_values
                probs = combined / np.sum(combined)
                
                choice = np.random.choice(len(valid), p=probs)
                new_positions.append(valid[choice])
        
        self.hyphae_positions.extend(new_positions)
        
        # Electrical spikes ANTICIPATE branching (Δt < 0)
        # Spike precedes growth by accessing global field
        anticipatory_signal = np.mean([self.global_info_field[ny,nx] 
                                       for nx,ny in new_positions[:10]]) if new_positions else 0
        spike_amplitude = anticipatory_signal * 10 + np.random.normal(0, 0.5)
        self.spikes.append(spike_amplitude)
    
    def run(self):
        for _ in range(self.steps):
            self.grow_step()
        return self.compute_metrics()
    
    def compute_metrics(self):
        unique_pos = len(set(self.hyphae_positions))
        branching_ratio = unique_pos / len(self.hyphae_positions)
        
        spike_array = np.array(self.spikes)
        fidelity = np.mean(np.abs(spike_array))
        
        # Topology score: how well does network spread vs. clump?
        positions = np.array(list(set(self.hyphae_positions)))
        if len(positions) > 1:
            distances = np.linalg.norm(positions[:, None] - positions, axis=2)
            topology_score = np.mean(distances)  # Higher = better spreading
        else:
            topology_score = 0.0
        
        return {
            "model": "Syntactic_Coupling",
            "coupling_strength": self.coupling_strength,
            "branching_ratio": float(branching_ratio),
            "fidelity": float(fidelity),
            "spike_variance": float(np.var(spike_array)),
            "topology_score": float(topology_score),
            "n_hyphae_final": len(self.hyphae_positions),
            "timestamp": datetime.now().isoformat()
        }

# Run Model B
print("Running Model B: Syntactic Coupling (CTMU)...")
print("=" * 60)
np.random.seed(42)
model_b = SyntacticMycelium(grid_size=50, n_hyphae=100, steps=500, coupling_strength=0.3)
results_b = model_b.run()

print("\nModel B (Syntactic Coupling) Results:")
print(json.dumps(results_b, indent=2))
print("\n" + "=" * 60)