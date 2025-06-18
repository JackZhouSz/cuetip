import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os
from scipy import stats
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poolagent.pool import Pool
from poolagent.utils import SKILL_LEVELS, State, random_ball_shot, Agent, Event, blur_shot

def analyze_params(get_param, n_samples=10000000):
    samples = defaultdict(list)
    for _ in range(n_samples):
        params = get_param()
        for key, value in params.items():
            samples[key].append(value)
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.ravel()
    
    limits = {
        "V0": (0.25, 4),
        "phi": (0, 360),
        "theta": (5, 60),
        "a": (-0.25, 0.25),
        "b": (-0.25, 0.25)
    }
    
    for idx, (param, values) in enumerate(samples.items()):
        values = np.array(values)
        ax = axs[idx]
        
        # Plot histogram
        bins = 50
        ax.hist(values, bins=bins, density=True)
        ax.set_title(f'{param} Distribution')
        ax.axvline(limits[param][0], color='r', linestyle='--', alpha=0.5)
        ax.axvline(limits[param][1], color='r', linestyle='--', alpha=0.5)
        
        # Add statistics
        mean = np.mean(values)
        std = np.std(values)
        
        # Calculate expected uniform distribution values
        expected_mean = (limits[param][1] + limits[param][0]) / 2
        expected_std = (limits[param][1] - limits[param][0]) / np.sqrt(12)
        
        # Perform Kolmogorov-Smirnov test for uniformity
        # Normalize values to [0,1] range for test
        normalized_values = (values - limits[param][0]) / (limits[param][1] - limits[param][0])
        ks_stat, p_value = stats.kstest(normalized_values, 'uniform')
        
        # Calculate histogram uniformity
        hist, _ = np.histogram(values, bins=bins)
        expected_count = n_samples/bins
        chi_squared = np.sum((hist - expected_count)**2 / expected_count)
        
        stats_text = (
            f'Mean: {mean:.2f} (Expected: {expected_mean:.2f})\n'
            f'Std: {std:.2f} (Expected: {expected_std:.2f})\n'
            f'KS test p-value: {p_value:.4f}\n'
            f'χ² value: {chi_squared:.1f}'
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=8)
        
        # Print violations and uniformity measures
        violations = np.sum((values < limits[param][0]) | (values > limits[param][1]))
        print(f"\n{param}:")
        print(f"Violations: {violations} ({violations/n_samples*100:.2f}%)")
        print(f"Mean deviation from uniform: {abs(mean - expected_mean):.3f}")
        print(f"Std deviation from uniform: {abs(std - expected_std):.3f}")
        print(f"KS test p-value: {p_value:.4f} ({'uniform' if p_value > 0.05 else 'not uniform'})")
        print(f"χ² value: {chi_squared:.1f} (lower is more uniform)")
    
    plt.tight_layout()
    plt.show()

# Usage:
env = Pool()
analyze_params(env.random_params)