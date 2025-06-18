import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from scipy import stats
import sys
import os
sys.path.append(os.path.dirname((os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

# Assuming DATA_DIR is defined in your environment or you can replace it with the actual path
from poolagent import DATA_DIR, VISUALISATIONS_DIR

def load_results() -> Dict[str, Any]:
    with open(f"{DATA_DIR}/ablation_results.json", 'r') as f:
        return json.load(f)

def process_results(results: Dict[str, Any]):
    base_values = results['base_model']['runs']

    value_features = []
    value_distributions = []

    difficulty_features = []
    difficulty_distributions = []

    for key, value in results.items():
        if key.startswith('value_holdout_'):
            feature_index = int(key.split('_')[-1])
            value_features.append(feature_index)
            value_distributions.append(value['runs'])
        elif key.startswith('difficulty_holdout_'):
            feature_index = int(key.split('_')[-1])
            difficulty_features.append(feature_index)
            difficulty_distributions.append(value['runs'])

    return (value_features, value_distributions), (difficulty_features, difficulty_distributions), base_values

def plot_distribution_comparison(features, distributions, base_values, title, min_max):
    if not os.path.exists(f"{VISUALISATIONS_DIR}/ablation"):
        os.makedirs(f"{VISUALISATIONS_DIR}/ablation")
    for feature, distribution in zip(features, distributions):
        plt.figure(figsize=(12, 6))
        
        # Plot kernel density estimation
        kde_base = stats.gaussian_kde(base_values)
        kde_feature = stats.gaussian_kde(distribution)
        x_range = np.linspace(min_max[0], min_max[1], 
                              100)
        
        plt.plot(x_range, kde_base(x_range), label='Base Model', color='blue')
        plt.plot(x_range, kde_feature(x_range), label=f'Feature {feature}', color='red')
        
        plt.title(f"{title} - Feature {feature} vs Base Model")
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.ylim(0,1000)
        
        # Add summary statistics
        plt.text(0.05, 0.95, f"Base Mean: {np.mean(base_values):.4f}\nBase Std: {np.std(base_values):.4f}", 
                 transform=plt.gca().transAxes, verticalalignment='top', color='blue')
        plt.text(0.05, 0.85, f"Feature Mean: {np.mean(distribution):.4f}\nFeature Std: {np.std(distribution):.4f}", 
                 transform=plt.gca().transAxes, verticalalignment='top', color='red')

        plt.tight_layout()
        plt.savefig(f"{VISUALISATIONS_DIR}/ablation/{title.lower().replace(' ', '_')}_feature_{feature}.png")
        plt.close()

def main():
    results = load_results()
    value_data, difficulty_data, base_values = process_results(results)

    min_value = min(np.min(base_values), np.min(np.concatenate(value_data[1])))
    min_difficulty = min(np.min(base_values), np.min(np.concatenate(difficulty_data[1])))
    max_value = max(np.max(base_values), np.max(np.concatenate(value_data[1])))
    max_difficulty = max(np.max(base_values), np.max(np.concatenate(difficulty_data[1])))
    min_max = (min(min_value, min_difficulty), max(max_value, max_difficulty))

    plot_distribution_comparison(*value_data, base_values, "Value Feature Distribution", min_max)
    plot_distribution_comparison(*difficulty_data, base_values, "Difficulty Feature Distribution", min_max)

    print(f"Plots have been saved in {VISUALISATIONS_DIR}")

if __name__ == "__main__":
    main()