import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import sys
import os
sys.path.append(os.path.dirname((os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

# Assuming DATA_DIR is defined in your environment or you can replace it with the actual path
from poolagent import DATA_DIR, VISUALISATIONS_DIR

def load_results() -> Dict[str, Any]:
    with open(f"{DATA_DIR}/ablation_results.json", 'r') as f:
        return json.load(f)

def process_results(results: Dict[str, Any]):
    base_avg = results['base_model']['average']
    base_std = results['base_model']['std_dev']

    value_features = []
    value_diffs = []
    value_stds = []

    difficulty_features = []
    difficulty_diffs = []
    difficulty_stds = []

    for key, value in results.items():
        if key.startswith('value_holdout_'):
            feature_index = int(key.split('_')[-1])
            value_features.append(feature_index)
            value_diffs.append(value['average'] - base_avg)
            value_stds.append(np.sqrt(value['std_dev']**2 + base_std**2))
        elif key.startswith('difficulty_holdout_'):
            feature_index = int(key.split('_')[-1])
            difficulty_features.append(feature_index)
            difficulty_diffs.append(value['average'] - base_avg)
            difficulty_stds.append(np.sqrt(value['std_dev']**2 + base_std**2))

    return (value_features, value_diffs, value_stds), (difficulty_features, difficulty_diffs, difficulty_stds)

def plot_feature_importance(features, diffs, stds, title, ylabel):
    plt.figure(figsize=(12, 6))
    
    # Create a color array based on the difference values
    colors = ['green' if diff <= 0 else 'red' for diff in diffs]
    
    bars = plt.bar(features, diffs, yerr=stds, capsize=5, color=colors)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.title(title)
    plt.xlabel('Feature Index')
    plt.ylabel(ylabel)
    plt.xticks(features)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar, diff, std in zip(bars, diffs, stds):
        height = bar.get_height()
        text_color = 'black'  # Use black for all text to ensure readability
        va = 'bottom' if height >= 0 else 'top'
        plt.text(bar.get_x() + bar.get_width()/2, height,
                 f'{diff:.4f}\n±{std:.4f}', 
                 ha='center', va=va,
                 color=text_color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{VISUALISATIONS_DIR}/{title.lower().replace(' ', '_')}.png")
    plt.close()

def main():
    results = load_results()
    value_data, difficulty_data = process_results(results)

    plot_feature_importance(*value_data, 
                            "Value Feature Importance", 
                            "Difference in Loss (Holdout - Base)")
    
    plot_feature_importance(*difficulty_data, 
                            "Difficulty Feature Importance", 
                            "Difference in Loss (Holdout - Base)")

    print(f"Plots have been saved in {VISUALISATIONS_DIR}")

if __name__ == "__main__":
    main()