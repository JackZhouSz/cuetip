import json
import matplotlib.pyplot as plt
import re
import numpy as np
from pathlib import Path
import yaml
from scipy import stats

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poolagent.path import DATA_DIR, ROOT_DIR, VISUALISATIONS_DIR

PLOT_K = 3

def load_plot_config(config_path='../plot_config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Set matplotlib rcParams
    for key, value in config['rcParams'].items():
        plt.rcParams[key] = value
    
    # Set style
    plt.style.use(config['style'])
    
    return config

config = load_plot_config()

def extract_info_from_key(key):
    shot_match = re.search(r'shot_(\d+)', key)
    shot = int(shot_match.group(1)) if shot_match else None
    parts = key.split('_')
    agent = parts[0]
    model = parts[1]
    return agent, model, shot

def load_tasks():
    with open(f'{DATA_DIR}/shot_task_dataset.json', 'r') as f:
        data = json.load(f)
    return data

def load_results():
    results_dir = Path(f'{ROOT_DIR}/experiments/experiment_skill/results')
    latest_file = max(list(results_dir.glob('*.json')), key=lambda x: x.stat().st_mtime)
    with open(latest_file, 'r') as f:
        results = json.load(f)
    return results

def calculate_stats(difficulties):
    means = np.array([np.mean(d) if d else 0 for d in difficulties])
    errors = np.array([stats.sem(d) if d and len(d) > 1 else 0 for d in difficulties])
    return means, errors

def plot_difficulty_deltas_both():
    tasks = load_tasks()
    results = load_results()
    
    agents = sorted(list({extract_info_from_key(key)[0] for key in results.keys()}))
    first_shot_key = next(key for key in tasks if 'difficulties' in tasks[key])
    vector_length = len(tasks[first_shot_key]['difficulties'])
    
    fig, axes = plt.subplots(len(agents), 1, figsize=(config['figure_sizes']['default'][0], 
                                                     config['figure_sizes']['default'][1] * 0.8 * len(agents)))
    if len(agents) == 1:
        axes = [axes]
        
    fig.set_facecolor(config['aesthetics']['figure']['facecolor'])
    fig.suptitle('Comparison of Rule Difficulties Between Success Levels',
                 fontsize=config['fonts']['title']['size'] + 2,
                 fontweight=config['fonts']['title']['weight'],
                 y=1.02)
    
    MAX_VAL = 0.25
    
    for agent_idx, agent in enumerate(agents):
        ax = axes[agent_idx]
        ax.set_facecolor(config['aesthetics']['axes']['facecolor'])
        ax.grid(True, alpha=config['aesthetics']['axes']['grid_alpha'], linestyle='--')
        ax.set_axisbelow(True)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(config['aesthetics']['axes']['spine_width'])
            spine.set_color(config['aesthetics']['axes']['spine_color'])
        
        agent_results = {k: v for k, v in results.items() 
                        if extract_info_from_key(k)[0] == agent}
        
        low_success_difficulties = [[] for _ in range(vector_length)]
        high_success_difficulties = [[] for _ in range(vector_length)]
        
        for key, data in agent_results.items():
            success_rate = data['success_rate'] * 100
            _, _, shot = extract_info_from_key(key)
            shot_key = f'shot_{shot}'
            
            if shot_key in tasks and 'difficulties' in tasks[shot_key]:
                difficulties = tasks[shot_key]['difficulties']
                for i, diff in enumerate(difficulties):
                    if success_rate <= 25:
                        low_success_difficulties[i].append(diff)
                    elif success_rate >= 75:
                        high_success_difficulties[i].append(diff)
        
        avg_low, err_low = calculate_stats(low_success_difficulties)
        avg_high, err_high = calculate_stats(high_success_difficulties)
        
        deltas_high_low = avg_high - avg_low
        deltas_low_high = avg_low - avg_high
        errors = np.sqrt(err_low**2 + err_high**2)
        MAX_VAL = max(np.max(deltas_high_low), MAX_VAL)
        MAX_VAL = max(np.max(deltas_low_high), MAX_VAL)

        # Get indices where deltas exceed threshold
        indices_high_low = np.where(deltas_high_low > 0.05)[0]
        indices_low_high = np.where(deltas_low_high > 0.05)[0]
        
        # Get top K indices among those exceeding threshold
        top_indices_high_low = indices_high_low[np.argsort(deltas_high_low[indices_high_low])[-min(PLOT_K, len(indices_high_low)):]]
        top_indices_low_high = indices_low_high[np.argsort(deltas_low_high[indices_low_high])[-min(PLOT_K, len(indices_low_high)):]]
        
        # Create labels for all 2*K rules
        rule_labels = ([f"Rule {i+1} (H-L)" for i in top_indices_high_low] + 
                      [f"Rule {i+1} (L-H)" for i in top_indices_low_high])
        
        # Sort indices by delta values
        top_indices_high_low = top_indices_high_low[np.argsort(deltas_high_low[top_indices_high_low])]
        top_indices_low_high = top_indices_low_high[np.argsort(deltas_low_high[top_indices_low_high])]
        
        n_high_low = len(top_indices_high_low)
        n_low_high = len(top_indices_low_high)
        
        # Plot high-low deltas (blue)
        if n_high_low > 0:
            bars_high_low = ax.barh(np.arange(n_high_low) + n_low_high, deltas_high_low[top_indices_high_low],
                                  height=0.6, color=config['colors'][0], alpha=0.8)
            ax.errorbar(deltas_high_low[top_indices_high_low], np.arange(n_high_low) + n_low_high,
                      xerr=errors[top_indices_high_low],
                      fmt='none', color='black', capsize=5)
        
        # Plot low-high deltas (red)
        if n_low_high > 0:
            bars_low_high = ax.barh(np.arange(n_low_high), deltas_low_high[top_indices_low_high],
                                  height=0.6, color=config['colors'][1], alpha=0.8)
            ax.errorbar(deltas_low_high[top_indices_low_high], np.arange(n_low_high),
                      xerr=errors[top_indices_low_high],
                      fmt='none', color='black', capsize=5)
        
        ax.set_title(f'{agent}',
                    fontsize=config['fonts']['title']['size'],
                    fontweight=config['fonts']['title']['weight'],
                    pad=config['labels']['title_padding'])
        
        ax.set_xlabel('Difficulty Delta',
                     fontsize=config['fonts']['bold']['size'],
                     fontweight=config['fonts']['bold']['weight'],
                     labelpad=config['labels']['padding'])
        
        # Set yticks only for rules that exceed threshold
        rule_labels = ([f"Rule {i+1} (L-H)" for i in top_indices_low_high] + 
                      [f"Rule {i+1} (H-L)" for i in top_indices_high_low])
        ax.set_yticks(np.arange(n_low_high + n_high_low))
        ax.set_yticklabels(rule_labels, fontsize=config['fonts']['regular']['size'])
        
        ax.axvline(x=0, color=config['aesthetics']['axes']['spine_color'],
                  linestyle='-', linewidth=0.5)
            
    for agent_idx, agent in enumerate(agents):
        ax = axes[agent_idx]
        ax.set_xlim(0, MAX_VAL*1.1)
    
    plt.tight_layout()
    plt.savefig(f'{VISUALISATIONS_DIR}/demo_skill/difficulty_deltas.png',
                dpi=config['aesthetics']['figure']['dpi'],
                bbox_inches='tight',
                facecolor=config['aesthetics']['figure']['facecolor'])
    print(f"Saved '{VISUALISATIONS_DIR}/demo_skill/difficulty_deltas.png'")
    plt.close()

if __name__ == "__main__":
    plot_difficulty_deltas_both()