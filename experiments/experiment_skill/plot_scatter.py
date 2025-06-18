import json, argparse
import matplotlib.pyplot as plt
import re
import numpy as np
from pathlib import Path
import yaml
import sys
import os

from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from poolagent.path import DATA_DIR, VISUALISATIONS_DIR

def load_plot_config(config_path='../plot_config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Set matplotlib rcParams
    for key, value in config['rcParams'].items():
        plt.rcParams[key] = value
    
    # Set style
    plt.style.use(config['style'])
    
    return config

# Load configuration
config = load_plot_config()

def extract_info_from_key(key):
    shot_match = re.search(r'shot_(\d+)', key)
    shot = int(shot_match.group(1)) if shot_match else None
    parts = key.split('-')
    agent = parts[0]
    model = '-'.join(parts[1:-1])
    return agent, model, shot

def load_value_estimates():
    with open(f'{DATA_DIR}/shot_task_dataset.json', 'r') as f:
        data = json.load(f)
    estimates = {}
    for key, value in data.items():
        shot = int(key.split('_')[1])
        estimates[shot] = value['value_estimate']
    return estimates

def get_shots_with_all_agent_data(results, model_name):
    agent_data = {}
    
    for key, data in results.items():
        if model_name not in key:
            continue
        
        agent, _, shot = extract_info_from_key(key)
        if shot is not None:
            if agent not in agent_data:
                agent_data[agent] = {}
            
            attempts = [x[0] for x in data['attempts']]
            if attempts:
                agent_data[agent][shot] = attempts

    if not agent_data:
        return set()

    all_shots = set().union(*[set(agent_shots.keys()) for agent_shots in agent_data.values()])
    shots_with_all_data = set()
    
    for shot in all_shots:
        if all(shot in agent_shots for agent_shots in agent_data.values()):
            shots_with_all_data.add(shot)
    
    return shots_with_all_data

def plot_results(results_file, model_name):
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    value_estimates = load_value_estimates()
    data_by_combination = {}
    
    valid_shots = get_shots_with_all_agent_data(results, model_name)
    if not valid_shots:
        print("No shots found with data from all agents")
        return
    
    print(f"\nShots with data from all agents: {sorted(list(valid_shots))}")
    
    # Data collection
    for key, data in results.items():
        if model_name not in key:
            continue

        agent, model, shot = extract_info_from_key(key)
        if shot is not None and shot in valid_shots:
            combination = f"{agent}-{model}"
            if combination not in data_by_combination:
                data_by_combination[combination] = {'shots': [], 'deltas': [], 'std_errs': [], 'success_rate': []}
            
            attempts = [x[0] for x in data['attempts']]
            if attempts:
                mean_val = np.mean(attempts)
                mean_success = np.mean([x[1] for x in data['attempts']])
                std_err = stats.sem(attempts)
                
                if shot in value_estimates:
                    delta = mean_val - value_estimates[shot]
                    data_by_combination[combination]['shots'].append(shot)
                    data_by_combination[combination]['deltas'].append(delta)
                    data_by_combination[combination]['std_errs'].append(std_err)
                    data_by_combination[combination]['success_rate'].append(mean_success)

    # Print statistics
    print("\nAccumulated Deltas by Combination:")
    for combination, data in data_by_combination.items():
        shots = np.array(data['shots'])
        deltas = np.array(data['deltas'])
        
        sort_idx = np.argsort(shots)
        shots = shots[sort_idx]
        deltas = deltas[sort_idx]
        
        accumulated_delta = np.sum(deltas)
        mean_delta = np.mean(deltas)
        mean_success = np.mean(data['success_rate'])
        
        print(f"\n{combination}:")
        print(f"  Mean Success Rate: {mean_success:.2f}")
        print(f"  Accumulated Delta: {accumulated_delta:.4f}")
        print(f"  Mean Delta: {mean_delta:.4f}")
        print(f"  Delta by shot:")
        for shot, delta in zip(shots, deltas):
            print(f"    Shot {shot}: {delta:.4f}")

    # Create plot
    fig, ax = plt.subplots(figsize=config['figure_sizes']['default'])
    fig.set_facecolor(config['aesthetics']['figure']['facecolor'])
    ax.set_facecolor(config['aesthetics']['axes']['facecolor'])
    
    for i, (combination, data) in enumerate(data_by_combination.items()):
        shots = np.array(data['shots'])
        deltas = np.array(data['deltas'])
        
        sort_idx = np.argsort(shots)
        shots = shots[sort_idx]
        deltas = deltas[sort_idx]

        agent_name = config['agent_names'][combination.split('_')[0]]
        color = config['colors'][i % len(config['colors'])]
        marker = config['markers']['scatter'][i]
        
        label = f"{agent_name} (Σ={np.sum(deltas):.2f})"
        
        # Plot connecting line with higher alpha
        ax.plot(shots, deltas, color=color, alpha=0.8,
                linestyle='-', linewidth=2.5, zorder=2)
        
        # Plot scatter points
        ax.scatter(shots, deltas, label=label, marker=marker,
                  color=color, s=150, zorder=3,
                  edgecolor='white', linewidth=1)

    # Add zero line
    ax.axhline(y=0, color='#666666', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
    
    # Grid and axis appearance
    ax.grid(True, alpha=config['aesthetics']['axes']['grid_alpha'], linestyle='--')
    ax.set_axisbelow(True)

    # Labels and title
    ax.set_xlabel('Shot Number', fontsize=config['fonts']['bold']['size'],
                 fontweight=config['fonts']['bold']['weight'],
                 labelpad=config['labels']['padding'])
    ax.set_ylabel('Delta from Value Estimate',
                 fontsize=config['fonts']['bold']['size'],
                 fontweight=config['fonts']['bold']['weight'],
                 labelpad=config['labels']['padding'])
    ax.set_title(f'Performance Relative to Value Estimate by Shot Number\n{model_name}',
                 fontsize=config['fonts']['title']['size'],
                 fontweight=config['fonts']['title']['weight'],
                 pad=config['labels']['title_padding'])

    # Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(config['aesthetics']['axes']['spine_width'])
        spine.set_color(config['aesthetics']['axes']['spine_color'])

    # Legend
    legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                      frameon=True,
                      fancybox=config['aesthetics']['legend']['fancybox'],
                      shadow=config['aesthetics']['legend']['shadow'])
    legend.get_frame().set_facecolor(config['aesthetics']['legend']['facecolor'])
    legend.get_frame().set_alpha(config['aesthetics']['legend']['alpha'])

    # Integer ticks for shot numbers
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    output_path = f'{VISUALISATIONS_DIR}/experiment_skill/scatter_delta.png'
    plt.savefig(output_path, bbox_inches='tight',
                dpi=config['aesthetics']['figure']['dpi'],
                facecolor=config['aesthetics']['figure']['facecolor'])
    print(f"\nPlot saved as {output_path}")
    plt.close()

def main(args):
    results_dir = Path('results')
    latest_file = max(list(results_dir.glob('*.json')), key=lambda x: x.stat().st_mtime)
    print(f"Processing {latest_file}")
    
    plot_results(latest_file, args.model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot scatter of deltas from value estimate by shot number')
    parser.add_argument('--model', type=str, default='Meta-Llama-3.1-70B-Instruct', help='Model name')
    args = parser.parse_args()

    main(args)