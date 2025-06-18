import os
import json
from glob import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from poolagent import VISUALISATIONS_DIR, ROOT_DIR

def parse_agents_from_task(task_name):
    agents = task_name.split('---')
    return agents[0].split('_')[0], agents[1].split('_')[0]

def load_plot_config(config_path='../plot_config.yaml'):
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    for key, value in config['rcParams'].items():
        plt.rcParams[key] = value
    plt.style.use(config['style'])
    return config

def analyze_and_plot_games():
    config = load_plot_config()
    stats = defaultdict(lambda: {'as_p1': [0, 0], 'as_p2': [0, 0]})
    
    # Data collection remains unchanged
    for game_file in glob('logs/*/tasks/*/game_*.json'):
        path_parts = Path(game_file).parts
        task_name = path_parts[path_parts.index('tasks') + 1]
        agent1, agent2 = parse_agents_from_task(task_name)
        
        try:
            with open(game_file, 'r') as f:
                game_data = json.load(f)
                
            game_stats = defaultdict(lambda: {"pockets": 0, "shots": 0})
            for shot_data in game_data.values():
                if 'events' not in shot_data:
                    continue
                    
                player = shot_data["player"]
                game_stats[player]["shots"] += 1
                if any("pocket" in event["encoding"] and "cue" not in event["encoding"] 
                      for event in shot_data["events"]):
                    game_stats[player]["pockets"] += 1
            
            stats[agent1]['as_p1'][0] += game_stats['one']['pockets']
            stats[agent1]['as_p1'][1] += game_stats['one']['shots']
            stats[agent2]['as_p2'][0] += game_stats['two']['pockets']
            stats[agent2]['as_p2'][1] += game_stats['two']['shots']
                
        except Exception as e:
            print(f"Error processing {game_file}: {e}")

    agent_order = ['ProRandomBallAgent', 'LanguageAgent', 'LanguageDEFAgent', 'LanguageFunctionAgent', 'FunctionAgent', 'PoolMasterAgent']

    agent_types = [agent for agent in agent_order if agent in stats]
    x = np.arange(len(agent_types))
    width = 0.25  # Width of bars
    
    p1_success_rates = []
    p2_success_rates = []
    p1_errors = []
    p2_errors = []
    combined_rates = []
    combined_errors = []
    
    for agent in agent_types:
        # Calculate rates and errors
        p1_pockets, p1_shots = stats[agent]['as_p1']
        p2_pockets, p2_shots = stats[agent]['as_p2']
        
        # Player 1
        p1_rate = p1_pockets / p1_shots if p1_shots > 0 else 0
        p1_error = np.sqrt(p1_rate * (1 - p1_rate) / p1_shots) if p1_shots > 0 else 0
        p1_success_rates.append(p1_rate * 100)
        p1_errors.append(p1_error * 100)
        
        # Player 2
        p2_rate = p2_pockets / p2_shots if p2_shots > 0 else 0
        p2_error = np.sqrt(p2_rate * (1 - p2_rate) / p2_shots) if p2_shots > 0 else 0
        p2_success_rates.append(p2_rate * 100)
        p2_errors.append(p2_error * 100)

        # Combined
        total_pockets = p1_pockets + p2_pockets
        total_shots = p1_shots + p2_shots
        combined_rate = (total_pockets / total_shots if total_shots > 0 else 0) * 100
        combined_error = np.sqrt(combined_rate/100 * (1 - combined_rate/100) / total_shots) * 100 if total_shots > 0 else 0
        combined_rates.append(combined_rate)
        combined_errors.append(combined_error)

    # Create plot
    fig, ax = plt.subplots(figsize=config['figure_sizes']['default'])
    fig.set_facecolor(config['aesthetics']['figure']['facecolor'])
    ax.set_facecolor(config['aesthetics']['axes']['facecolor'])
    
    # Plot bars instead of lines
    ax.bar(x - width, p1_success_rates, width, yerr=p1_errors, 
           color=config['colors'][0], label='As Player 1')
    ax.bar(x, combined_rates, width, yerr=combined_errors,
           color=config['colors'][1], label='Overall')
    ax.bar(x + width, p2_success_rates, width, yerr=p2_errors,
           color=config['colors'][2], label='As Player 2')

    # Plot formatting
    ax.grid(True, alpha=config['aesthetics']['axes']['grid_alpha'], linestyle='--')
    ax.set_axisbelow(True)
    
    ax.set_xlabel('Agent Type', fontsize=config['fonts']['bold']['size'],
                 fontweight=config['fonts']['bold']['weight'],
                 labelpad=config['labels']['padding'])
    ax.set_ylabel('Pot Rate (%)', fontsize=config['fonts']['bold']['size'],
                 fontweight=config['fonts']['bold']['weight'],
                 labelpad=config['labels']['padding'])
    
    ax.set_xticks(x)
    ax.set_xticklabels([config['agent_names'].get(agent, agent) for agent in agent_types],
                       rotation=45, ha='right', fontsize=config['fonts']['tick']['size'])
    ax.tick_params(axis='y', labelsize=config['fonts']['tick']['size'])

    legend = ax.legend(loc='upper left', frameon=True,
                      fancybox=config['aesthetics']['legend']['fancybox'],
                      shadow=config['aesthetics']['legend']['shadow'],
                      fontsize=config['fonts']['tick']['size'])
    legend.get_frame().set_facecolor(config['aesthetics']['legend']['facecolor'])
    legend.get_frame().set_alpha(config['aesthetics']['legend']['alpha'])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(config['aesthetics']['axes']['spine_width'])
        spine.set_color(config['aesthetics']['axes']['spine_color'])

    plt.ylim(20, 100)

    plt.tight_layout()
    
    output_dir = f"{VISUALISATIONS_DIR}/experiment_one/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "shot_success_rates.pdf")
    plt.savefig(output_path, dpi=config['aesthetics']['figure']['dpi'],
                bbox_inches='tight',
                facecolor=config['aesthetics']['figure']['facecolor'])
    plt.close()
    
    return output_path

if __name__ == "__main__":
    output_path = analyze_and_plot_games()
    print(f"Shot success rate plot saved as: {output_path}")