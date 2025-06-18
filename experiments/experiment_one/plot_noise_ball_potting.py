import os
import json
from glob import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import yaml
import sys
from typing import Dict, List, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from poolagent import VISUALISATIONS_DIR, ROOT_DIR

def parse_agents_from_task(task_name: str) -> Tuple[str, str]:
    agents = task_name.split('---')
    return agents[0].split('_')[0], agents[1].split('_')[0]

def load_plot_config(config_path='../plot_config.yaml') -> dict:
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    for key, value in config['rcParams'].items():
        plt.rcParams[key] = value
    plt.style.use(config['style'])
    return config

def calculate_statistics(game_stats: Dict) -> Tuple[float, float]:
    total_pockets = game_stats['pockets']
    total_shots = game_stats['shots']
    if total_shots == 0:
        return 0, 0
    success_rate = (total_pockets / total_shots) * 100
    std_error = np.sqrt((success_rate/100 * (1 - success_rate/100)) / total_shots) * 100
    return success_rate, std_error

def analyze_games_by_noise():
    config = load_plot_config()
    stats = defaultdict(lambda: defaultdict(lambda: {
        'pockets': 0, 'shots': 0
    }))
    
    for game_file in glob('noise_logs/*/*/tasks/*/game_*.json'):
        path_parts = Path(game_file).parts
        noise_level = path_parts[path_parts.index('tasks') - 1]
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
            
            stats[noise_level][agent1]['pockets'] += game_stats['one']['pockets']
            stats[noise_level][agent1]['shots'] += game_stats['one']['shots']
            stats[noise_level][agent2]['pockets'] += game_stats['two']['pockets']
            stats[noise_level][agent2]['shots'] += game_stats['two']['shots']
                
        except Exception as e:
            print(f"Error processing {game_file}: {e}")

    return stats, config

def plot_noise_analysis(stats: Dict, config: Dict) -> str:
    agent_order = ['LanguageAgent', 'LanguageDEFAgent', 'LanguageFunctionAgent']
    noise_map = {
        'no_noise': 'None',
        'amateur': 'Low',
        'novice': 'Medium',
        'pro': 'High'
    }
    noise_order = ['no_noise', 'amateur', 'novice', 'pro']
    noise_levels = [level for level in noise_order if level in stats]
    agent_types = [agent for agent in agent_order if any(agent in stats[noise] 
                                                       for noise in noise_levels)]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.set_facecolor(config['aesthetics']['figure']['facecolor'])
    ax.set_facecolor(config['aesthetics']['axes']['facecolor'])
    
    x = np.arange(len(noise_levels))
    width = 0.25  # Width of bars
    
    # Plot bars for each agent
    for idx, agent in enumerate(agent_types):
        rates, errors = [], []
        for noise in noise_levels:
            if agent in stats[noise]:
                rate, error = calculate_statistics(stats[noise][agent])
                rates.append(rate)
                errors.append(error)
        
        position = x + (idx - len(agent_types)/2 + 0.5) * width
        ax.bar(position, rates, width, 
               yerr=errors,
               color=config['colors'][idx],
               label=config['agent_names'].get(agent, agent), zorder=3)

    # Customize plot
    ax.grid(True, alpha=config['aesthetics']['axes']['grid_alpha'], linestyle='--')
    ax.set_axisbelow(True)
    
    ax.set_ylabel('Pot Rate (%)', fontsize=config['fonts']['bold']['size'],
                 fontweight=config['fonts']['bold']['weight'],
                 labelpad=config['labels']['padding'])
    
    ax.set_xlabel('Noise Level', fontsize=config['fonts']['bold']['size'],
                 fontweight=config['fonts']['bold']['weight'],
                 labelpad=config['labels']['padding'])
    
    ax.tick_params(axis='both', labelsize=config['fonts']['tick']['size'])
    
    legend = ax.legend(loc='upper right', frameon=True,
                      fancybox=config['aesthetics']['legend']['fancybox'],
                      shadow=config['aesthetics']['legend']['shadow'],
                      fontsize=config['fonts']['tick']['size'])
    legend.get_frame().set_facecolor(config['aesthetics']['legend']['facecolor'])
    legend.get_frame().set_alpha(config['aesthetics']['legend']['alpha'])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(config['aesthetics']['axes']['spine_width'])
        spine.set_color(config['aesthetics']['axes']['spine_color'])

    plt.xticks(x, [noise_map[level] for level in noise_levels])

    plt.tight_layout()
    
    output_dir = f"{VISUALISATIONS_DIR}/experiment_one/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "noise_level_analysis.pdf")
    plt.savefig(output_path, dpi=config['aesthetics']['figure']['dpi'],
                bbox_inches='tight',
                facecolor=config['aesthetics']['figure']['facecolor'])
    plt.close()
    
    return output_path

def analyze_and_plot_noise_levels():
    stats, config = analyze_games_by_noise()
    output_path = plot_noise_analysis(stats, config)
    return output_path

if __name__ == "__main__":
    output_path = analyze_and_plot_noise_levels()
    print(f"Noise level analysis plot saved as: {output_path}")