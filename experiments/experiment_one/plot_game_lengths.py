import os
from collections import defaultdict
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import yaml
import sys
import os

from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from poolagent import VISUALISATIONS_DIR, ROOT_DIR

def load_plot_config(config_path='../plot_config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    for key, value in config['rcParams'].items():
        plt.rcParams[key] = value
    plt.style.use(config['style'])
    return config

config = load_plot_config()

def load_most_recent_file(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    if not files:
        raise FileNotFoundError("No JSON files found in the directory")
    return os.path.join(directory, max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f))))

def extract_game_length_stats(data):
    # Extract function remains unchanged
    agent_types = defaultdict(lambda: {
        'player1_wins': [],
        'player1_losses': [],
        'player2_wins': [],
        'player2_losses': []
    })
    
    for key, value in data.items():
        agent1, agent2 = key.split('---')
        agent1_base = agent1.split('_')[0]
        agent2_base = agent2.split('_')[0]
        
        for length, win in zip(value['game_lengths'], value['games']):
            if win == 1:
                agent_types[agent1_base]['player1_wins'].append(length)
                agent_types[agent2_base]['player2_losses'].append(length)
            else:
                agent_types[agent1_base]['player1_losses'].append(length)
                agent_types[agent2_base]['player2_wins'].append(length)

    results = {}
    for agent_type, data in agent_types.items():
        if any(data.values()):
            results[agent_type] = {
                'all': {
                    'player1_mean': np.mean(data['player1_wins'] + data['player1_losses']),
                    'player1_stderr': stats.sem(data['player1_wins'] + data['player1_losses']) / np.sqrt(len(data['player1_wins'] + data['player1_losses'])),
                    'player1_games': len(data['player1_wins'] + data['player1_losses']),
                    'player2_mean': np.mean(data['player2_wins'] + data['player2_losses']),
                    'player2_stderr': stats.sem(data['player2_wins'] + data['player2_losses']) / np.sqrt(len(data['player2_wins'] + data['player2_losses'])),
                    'player2_games': len(data['player2_wins'] + data['player2_losses'])
                },
                'wins': {
                    'player1_mean': np.mean(data['player1_wins']) if data['player1_wins'] else np.nan,
                    'player1_stderr': stats.sem(data['player1_wins']) / np.sqrt(len(data['player1_wins'])) if data['player1_wins'] else np.nan,
                    'player1_games': len(data['player1_wins']),
                    'player2_mean': np.mean(data['player2_wins']) if data['player2_wins'] else np.nan,
                    'player2_stderr': stats.sem(data['player2_wins']) / np.sqrt(len(data['player2_wins'])) if data['player2_wins'] else np.nan,
                    'player2_games': len(data['player2_wins'])
                },
                'losses': {
                    'player1_mean': np.mean(data['player1_losses']) if data['player1_losses'] else np.nan,
                    'player1_stderr': stats.sem(data['player1_losses']) / np.sqrt(len(data['player1_losses'])) if data['player1_losses'] else np.nan,
                    'player1_games': len(data['player1_losses']),
                    'player2_mean': np.mean(data['player2_losses']) if data['player2_losses'] else np.nan,
                    'player2_stderr': stats.sem(data['player2_losses']) / np.sqrt(len(data['player2_losses'])) if data['player2_losses'] else np.nan,
                    'player2_games': len(data['player2_losses'])
                }
            }
        else:
            results[agent_type] = None
    
    return results

def create_game_length_plot(model_name, stats, output_dir, single_plot=True):
    if not single_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=config['figure_sizes']['tall'])
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=config['figure_sizes']['default'])
    fig.set_facecolor(config['aesthetics']['figure']['facecolor'])
    
    agent_order = ['ProRandomBallAgent', 'LanguageAgent', 'LanguageDEFAgent', 'LanguageFunctionAgent', 'FunctionAgent', 'PoolMasterAgent']
    x = np.arange(len(agent_order))
    width = 0.25  # Width of bars
    
    plots_data = {
        'all': {'ax': ax1, 'title': 'All Games', 'data': 'all'},
        'wins': {'ax': ax2, 'title': 'Winning Games Only', 'data': 'wins'},
        'losses': {'ax': ax3, 'title': 'Losing Games Only', 'data': 'losses'}
    } if not single_plot else {
        'all': {'ax': ax1, 'title': 'All Games', 'data': 'all'}
    }
    
    for plot_type, plot_info in plots_data.items():
        ax = plot_info['ax']
        data_key = plot_info['data']
        ax.set_facecolor(config['aesthetics']['axes']['facecolor'])
        
        p1_means = []
        p1_errs = []
        p2_means = []
        p2_errs = []
        p1_ns = []
        p2_ns = []
        
        for agent in agent_order:
            if stats[agent] is not None:
                data = stats[agent][data_key]
                p1_means.append(data['player1_mean'])
                p1_errs.append(data['player1_stderr'])
                p2_means.append(data['player2_mean'])
                p2_errs.append(data['player2_stderr'])
                p1_ns.append(data['player1_games'])
                p2_ns.append(data['player2_games'])
            else:
                p1_means.append(np.nan)
                p1_errs.append(np.nan)
                p2_means.append(np.nan)
                p2_errs.append(np.nan)
                p1_ns.append(0)
                p2_ns.append(0)

        # Calculate combined stats
        combined_means = []
        combined_errs = []
        for p1_mean, p2_mean, p1_err, p2_err, p1_n, p2_n in zip(
            p1_means, p2_means, p1_errs, p2_errs, p1_ns, p2_ns):
            if np.isnan(p1_mean) or np.isnan(p2_mean):
                combined_means.append(np.nan)
                combined_errs.append(np.nan)
            else:
                total_n = p1_n + p2_n
                combined_means.append((p1_mean * p1_n + p2_mean * p2_n) / total_n)
                combined_errs.append(np.sqrt((p1_err**2 * p1_n**2 + p2_err**2 * p2_n**2) / total_n**2))

        # Plot bars
        ax.bar(x - width, p1_means, width, yerr=p1_errs,
               color=config['colors'][0], label='As Player 1')
        ax.bar(x, combined_means, width, yerr=combined_errs,
               color=config['colors'][1], label='Overall')
        ax.bar(x + width, p2_means, width, yerr=p2_errs,
               color=config['colors'][2], label='As Player 2')

        # Grid and formatting
        ax.grid(True, alpha=config['aesthetics']['axes']['grid_alpha'], linestyle='--')
        ax.set_axisbelow(True)

        ax.set_xlabel('Agent Type', 
                     fontsize=config['fonts']['bold']['size'],
                     fontweight=config['fonts']['bold']['weight'],
                     labelpad=config['labels']['padding'])
        ax.set_ylabel('Game Length (moves)',
                     fontsize=config['fonts']['bold']['size'],
                     fontweight=config['fonts']['bold']['weight'],
                     labelpad=config['labels']['padding'])

        ax.set_xticks(x)
        ax.set_xticklabels([
            config['agent_names'][agent] if agent in config['agent_names'] else agent
            for agent in agent_order
        ], rotation=45, ha='right', fontsize=config['fonts']['tick']['size'])
        
        ax.tick_params(axis='y', labelsize=config['fonts']['tick']['size'])

        legend = ax.legend(loc='lower left',
                         frameon=True,
                         fancybox=config['aesthetics']['legend']['fancybox'],
                         shadow=config['aesthetics']['legend']['shadow'],
                         fontsize=config['fonts']['tick']['size'])
        legend.get_frame().set_facecolor(config['aesthetics']['legend']['facecolor'])
        legend.get_frame().set_alpha(config['aesthetics']['legend']['alpha'])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(config['aesthetics']['axes']['spine_width'])
            spine.set_color(config['aesthetics']['axes']['spine_color'])
    
    plt.ylim(6,13)

    plt.tight_layout(h_pad=3.0)
    output_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}_game_lengths.pdf")
    plt.savefig(output_path,
                dpi=config['aesthetics']['figure']['dpi'],
                bbox_inches='tight',
                facecolor=config['aesthetics']['figure']['facecolor'])
    plt.close()
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate game length plots for a specific model')
    parser.add_argument('--model_name', type=str, default='Meta-Llama-3.1-70B-Instruct-Turbo')
    parser.add_argument('--single', action='store_true', help='Create a single plot with all data', default=True)
    
    args = parser.parse_args()
    
    results_dir = f"{ROOT_DIR}/experiments/experiment_one/results"
    output_dir = f"{VISUALISATIONS_DIR}/experiment_one/single_model/"
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(load_most_recent_file(results_dir), 'r') as f:
        data = json.load(f)
    
    stats = extract_game_length_stats(data)
    output_path = create_game_length_plot(args.model_name, stats, output_dir, single_plot=args.single)
    
    print(f"Game length plots saved as: {output_path}")

if __name__ == "__main__":
    main()