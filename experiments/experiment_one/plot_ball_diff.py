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
    
    # Set matplotlib rcParams
    for key, value in config['rcParams'].items():
        plt.rcParams[key] = value
    
    # Set style
    plt.style.use(config['style'])
    
    return config

# Load configuration
config = load_plot_config()

def load_most_recent_file(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    if not files:
        raise FileNotFoundError("No JSON files found in the directory")
    return os.path.join(directory, max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f))))

def extract_differential_stats(data, model_name):
    agent_types = defaultdict(lambda: {
        'player1_diffs': [],
        'player2_diffs': [],
        'player1_wins': [],
        'player2_wins': [],
        'player1_losses': [],
        'player2_losses': []
    })
    
    for key, value in data.items():
        agent1, agent2 = key.split('---')
        agent1_base = agent1.split('_')[0]
        agent2_base = agent2.split('_')[0]
        
        for i, diff in enumerate(value['ball_differentials']):
            is_win = diff > 0
            # Player 1 stats
            agent_types[agent1_base]['player1_diffs'].append(diff)
            if is_win:
                agent_types[agent1_base]['player1_wins'].append(diff)
            else:
                agent_types[agent1_base]['player1_losses'].append(diff)
            
            # Player 2 stats (negated)
            agent_types[agent2_base]['player2_diffs'].append(-diff)
            if not is_win:  # Player 2 wins when Player 1 loses
                agent_types[agent2_base]['player2_wins'].append(-diff)
            else:
                agent_types[agent2_base]['player2_losses'].append(-diff)

    results = {}
    for agent_type, data in agent_types.items():
        if any(data.values()):
            results[agent_type] = {
                'player1_mean': np.mean(data['player1_diffs']) if data['player1_diffs'] else np.nan,
                'player1_stderr': stats.sem(data['player1_diffs']) / np.sqrt(len(data['player1_diffs'])) if data['player1_diffs'] else np.nan,
                'player1_games': len(data['player1_diffs']),
                'player2_mean': np.mean(data['player2_diffs']) if data['player2_diffs'] else np.nan,
                'player2_stderr': stats.sem(data['player2_diffs']) / np.sqrt(len(data['player2_diffs'])) if data['player2_diffs'] else np.nan,
                'player2_games': len(data['player2_diffs']),
                'wins': {
                    'player1_mean_wins': np.mean(data['player1_wins']) if data['player1_wins'] else np.nan,
                    'player1_stderr_wins': stats.sem(data['player1_wins']) / np.sqrt(len(data['player1_wins'])) if data['player1_wins'] else np.nan,
                    'player1_games_wins': len(data['player1_wins']),
                    'player2_mean_wins': np.mean(data['player2_wins']) if data['player2_wins'] else np.nan,
                    'player2_stderr_wins': stats.sem(data['player2_wins']) / np.sqrt(len(data['player2_wins'])) if data['player2_wins'] else np.nan,
                    'player2_games_wins': len(data['player2_wins'])
                },
                'losses': {
                    'player1_mean_losses': np.mean(data['player1_losses']) if data['player1_losses'] else np.nan,
                    'player1_stderr_losses': stats.sem(data['player1_losses']) / np.sqrt(len(data['player1_losses'])) if data['player1_losses'] else np.nan,
                    'player1_games_losses': len(data['player1_losses']),
                    'player2_mean_losses': np.mean(data['player2_losses']) if data['player2_losses'] else np.nan,
                    'player2_stderr_losses': stats.sem(data['player2_losses']) / np.sqrt(len(data['player2_losses'])) if data['player2_losses'] else np.nan,
                    'player2_games_losses': len(data['player2_losses'])
                }
            }
        else:
            results[agent_type] = None
    
    return results

def create_differential_plot(model_name, stats, output_dir, single_plot=True):
    if not single_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=config['figure_sizes']['tall'])
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=config['figure_sizes']['default'])
    fig.set_facecolor(config['aesthetics']['figure']['facecolor'])
    
    agent_order = ['ProRandomBallAgent', 'LanguageAgent', 'LanguageDEFAgent', 'LanguageFunctionAgent', 'FunctionAgent', 'PoolMasterAgent']
    x = np.arange(len(agent_order))
    width = 0.25  # Width of the bars

    plots_data = {
        'all': {'title': 'All Games'},
        'wins': {'title': 'Winning Games Only'},
        'losses': {'title': 'Losing Games Only'}
    }
    
    for game_type in plots_data:
        plots_data[game_type].update({
            'p1_means': [], 'p1_errs': [],
            'p2_means': [], 'p2_errs': [],
            'p1_ns': [], 'p2_ns': []
        })
        
        for agent in agent_order:
            if stats[agent] is not None:
                if game_type == 'all':
                    plots_data[game_type]['p1_means'].append(stats[agent]['player1_mean'])
                    plots_data[game_type]['p1_errs'].append(stats[agent]['player1_stderr'])
                    plots_data[game_type]['p2_means'].append(stats[agent]['player2_mean'])
                    plots_data[game_type]['p2_errs'].append(stats[agent]['player2_stderr'])
                    plots_data[game_type]['p1_ns'].append(stats[agent]['player1_games'])
                    plots_data[game_type]['p2_ns'].append(stats[agent]['player2_games'])
                elif game_type == 'wins':
                    data = stats[agent]['wins']
                    plots_data[game_type]['p1_means'].append(data['player1_mean_wins'])
                    plots_data[game_type]['p1_errs'].append(data['player1_stderr_wins'])
                    plots_data[game_type]['p2_means'].append(data['player2_mean_wins'])
                    plots_data[game_type]['p2_errs'].append(data['player2_stderr_wins'])
                    plots_data[game_type]['p1_ns'].append(data['player1_games_wins'])
                    plots_data[game_type]['p2_ns'].append(data['player2_games_wins'])
                else:  # losses
                    data = stats[agent]['losses']
                    plots_data[game_type]['p1_means'].append(data['player1_mean_losses'])
                    plots_data[game_type]['p1_errs'].append(data['player1_stderr_losses'])
                    plots_data[game_type]['p2_means'].append(data['player2_mean_losses'])
                    plots_data[game_type]['p2_errs'].append(data['player2_stderr_losses'])
                    plots_data[game_type]['p1_ns'].append(data['player1_games_losses'])
                    plots_data[game_type]['p2_ns'].append(data['player2_games_losses'])
            else:
                for key in ['p1_means', 'p1_errs', 'p2_means', 'p2_errs', 'p1_ns', 'p2_ns']:
                    plots_data[game_type][key].append(np.nan)
    
    axes = {'all': ax1, 'wins': ax2, 'losses': ax3} if not single_plot else {'all': ax1}
    
    for game_type, ax in axes.items():
        data = plots_data[game_type]
        ax.set_facecolor(config['aesthetics']['axes']['facecolor'])
        
        # Calculate combined means and errors
        combined_means = []
        combined_errs = []
        for p1_mean, p2_mean, p1_err, p2_err, p1_n, p2_n in zip(
            data['p1_means'], data['p2_means'], 
            data['p1_errs'], data['p2_errs'],
            data['p1_ns'], data['p2_ns']):
            if np.isnan(p1_mean) or np.isnan(p2_mean):
                combined_means.append(np.nan)
                combined_errs.append(np.nan)
            else:
                combined_mean = (p1_mean * p1_n + p2_mean * p2_n) / (p1_n + p2_n)
                combined_err = np.sqrt((p1_err**2 * p1_n**2 + p2_err**2 * p2_n**2) / (p1_n + p2_n)**2)
                combined_means.append(combined_mean)
                combined_errs.append(combined_err)

        # Convert to numpy arrays
        p1_means = np.array(data['p1_means'])
        p2_means = np.array(data['p2_means'])
        combined_means = np.array(combined_means)
        p1_errs = np.array(data['p1_errs'])
        p2_errs = np.array(data['p2_errs'])
        combined_errs = np.array(combined_errs)

        # Plot bars
        ax.bar(x - width, p1_means, width, label='As Player 1', color=config['colors'][0],
               yerr=p1_errs, zorder=3)
        ax.bar(x, combined_means, width, label='Overall', color=config['colors'][1],
               yerr=combined_errs, zorder=3)
        ax.bar(x + width, p2_means, width, label='As Player 2', color=config['colors'][2],
               yerr=p2_errs, zorder=3)

        # Grid and axis appearance
        ax.grid(True, alpha=config['aesthetics']['axes']['grid_alpha'], linestyle='--')
        ax.set_axisbelow(True)

        # Labels and formatting
        ax.set_xlabel('Agent Type',
                     fontsize=config['fonts']['bold']['size'],
                     fontweight=config['fonts']['bold']['weight'],
                     labelpad=config['labels']['padding'])
        ax.set_ylabel('Ball Differential',
                     fontsize=config['fonts']['bold']['size'],
                     fontweight=config['fonts']['bold']['weight'],
                     labelpad=config['labels']['padding'])

        # Set tick labels
        ax.set_xticks(x)
        ax.set_xticklabels([
            config['agent_names'][agent] if agent in config['agent_names'] else agent
            for agent in agent_order
        ], rotation=45, ha='right', fontsize=config['fonts']['tick']['size'])
        
        ax.tick_params(axis='y', labelsize=config['fonts']['tick']['size'])

        # Legend
        legend = ax.legend(loc='upper left',
                          frameon=True,
                          fancybox=config['aesthetics']['legend']['fancybox'],
                          shadow=config['aesthetics']['legend']['shadow'],
                          fontsize=config['fonts']['tick']['size'])
        legend.get_frame().set_facecolor(config['aesthetics']['legend']['facecolor'])
        legend.get_frame().set_alpha(config['aesthetics']['legend']['alpha'])

        # Spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(config['aesthetics']['axes']['spine_width'])
            spine.set_color(config['aesthetics']['axes']['spine_color'])

    plt.tight_layout(h_pad=3.0)
    
    output_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}_differentials.pdf")
    plt.savefig(output_path, 
                dpi=config['aesthetics']['figure']['dpi'],
                bbox_inches='tight',
                facecolor=config['aesthetics']['figure']['facecolor'])
    plt.close()
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate ball differential plots for a specific model')
    parser.add_argument('--model_name', type=str, default='Meta-Llama-3.1-70B-Instruct-Turbo')
    parser.add_argument('--single', action='store_true', help='Generate a single plot with all games, wins, and losses', default=True)
    
    args = parser.parse_args()
    
    results_dir = f"{ROOT_DIR}/experiments/experiment_one/results"
    output_dir = f"{VISUALISATIONS_DIR}/experiment_one/single_model/"
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(load_most_recent_file(results_dir), 'r') as f:
        data = json.load(f)
    
    stats = extract_differential_stats(data, args.model_name)
    output_path = create_differential_plot(args.model_name, stats, output_dir, single_plot=args.single)
    
    print(f"Player position differential plot saved as: {output_path}")

if __name__ == "__main__":
    main()