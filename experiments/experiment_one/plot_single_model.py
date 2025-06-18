import os
from collections import defaultdict
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import yaml
import sys
import os
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
IGNORE_AGENTS = ['BruteForceAgent']

def load_most_recent_file(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    if not files:
        raise FileNotFoundError("No JSON files found in the directory")
    return os.path.join(directory, max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f))))

def extract_model_stats(data, model_name):
    agent_types = defaultdict(lambda: {'games': [], 'as_agent1': [], 'as_agent2': []})
    
    for key, value in data.items():
        agent1, agent2 = key.split('---')
        agent1_base = agent1.split('_')[0]
        agent2_base = agent2.split('_')[0]

        if agent1_base in IGNORE_AGENTS or agent2_base in IGNORE_AGENTS:
            continue
        
        print(f"Processing: {agent1} vs {agent2}")
        
        agent_types[agent1_base]['games'].extend(value['games'])
        agent_types[agent2_base]['games'].extend([1 - g for g in value['games']])
        
        agent_types[agent1_base]['as_agent1'].extend(value['games'])
        agent_types[agent2_base]['as_agent2'].extend([1 - g for g in value['games']])

    results = {}
    for agent_type, data in agent_types.items():
        if data['games']:
            winrate = np.mean(data['games'])
            std_err = np.sqrt((winrate * (1 - winrate)) / len(data['games']))
            
            agent1_winrate = np.mean(data['as_agent1']) if data['as_agent1'] else np.nan
            agent1_stderr = np.sqrt((agent1_winrate * (1 - agent1_winrate)) / len(data['as_agent1'])) if data['as_agent1'] else np.nan
            
            agent2_winrate = np.mean(data['as_agent2']) if data['as_agent2'] else np.nan
            agent2_stderr = np.sqrt((agent2_winrate * (1 - agent2_winrate)) / len(data['as_agent2'])) if data['as_agent2'] else np.nan
            
            results[agent_type] = {
                'winrate': winrate,
                'std_err': std_err,
                'n_games': len(data['games']),
                'agent1_winrate': agent1_winrate,
                'agent1_stderr': agent1_stderr,
                'agent1_games': len(data['as_agent1']),
                'agent2_winrate': agent2_winrate,
                'agent2_stderr': agent2_stderr,
                'agent2_games': len(data['as_agent2'])
            }
        else:
            results[agent_type] = None
    
    return results

def create_bar_plot(model_name, stats, output_dir):
    fig, ax = plt.subplots(figsize=config['figure_sizes']['default'])
    fig.set_facecolor(config['aesthetics']['figure']['facecolor'])
    ax.set_facecolor(config['aesthetics']['axes']['facecolor'])
    
    agent_order = ['ProRandomBallAgent', 'LanguageAgent', 'LanguageDEFAgent', 'LanguageFunctionAgent', 'FunctionAgent', 'PoolMasterAgent']
    x = np.arange(len(agent_order))
    width = 0.25  # Width of bars
    
    winrates = []
    std_errs = []
    agent1_rates = []
    agent1_errs = []
    agent2_rates = []
    agent2_errs = []
    
    for agent in agent_order:
        if stats[agent] is not None:
            winrates.append(stats[agent]['winrate'])
            std_errs.append(stats[agent]['std_err'])
            agent1_rates.append(stats[agent]['agent1_winrate'])
            agent1_errs.append(stats[agent]['agent1_stderr'])
            agent2_rates.append(stats[agent]['agent2_winrate'])
            agent2_errs.append(stats[agent]['agent2_stderr'])
        else:
            winrates.append(np.nan)
            std_errs.append(np.nan)
            agent1_rates.append(np.nan)
            agent1_errs.append(np.nan)
            agent2_rates.append(np.nan)
            agent2_errs.append(np.nan)
    
    # Plot bars
    ax.bar(x - width, agent1_rates, width, yerr=agent1_errs,
           color=config['colors'][0], label='As Player 1')
    ax.bar(x, winrates, width, yerr=std_errs,
           color=config['colors'][1], label='Overall')
    ax.bar(x + width, agent2_rates, width, yerr=agent2_errs,
           color=config['colors'][2], label='As Player 2')
    
    # Grid and formatting
    ax.grid(True, alpha=config['aesthetics']['axes']['grid_alpha'], linestyle='--')
    ax.set_axisbelow(True)
    
    ax.set_xlabel('Agent Type',
                 fontsize=config['fonts']['bold']['size'],
                 fontweight=config['fonts']['bold']['weight'],
                 labelpad=config['labels']['padding'])
    ax.set_ylabel('Win Rate',
                 fontsize=config['fonts']['bold']['size'],
                 fontweight=config['fonts']['bold']['weight'],
                 labelpad=config['labels']['padding'])
    
    # Set axis limits
    all_rates = [r for rates in [winrates, agent1_rates, agent2_rates] 
                for r in rates if not np.isnan(r)]
    all_errs = [e for errs in [std_errs, agent1_errs, agent2_errs] 
                for e in errs if not np.isnan(e)]
    if all_rates and all_errs:
        max_val = max(r + e for r, e in zip(all_rates, all_errs))
        min_val = min(r - e for r, e in zip(all_rates, all_errs))
        ax.set_ylim(max(0, min_val - 0.05), min(1, max_val + 0.05))
    
    ax.set_xticks(x)
    ax.set_xticklabels([
        config['agent_names'][agent] if agent in config['agent_names'] else agent
        for agent in agent_order
    ], rotation=45, ha='right')
    
    legend = ax.legend(loc='upper left',
                      frameon=True,
                      fancybox=config['aesthetics']['legend']['fancybox'],
                      shadow=config['aesthetics']['legend']['shadow'])
    legend.get_frame().set_facecolor(config['aesthetics']['legend']['facecolor'])
    legend.get_frame().set_alpha(config['aesthetics']['legend']['alpha'])
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(config['aesthetics']['axes']['spine_width'])
        spine.set_color(config['aesthetics']['axes']['spine_color'])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}_self_play.pdf")
    plt.savefig(output_path,
                dpi=config['aesthetics']['figure']['dpi'],
                bbox_inches='tight',
                facecolor=config['aesthetics']['figure']['facecolor'])
    plt.close()
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate performance bar plot for a specific model')
    parser.add_argument('--model_name', type=str, help='Name of the model to analyze', 
                      default='Meta-Llama-3.1-70B-Instruct-Turbo')
    
    args = parser.parse_args()
    
    results_dir = f"{ROOT_DIR}/experiments/experiment_one/results"
    output_dir = f"{VISUALISATIONS_DIR}/experiment_one/single_model/"
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(load_most_recent_file(results_dir), 'r') as f:
        data = json.load(f)
    
    stats = extract_model_stats(data, args.model_name)
    output_path = create_bar_plot(args.model_name, stats, output_dir)
    
    print(f"Performance plot saved as: {output_path}")
    
    print(f"\nDetailed statistics for {args.model_name} (self-play):")
    for agent_type, results in stats.items():
        if results is not None:
            print(f"\n{agent_type}:")
            print(f"  Overall win rate: {results['winrate']:.3f} ± {results['std_err']:.3f}")
            print(f"  Total games: {results['n_games']}")
        else:
            print(f"\n{agent_type}: No games found")

if __name__ == "__main__":
    main()