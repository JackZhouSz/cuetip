import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import yaml

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poolagent import VISUALISATIONS_DIR, ROOT_DIR

IGNORE_AGENTS = ['BruteForceAgent']
AGENT_NAMES = {
    'ProRandomBallAgent': 'SA',
    'LanguageAgent': 'LA',
    'LanguageDEFAgent': 'LHA',
    'LanguageFunctionAgent': 'LNA',
    'FunctionAgent': 'NA',
    'PoolMasterAgent': "PM"
}
# List of non-LM agents that should always be included
NON_LM_AGENTS = {
    'ProRandomBallAgent',
    'FunctionAgent'
}

def load_most_recent_file(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    if not files:
        raise FileNotFoundError("No JSON files found in the directory")
    
    most_recent_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return os.path.join(directory, most_recent_file)

def should_include_configuration(config, model_filter):
    """
    Determine if a configuration should be included based on the model filter.
    Returns True if:
    - It's a non-LM agent
    - It matches the model filter
    - There's no model filter applied
    """
    base_agent = config.split('_')[0]
    if base_agent == 'PoolMasterAgent':
        return True
    if base_agent in IGNORE_AGENTS:
        return False
    if base_agent in NON_LM_AGENTS:
        return True
    if not model_filter:
        return True
    return model_filter.lower() in config.lower()

def extract_configurations(data, model_filter=None):
    configurations = set()
    for key in data.keys():
        agents = key.split('---')
        # Only add agents that pass the filter
        for agent in agents:
            if should_include_configuration(agent, model_filter):
                configurations.add(agent)

    configurations = [c.replace('-gguf', '') for c in configurations]
    configurations = [c.split('_')[0] + f"({c.split('_')[1]})" for c in configurations]
    configurations = [c.replace('(None)', '') for c in configurations]

    # Sort configurations based on the order in AGENT_NAMES
    ordered_configurations = sorted(configurations, 
                                    key=lambda x: list(AGENT_NAMES.keys()).index(x.split('(')[0]) 
                                    if x.split('(')[0] in AGENT_NAMES else float('inf'))
    return ordered_configurations


    return configurations

def create_heatmap_data(data, configurations, model_filter=None):
    n = len(configurations)
    heatmap_data = pd.DataFrame(index=configurations[::-1], columns=configurations)
    
    for key, value in data.items():
        agent1, agent2 = key.split('---')
        
        # Skip pairs that don't meet our filtering criteria
        if not (should_include_configuration(agent1, model_filter) and 
                should_include_configuration(agent2, model_filter)):
            continue

        # Process agent1
        agent1 = agent1.replace('-gguf', '')
        agent1 = agent1.split('_')[0] + f"({agent1.split('_')[1]})"
        agent1 = agent1.replace('(None)', '')
        
        # Process agent2
        agent2 = agent2.replace('-gguf', '')
        agent2 = agent2.split('_')[0] + f"({agent2.split('_')[1]})"
        agent2 = agent2.replace('(None)', '')

        # Only add data if both agents are in our filtered configurations
        if agent1 in configurations and agent2 in configurations:
            heatmap_data.loc[agent1, agent2] = value['winrate']
    
    # Convert to numeric, replacing any non-numeric values with NaN
    heatmap_data = heatmap_data.apply(pd.to_numeric, errors='coerce')
    
    # Remove empty rows and columns
    heatmap_data = heatmap_data.dropna(how='all').dropna(how='all', axis=1)
    
    return heatmap_data

def load_plot_config(config_path='../plot_config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    for key, value in config['rcParams'].items():
        plt.rcParams[key] = value
    plt.style.use(config['style'])
    return config

def plot_heatmap(heatmap_data, model_filter=None):
    plt.figure(figsize=(15, 12))
    
    # Create mask for NaN values only
    mask = np.isnan(heatmap_data.values)
    
    # Create the heatmap without masking the diagonal
    ax = sns.heatmap(heatmap_data, 
                     annot=True, 
                     cmap="YlGnBu", 
                     fmt='.2f',
                     cbar_kws={'label': 'Win Rate'}, 
                     mask=mask,
                     annot_kws={'size': plt.rcParams['font.size']})
    
    # Add red diagonal squares (except for corners)
    n = len(heatmap_data)
    for i in range(n+1):
        ax.add_patch(plt.Rectangle((i-1, n-i), 1, 1, 
                                    fill=True, 
                                    facecolor='red', 
                                    edgecolor='none'))
    
    # Add red lines to separate agents
    agent_types = [config.split('(')[0] for config in heatmap_data.index]
    separators = [i for i in range(1, len(agent_types)) 
                 if agent_types[i] != agent_types[i-1]]
    
    if not model_filter:
        for sep in separators:
            ax.axhline(y=sep, color='red', linewidth=2)
            ax.axvline(x=sep, color='red', linewidth=2)
    
    plt.xlabel("Agent Two", fontsize=plt.rcParams['axes.labelsize'])
    plt.ylabel("Agent One", fontsize=plt.rcParams['axes.labelsize'])

    # Set x and y labels to AGENT_NAMES
    x_labels = [AGENT_NAMES.get(agent.split('(')[0], agent) if '(' in agent else AGENT_NAMES.get(agent, agent) for agent in heatmap_data.columns]
    y_labels = [AGENT_NAMES.get(agent.split('(')[0], agent) if '(' in agent else AGENT_NAMES.get(agent, agent) for agent in heatmap_data.index]
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_yticklabels(y_labels, rotation=0)
    plt.tight_layout()
    
    # Save with model filter in filename if specified
    filename = "heatmap.pdf"
    if model_filter:
        filename = f"single_model/heatmap_{model_filter.lower().replace(' ', '_')}.pdf"
    
    plt.savefig(f"{VISUALISATIONS_DIR}/experiment_one/{filename}", 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate heatmap for agent performance')
    parser.add_argument('--model', type=str, default=None, help='Filter results to include only specified model (and non-LM agents)')
    args = parser.parse_args()
    
    results_directory = f"{ROOT_DIR}/experiments/experiment_one/results/"
    most_recent_file = load_most_recent_file(results_directory)
    os.makedirs(f"{VISUALISATIONS_DIR}/experiment_one", exist_ok=True)
    
    # Load plot configuration
    config = load_plot_config()

    with open(most_recent_file, 'r') as f:
        data = json.load(f)
    
    configurations = extract_configurations(data, args.model)
    heatmap_data = create_heatmap_data(data, configurations, args.model)
    plot_heatmap(heatmap_data, args.model)
    
    print(f"Heatmap generated from file: {most_recent_file}")
    filename = "heatmap.pdf"
    if args.model:
        filename = f"heatmap_{args.model.lower().replace(' ', '_')}.pdf"
    print(f"Heatmap saved as '{filename}'")

if __name__ == "__main__":
    main()