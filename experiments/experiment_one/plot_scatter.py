import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
AGENT_NAMES = {
    'ProRandomBallAgent': 'Random Ball Agent',
    'LanguageAgent': 'Language Agent',
    'LanguageDEFAgent': 'RA-Language Agent',
    'LanguageFunctionAgent': 'NA-Language Agent',
    'FunctionAgent': 'Network Agent'
}
IGNORE_AGENTS = ['BruteForceAgent']
from poolagent import VISUALISATIONS_DIR, ROOT_DIR

def load_most_recent_file(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    if not files:
        raise FileNotFoundError("No JSON files found in the directory")
    
    most_recent_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return os.path.join(directory, most_recent_file)

def extract_configurations(data):
    configurations = set()
    for key in data.keys():
        agents = key.split('---')

        if any(agent.split('_')[0] in IGNORE_AGENTS for agent in agents):
            continue

        configurations.update(agents)

    configurations = [c.replace('-gguf', '') for c in configurations]
    configurations = [c.split('_')[0] + f"({c.split('_')[1]})" for c in configurations]
    configurations = [c.replace('(None)', '') for c in configurations]

    return sorted(list(configurations), key=lambda x: x.lower())

def calculate_winrates_with_std(data, configurations):
    """Calculate average winrates and standard deviations for each agent across both player positions."""
    winrates = {}
    std_errs = {}
    
    for agent in configurations:
        all_games = []
        
        # Look for matches where this agent appears
        for key, value in data.items():
            agent1, agent2 = key.split('---')
            
            # Clean agent names to match configuration format
            agent1 = clean_agent_name(agent1)
            agent2 = clean_agent_name(agent2)
            
            if agent1 == agent:
                # Games array directly shows agent1's wins (1) and losses (0)
                all_games.extend(value['games'])
            elif agent2 == agent:
                # Need to invert the results for agent2's perspective
                all_games.extend([1 - g for g in value['games']])
        
        if all_games:
            winrates[agent] = np.mean(all_games)
            # For binary outcomes, std = sqrt(p*(1-p)/n) where p is probability of success
            std_errs[agent] = np.sqrt((winrates[agent] * (1 - winrates[agent])) / len(all_games))
    
    return winrates, std_errs

def calculate_baseline_winrates_with_std(data):
    """Calculate average winrates and standard deviations for baseline agents."""
    function_games = []
    random_games = []
    
    for key, value in data.items():
        agent1, agent2 = key.split('---')
        
        # Check if either agent is a baseline agent
        if 'FunctionAgent' in agent1:
            if not any(lm in agent2.lower() for lm in ['llama', 'mistral', 'mixtral', 'phi', 'solar', 'starling']):
                function_games.extend(value['games'])
        elif 'FunctionAgent' in agent2:
            if not any(lm in agent1.lower() for lm in ['llama', 'mistral', 'mixtral', 'phi', 'solar', 'starling']):
                function_games.extend([1 - g for g in value['games']])
                
        if 'ProRandomBallAgent' in agent1:
            if not any(lm in agent2.lower() for lm in ['llama', 'mistral', 'mixtral', 'phi', 'solar', 'starling']):
                random_games.extend(value['games'])
        elif 'ProRandomBallAgent' in agent2:
            if not any(lm in agent1.lower() for lm in ['llama', 'mistral', 'mixtral', 'phi', 'solar', 'starling']):
                random_games.extend([1 - g for g in value['games']])
    
    function_stats = (None, None)
    if function_games:
        mean = np.mean(function_games)
        std = np.sqrt((mean * (1 - mean)) / len(function_games))
        function_stats = (mean, std)
    
    random_stats = (None, None)
    if random_games:
        mean = np.mean(random_games)
        std = np.sqrt((mean * (1 - mean)) / len(random_games))
        random_stats = (mean, std)
    
    return function_stats, random_stats

def clean_agent_name(agent):
    """Clean agent name to match configuration format."""
    agent = agent.replace('-gguf', '')
    agent = agent.split('_')[0] + f"({agent.split('_')[1]})"
    return agent.replace('(None)', '')

def create_scatter_plot(configurations, winrates, std_devs, model_sizes, function_baseline, random_baseline):
    plt.figure(figsize=(15, 8))
    
    # Separate agents by type and API/non-API
    language_agents_oss = [c for c in configurations if 'LanguageAgent' in c and 
                          not any(api in c.lower() for api in ['gpt-4o-mini', 'gpt-4o'])]
    function_agents_oss = [c for c in configurations if 'LanguageFunctionAgent' in c and 
                          not any(api in c.lower() for api in ['gpt-4o-mini', 'gpt-4o'])]
    
    api_agents = [c for c in configurations if any(api in c.lower() for api in ['gpt-4o-mini', 'gpt-4o'])]
    
    def plot_agent_group(agents, color, label, api=False):
        if not agents:
            return
            
        sizes = [model_sizes[agent.split('(')[1].rstrip(')')] for agent in agents]
        win_rates = [winrates[agent] for agent in agents]
        std_deviations = [std_devs[agent] for agent in agents]
        
        # For API models, use fixed x positions but keep original labels
        if api:
            x_positions = []
            for agent in agents:
                if 'gpt-4o-mini' in agent.lower():
                    x_positions.append(400)  # Position for API-small
                else:
                    x_positions.append(600)  # Position for API-large
            sizes = x_positions
            
        # Always use the original model names for point labels
        labels = [agent.split('(')[1].rstrip(')') for agent in agents]
        labels = [f"{AGENT_NAMES[label]} ({label}B)" for label in labels]
        
        # Plot points
        plt.scatter(sizes, win_rates, 
                   c=color,
                   s=100,  # Small fixed size for center points
                   alpha=1.0,
                   label=label,
                   zorder=3)
        
        # Plot standard deviation circles
        for size, rate, std in zip(sizes, win_rates, std_deviations):
            circle = plt.Circle((size, rate), std, 
                              color=color, 
                              alpha=0.2,
                              fill=True,
                              zorder=2)
            plt.gca().add_patch(circle)
        
        # Add labels for each point
        for i, (size, rate, label_text, std) in enumerate(zip(sizes, win_rates, labels, std_deviations)):
            plt.annotate(f"{label_text}",
                        (size, rate),
                        xytext=(5, 5),
                        textcoords='offset points',
                        zorder=4)
    
    # Plot open source models
    plot_agent_group(language_agents_oss, 'red', 'LanguageAgent')
    plot_agent_group(function_agents_oss, 'green', 'LanguageFunctionAgent')
    
    # Plot API models
    api_language = [a for a in api_agents if 'LanguageAgent' in a]
    api_function = [a for a in api_agents if 'LanguageFunctionAgent' in a]
    plot_agent_group(api_language, 'red', None, api=True)
    plot_agent_group(api_function, 'green', None, api=True)
    
    # Add baseline reference lines if available
    if function_baseline[0] is not None:
        plt.axhline(y=function_baseline[0], 
                   color='green', 
                   linestyle=':', 
                   alpha=0.8,
                   label=f'FunctionAgent')
    
    if random_baseline[0] is not None:
        plt.axhline(y=random_baseline[0], 
                   color='red', 
                   linestyle=':', 
                   alpha=0.8,
                   label=f'ProRandomBallAgent')
    
    # Set x-axis to log scale for the main section
    plt.xscale('log')
    
    # Customize x-axis ticks
    oss_sizes = sorted(set(size for name, size in model_sizes.items() 
                          if not any(api in name.lower() for api in ['gpt-4o-mini', 'gpt-4o'])))
    
    # Create two sets of ticks: one for OSS models, one for API
    all_ticks = list(oss_sizes) + [400, 600]  # Add API positions
    all_labels = [f'{size}B' if size < 200 else '' for size in oss_sizes] + ['API-small', 'API-large']
    
    plt.xticks(all_ticks, all_labels)
    
    # Add vertical line to separate API section
    plt.axvline(x=200, color='gray', linestyle='--', alpha=0.5)
    
    # Set x-axis limits to give more room to API section
    plt.xlim(2, 800)
    
    plt.xlabel('Model Size')
    plt.ylabel('Average Win Rate')
    plt.title('LM Win Rates vs Model Size (with Standard Deviation)')
    plt.legend()
    
    # Set reasonable y-axis limits with some padding
    all_rates = list(winrates.values())
    all_stds = list(std_devs.values())
    max_rate_with_std = max(rate + std for rate, std in zip(all_rates, all_stds))
    min_rate_with_std = min(rate - std for rate, std in zip(all_rates, all_stds))
    plt.ylim(0,1)
    
    plt.tight_layout()
    plt.savefig(f"{VISUALISATIONS_DIR}/experiment_one/scatter.png", 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

def main():
    # Dictionary mapping model names to their sizes in billions of parameters
    MODEL_SIZES = {
        'gpt-4o': 200,
        'gpt-4o-mini': 100,
        'llava-v1.6-mistral-7b-hf': 7,
        'Phi-3.5-vision-instruct': 3.5,
        'llava-v1.6-vicuna-13b-hf': 13,
        'llava-v1.6-34b-hf': 34,
        'Llama-3.2-11B-Vision-Instruct': 11,
        'Llama-3.2-90B-Vision-Instruct': 90,
        'Llama-3.2-3B-Instruct-Turbo': 4,
        'Meta-Llama-3.1-8B-Instruct-Turbo': 8,
        'Meta-Llama-3.1-70B-Instruct-Turbo': 70,
        'Meta-Llama-3.1-405B-Instruct-Turbo': 405,
    }
    
    results_directory = f"{ROOT_DIR}/experiments/experiment_one/results/"
    most_recent_file = load_most_recent_file(results_directory)
    os.makedirs(f"{VISUALISATIONS_DIR}/experiment_one", exist_ok=True)
    
    with open(most_recent_file, 'r') as f:
        data = json.load(f)
    
    configurations = extract_configurations(data)
    winrates, std_devs = calculate_winrates_with_std(data, configurations)
    
    # Calculate baseline winrates with standard deviations
    function_baseline, random_baseline = calculate_baseline_winrates_with_std(data)
    
    create_scatter_plot(configurations, winrates, std_devs, MODEL_SIZES, 
                       function_baseline, random_baseline)
    
    print(f"Scatter plot generated from file: {most_recent_file}")
    print("Scatter plot saved as 'exp1_scatter.png'")
    if function_baseline[0] is not None:
        print(f"FunctionAgent baseline winrate: {function_baseline[0]:.3f}")
    if random_baseline[0] is not None:
        print(f"ProRandomBallAgent baseline winrate: {random_baseline[0]:.3f}")

if __name__ == "__main__":
    main()