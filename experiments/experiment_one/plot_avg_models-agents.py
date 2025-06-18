import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poolagent import VISUALISATIONS_DIR, ROOT_DIR

def get_latest_json_file(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    return max(json_files, key=lambda x: datetime.strptime(x, '%Y%m%d_%H%M%S.json'))

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def process_data(data):
    agents = set()
    models = set()
    win_rates = {}

    for key, value in data.items():
        winrate = value['winrate']
        agent1model1, agent2model2 = key.split('---')
        agent1, model1 = agent1model1.split('_')
        agent2, model2 = agent2model2.split('_')
        agents.update([agent1, agent2])
        models.update([model1, model2])

        # Update win rates for agent1 with model1
        win_rates.setdefault(agent1, {}).setdefault(model1, []).append(winrate)

    # Calculate average win rates
    for agent in win_rates:
        for model in win_rates[agent]:
            win_rates[agent][model] = sum(win_rates[agent][model]) / len(win_rates[agent][model])

    return win_rates, agents, models

def add_value_labels(ax, spacing=5):
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        label = f"{y_value:.2f}"
        
        ax.annotate(label, (x_value, y_value), xytext=(0, spacing),
                    textcoords="offset points", ha='center', va='bottom')

def plot_average_agent_win_rates(win_rates, agents, models):
    avg_win_rates = {}
    for agent in agents:
        num_models = sum(1 for model in models if model in win_rates[agent])
        avg_win_rates[agent] = sum(win_rates[agent][model] for model in models if model in win_rates[agent]) / num_models
    
    # Sort agents by average win rate
    avg_win_rates = {k: v for k, v in sorted(avg_win_rates.items(), key=lambda item: item[1], reverse=True)}

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    bars = ax.bar(avg_win_rates.keys(), avg_win_rates.values())
    plt.title('Average Win Rate of Agents Across All Models')
    plt.xlabel('Agents')
    plt.ylabel('Average Win Rate')
    plt.ylim(0, 1)
    add_value_labels(ax)
    plt.tight_layout()
    plt.savefig(f'{VISUALISATIONS_DIR}/experiment_one/averages_agents.png')
    plt.close()

def plot_average_model_win_rates(win_rates, agents, models):
    avg_win_rates = {}
    for model in models:
        num_agents = sum(1 for agent in agents if agent in win_rates and model in win_rates[agent])
        avg_win_rates[model] = sum(win_rates[agent][model] for agent in agents if agent in win_rates and model in win_rates[agent]) / num_agents
    
    avg_win_rates.pop('None', None)

    # Sort models by average win rate
    avg_win_rates = {k: v for k, v in sorted(avg_win_rates.items(), key=lambda item: item[1], reverse=True)}

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    bars = ax.bar(avg_win_rates.keys(), avg_win_rates.values())
    plt.title('Average Win Rate of Models Across All Agents')
    plt.xlabel('Models')
    plt.ylabel('Average Win Rate')
    plt.ylim(0, 1)
    add_value_labels(ax)
    plt.tight_layout()
    plt.savefig(f'{VISUALISATIONS_DIR}/experiment_one/averages_model.png')
    plt.close()

def main():
    results_dir = f'{ROOT_DIR}/experiments/experiment_one/results/'
    latest_file = get_latest_json_file(results_dir)
    data = read_json_file(os.path.join(results_dir, latest_file))

    os.makedirs(f'{VISUALISATIONS_DIR}/experiment_one', exist_ok=True)

    win_rates, agents, models = process_data(data)

    print('Agents:', agents)
    print('Models:', models)
    print('Win Rates:', json.dumps(win_rates, indent=2))

    plot_average_agent_win_rates(win_rates, agents, models)
    plot_average_model_win_rates(win_rates, agents, models)

if __name__ == "__main__":
    main()
