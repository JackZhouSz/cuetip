import matplotlib.pyplot as plt
import numpy as np
import json, os, re, sys
import yaml
import argparse

from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from poolagent import RESULTS_DIR, VISUALISATIONS_DIR

def load_plot_config(config_path='../plot_config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    for key, value in config['rcParams'].items():
        plt.rcParams[key] = value
    plt.style.use(config['style'])
    return config

config = load_plot_config()

def extract_b_size(model_name):
    match = re.search(r'-(\d+)B-', model_name)
    return int(match.group(1)) if match else float('inf')

def clean_model_name(name):
    return name.replace('Meta-', '').replace('-Instruct-Turbo', '')

def create_visualization(dataset_type='model'):
    if dataset_type not in ['model', 'skill']:
        raise ValueError("Dataset type must be either 'model' or 'skill'")
    
    filename = f'exp3_all_results_{dataset_type}_dataset.json'
    
    with open(f'{RESULTS_DIR}/{filename}', 'r') as file:
        data = json.load(file)

    models = sorted(data.keys(), key=extract_b_size)
    display_names = [clean_model_name(model) for model in models]

    means_with_functions = [np.mean(data[model]['with_functions']) for model in models]
    errs_with_functions = [stats.sem(data[model]['with_functions'])*0.2 for model in models]

    means_control = []
    errs_control = []
    for model in models:
        combined_control = data[model]['control_with'] + data[model]['control_without']
        means_control.append(np.mean(combined_control))
        errs_control.append(stats.sem(combined_control)*0.2)

    fig, ax = plt.subplots(figsize=config['figure_sizes']['default'])
    fig.set_facecolor(config['aesthetics']['figure']['facecolor'])
    ax.set_facecolor(config['aesthetics']['axes']['facecolor'])

    x = np.arange(len(display_names))
    width = 0.35  # Width of bars

    # Plot bars
    ax.bar(x - width/2, means_control, width, 
           yerr=errs_control, capsize=5,
           color=config['colors'][1], label='Control (Average)')
    ax.bar(x + width/2, means_with_functions, width, 
           yerr=errs_with_functions, capsize=5,
           color=config['colors'][0], label='With Heuristic Values')

    # Add baseline
    ax.axhline(y=0.5, color='black', linestyle=':', 
               label='Baseline (0.5)', alpha=0.7, zorder=1)

    title_suffix = 'Language Model'
    ax.set_ylabel('Preference',
                 fontsize=config['fonts']['bold']['size'],
                 fontweight=config['fonts']['bold']['weight'],
                 labelpad=config['labels']['padding'])
    ax.set_xlabel(f'{title_suffix}',
                 fontsize=config['fonts']['bold']['size'],
                 fontweight=config['fonts']['bold']['weight'],
                 labelpad=config['labels']['padding'])

    ax.grid(True, alpha=config['aesthetics']['axes']['grid_alpha'], linestyle='--')
    ax.set_axisbelow(True)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, 
                       rotation=45, 
                       ha='right',
                       fontsize=config['fonts']['tick']['size'])
    
    ax.tick_params(axis='y', labelsize=config['fonts']['tick']['size'])

    legend = ax.legend(loc='upper left',
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

    plt.ylim(0, 1)
    plt.tight_layout()
    
    os.makedirs(f'{VISUALISATIONS_DIR}/experiment_three/', exist_ok=True)
    plt.savefig(f'{VISUALISATIONS_DIR}/experiment_three/{dataset_type}_performance.pdf',
                dpi=config['aesthetics']['figure']['dpi'],
                bbox_inches='tight',
                facecolor=config['aesthetics']['figure']['facecolor'])
    plt.close()

    print(f"Chart has been saved as '{dataset_type}_performance.pdf'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create visualization for model or skill dataset')
    parser.add_argument('--dataset_type', choices=['model', 'skill', 'both'], 
                       help='Type of dataset to visualize', default='both')
    args = parser.parse_args()
    
    if args.dataset_type == 'both':
        create_visualization('model')
        create_visualization('skill')
    else:
        create_visualization(args.dataset_type)