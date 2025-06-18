import matplotlib.pyplot as plt
import numpy as np
import json, os, re, sys
import yaml
import argparse

from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from poolagent import DATA_DIR, VISUALISATIONS_DIR

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

def extract_b_size(model_name):
    match = re.search(r'-(\d+)B-', model_name)
    return int(match.group(1)) if match else float('inf')

def clean_model_name(name):
    return name.replace('Meta-', '').replace('-Instruct-Turbo', '')

def create_visualization(dataset_type='model'):
    # Validate dataset type
    if dataset_type not in ['model', 'skill']:
        raise ValueError("Dataset type must be either 'model' or 'skill'")
    
    # Construct filename based on dataset type
    filename = f'exp3_all_results_{dataset_type}_dataset.json'
    
    # Load data
    with open(f'{DATA_DIR}/{filename}', 'r') as file:
        data = json.load(file)

    # Sort and prepare data
    models = sorted(data.keys(), key=extract_b_size)
    display_names = [clean_model_name(model) for model in models]

    # Calculate metrics
    means_with_functions = [np.mean(data[model]['with_functions']) for model in models]
    errs_with_functions = [stats.sem(data[model]['with_functions']) for model in models]

    # Calculate average control
    means_control = []
    errs_control = []
    for model in models:
        combined_control = data[model]['control_with'] + data[model]['control_without']
        means_control.append(np.mean(combined_control))
        errs_control.append(stats.sem(combined_control))

    # Create plot
    fig, ax = plt.subplots(figsize=config['figure_sizes']['default'])
    fig.set_facecolor(config['aesthetics']['figure']['facecolor'])
    ax.set_facecolor(config['aesthetics']['axes']['facecolor'])

    x = np.arange(len(display_names))

    # Plot with functions line and error band
    ax.plot(x, means_with_functions, '-', color=config['colors'][0], marker='o',
            linewidth=2.5, markersize=10, label='With Heuristic Values', zorder=3)
    ax.fill_between(x, 
                    np.array(means_with_functions) - np.array(errs_with_functions),
                    np.array(means_with_functions) + np.array(errs_with_functions),
                    color=config['colors'][0], alpha=config['aesthetics']['error_region']['alpha'],
                    zorder=2)

    # Plot control line and error band
    ax.plot(x, means_control, '-', color=config['colors'][1], marker='o',
            linewidth=2.5, markersize=10, label='Control (Average)', zorder=3)
    ax.fill_between(x, 
                    np.array(means_control) - np.array(errs_control),
                    np.array(means_control) + np.array(errs_control),
                    color=config['colors'][1], alpha=config['aesthetics']['error_region']['alpha'],
                    zorder=2)

    # Add baseline
    ax.axhline(y=0.5, color=config['colors'][2], linestyle=':', 
               label='Baseline (0.5)', alpha=0.7, zorder=1)

    # Customize plot
    title_suffix = 'Language Model' 
    ax.set_ylabel('Preference',
                 fontsize=config['fonts']['bold']['size'],
                 fontweight=config['fonts']['bold']['weight'],
                 labelpad=config['labels']['padding'])
    ax.set_xlabel(f'{title_suffix}',
                 fontsize=config['fonts']['bold']['size'],
                 fontweight=config['fonts']['bold']['weight'],
                 labelpad=config['labels']['padding'])
    # ax.set_title(f'{title_suffix} Performance Comparison',
    #              fontsize=config['fonts']['title']['size'],
    #              fontweight=config['fonts']['title']['weight'],
    #              pad=config['labels']['title_padding'])

    # Grid and axis appearance
    ax.grid(True, alpha=config['aesthetics']['axes']['grid_alpha'], linestyle='--')
    ax.set_axisbelow(True)

    # Set tick labels with consistent font size
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, 
                       rotation=30, 
                       ha='right',
                       fontsize=config['fonts']['tick']['size'])
    
    # Set y-tick font size
    ax.tick_params(axis='y', labelsize=config['fonts']['tick']['size'])

    # Updated legend to be inside plot at top left
    legend = ax.legend(loc='upper left',
                      frameon=True,
                      fancybox=config['aesthetics']['legend']['fancybox'],
                      shadow=config['aesthetics']['legend']['shadow'],
                      fontsize=config['fonts']['tick']['size'])  # Match legend font size with ticks
    legend.get_frame().set_facecolor(config['aesthetics']['legend']['facecolor'])
    legend.get_frame().set_alpha(config['aesthetics']['legend']['alpha'])

    # Spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(config['aesthetics']['axes']['spine_width'])
        spine.set_color(config['aesthetics']['axes']['spine_color'])

    # # Add value labels with consistent font size
    # annotation_fontsize = config['fonts']['tick']['size'] - 2  # Slightly smaller than tick labels
    # for i, (func_mean, func_std, ctrl_mean, ctrl_std) in enumerate(zip(
    #     means_with_functions, stds_with_functions, means_control, stds_control)):
    #     ax.annotate(f'{func_mean:.3f}\n±{func_std:.3f}', 
    #                 (i, func_mean),
    #                 textcoords="offset points",
    #                 xytext=(0,10),
    #                 ha='center',
    #                 va='bottom',
    #                 fontsize=annotation_fontsize)
    #     ax.annotate(f'{ctrl_mean:.3f}\n±{ctrl_std:.3f}', 
    #                 (i, ctrl_mean),
    #                 textcoords="offset points",
    #                 xytext=(0,-20),
    #                 ha='center',
    #                 va='top',
    #                 fontsize=annotation_fontsize)

    plt.ylim(0, 1)

    plt.tight_layout()
    os.makedirs(f'{VISUALISATIONS_DIR}/experiment_three/', exist_ok=True)
    plt.savefig(f'{VISUALISATIONS_DIR}/experiment_three/{dataset_type}_performance.png',
                dpi=config['aesthetics']['figure']['dpi'],
                bbox_inches='tight',
                facecolor=config['aesthetics']['figure']['facecolor'])
    plt.close()

    print(f"Chart has been saved as '{dataset_type}_performance.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create visualization for model or skill dataset')
    parser.add_argument('--dataset_type', choices=['model', 'skill', 'both'], help='Type of dataset to visualize (model or skill)', default='both')
    args = parser.parse_args()
    
    if args.dataset_type == 'both':
        create_visualization('model')
        create_visualization('skill')
    else:
        create_visualization(args.dataset_type)