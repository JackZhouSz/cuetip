import json
import glob
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from poolagent.path import VISUALISATIONS_DIR

def load_plot_config(config_path='../plot_config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    for key, value in config['rcParams'].items():
        plt.rcParams[key] = value
    plt.style.use(config['style'])
    return config

config = load_plot_config()

def process_file(filepath):
    with open(filepath, 'r') as f:
        print(f"Loading {filepath}")
        data = json.load(f)
        
    # Find experience level
    experience = None
    for entry in data:
        if entry.get('task') == 'experience':
            experience = entry.get('response', "None")
            if experience == "None":
                experience = 0
            elif experience == "Beginner":
                experience = 1
            elif experience == "Intermediate":
                experience = 2
            elif experience == "Expert":
                experience = 3
            break
    
    # Get ratings for each source group
    ratings_1_3 = []
    ratings_4_6 = []
    
    for entry in data:
        if entry.get('task') == 'rate_explanation':
            source_num = entry.get('source_number')
            rating = entry.get('rating')
            if source_num and rating:
                if 1 <= source_num <= 3:
                    ratings_1_3.append(rating)
                elif 4 <= source_num <= 6:
                    ratings_4_6.append(rating)
    
    # Calculate averages
    avg_1_3 = np.mean(ratings_1_3) if ratings_1_3 else None
    avg_4_6 = np.mean(ratings_4_6) if ratings_4_6 else None
    
    return experience, avg_1_3, avg_4_6

def main():
    data_dir = 'data'
    
    # Initialize dictionaries to store ratings by experience level
    exp_ratings = {
        0: {'1-3': [], '4-6': []},  # None
        1: {'1-3': [], '4-6': []},  # Beginner
        2: {'1-3': [], '4-6': []},  # Intermediate
        3: {'1-3': [], '4-6': []}   # Expert
    }
    
    # Process all files
    experiences_1_3 = []
    experiences_4_6 = []
    avgs_1_3 = []
    avgs_4_6 = []
    
    for filepath in glob.glob(os.path.join(data_dir, '*.json')):
        exp, avg_1_3, avg_4_6 = process_file(filepath)
        if exp is not None:
            if avg_1_3 is not None:
                experiences_1_3.append(exp)
                avgs_1_3.append(avg_1_3)
                exp_ratings[exp]['1-3'].append(avg_1_3)
            if avg_4_6 is not None:
                experiences_4_6.append(exp)
                avgs_4_6.append(avg_4_6)
                exp_ratings[exp]['4-6'].append(avg_4_6)
    
    # Calculate overall averages and standard deviations
    overall_avg_1_3 = np.mean(avgs_1_3)
    overall_avg_4_6 = np.mean(avgs_4_6)
    overall_std_1_3 = np.std(avgs_1_3)
    overall_std_4_6 = np.std(avgs_4_6)
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print("-" * 50)
    print(f"With Heuristic Values: {overall_avg_1_3:.3f} ± {overall_std_1_3:.3f}")
    print(f"Control: {overall_avg_4_6:.3f} ± {overall_std_4_6:.3f}")
    
    # Set style parameters
    plt.style.use('seaborn')
    
    # Create figure for experience breakdown
    fig, ax = plt.subplots(figsize=config['figure_sizes']['default'])
    fig.set_facecolor(config['aesthetics']['figure']['facecolor'])
    ax.set_facecolor(config['aesthetics']['axes']['facecolor'])
    
    # Plot experience breakdown
    exp_levels = ['None', 'Beginner', 'Intermediate', 'Expert']
    x = np.arange(len(exp_levels))
    width = 0.35
    
    avgs_by_exp_1_3 = [np.mean(exp_ratings[i]['1-3']) if exp_ratings[i]['1-3'] else 0 for i in range(4)]
    avgs_by_exp_4_6 = [np.mean(exp_ratings[i]['4-6']) if exp_ratings[i]['4-6'] else 0 for i in range(4)]
    err_by_exp_1_3 = [stats.sem(exp_ratings[i]['1-3']) if exp_ratings[i]['1-3'] else 0 for i in range(4)]
    err_by_exp_4_6 = [stats.sem(exp_ratings[i]['4-6']) if exp_ratings[i]['4-6'] else 0 for i in range(4)]
    
    ax.bar(
        x - width/2, 
        avgs_by_exp_1_3, 
        width, 
        yerr=err_by_exp_1_3,
        label='With Heuristic Values', 
        color=config['colors'][0],
        capsize=5
    )
    ax.bar(
        x + width/2, 
        avgs_by_exp_4_6, 
        width, 
        yerr=err_by_exp_4_6,
        label='Control', 
        color=config['colors'][1],
        capsize=5
    )
    
    ax.set_xlabel('Experience Level',
                 fontsize=config['fonts']['bold']['size'],
                 fontweight=config['fonts']['bold']['weight'],
                 labelpad=config['labels']['padding'])
    ax.set_ylabel('Average Rating',
                 fontsize=config['fonts']['bold']['size'],
                 fontweight=config['fonts']['bold']['weight'],
                 labelpad=config['labels']['padding'])

    ax.set_xticks(x)
    ax.set_xticklabels(exp_levels,
        rotation=45, 
        fontsize=config['fonts']['tick']['size']
    )
    ax.tick_params(axis='y', labelsize=config['fonts']['tick']['size'])
    
    # Style improvements
    ax.grid(True, alpha=config['aesthetics']['axes']['grid_alpha'], linestyle='--')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(config['aesthetics']['axes']['spine_width'])
        spine.set_color(config['aesthetics']['axes']['spine_color'])
    
    # Legend
    legend = ax.legend(loc='upper left',
                      frameon=True,
                      fancybox=config['aesthetics']['legend']['fancybox'],
                      shadow=config['aesthetics']['legend']['shadow'],
                      fontsize=config['fonts']['tick']['size'])
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Print detailed statistics
    print("\nDetailed Statistics by Experience Level:")
    print("-" * 50)
    for i, level in enumerate(exp_levels):
        ratings_1_3 = exp_ratings[i]['1-3']
        ratings_4_6 = exp_ratings[i]['4-6']
        
        avg_1_3 = np.mean(ratings_1_3) if ratings_1_3 else 0
        avg_4_6 = np.mean(ratings_4_6) if ratings_4_6 else 0
        std_1_3 = np.std(ratings_1_3) if ratings_1_3 else 0
        std_4_6 = np.std(ratings_4_6) if ratings_4_6 else 0
        n_1_3 = len(ratings_1_3)
        n_4_6 = len(ratings_4_6)
        
        print(f"\n{level}:")
        print(f"With Heuristic Values: {avg_1_3:.2f} ± {std_1_3:.2f} (n={n_1_3})")
        print(f"Control: {avg_4_6:.2f} ± {std_4_6:.2f} (n={n_4_6})")
    
    # Calculate and print correlations
    if len(experiences_1_3) > 1 and len(set(experiences_1_3)) > 1:
        corr_1_3, p_1_3 = stats.pearsonr(experiences_1_3, avgs_1_3)
        print(f'\nWith Heuristic Values correlation with experience: r = {corr_1_3:.3f}, p = {p_1_3:.3f}')
    else:
        print('\nWith Heuristic Values: Insufficient variation in experience levels for correlation')
        
    if len(experiences_4_6) > 1 and len(set(experiences_4_6)) > 1:
        corr_4_6, p_4_6 = stats.pearsonr(experiences_4_6, avgs_4_6)
        print(f'Control correlation with experience: r = {corr_4_6:.3f}, p = {p_4_6:.3f}')
    else:
        print('Control: Insufficient variation in experience levels for correlation')
    
    plt.tight_layout()
    plt.savefig(f'exp_human_eval.pdf', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()