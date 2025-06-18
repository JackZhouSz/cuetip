import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from poolagent import VISUALISATIONS_DIR, DATA_DIR, Pool


def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def perform_pca_analysis(data):
    values = np.array([entry['values'] for entry in data])
    mean_values = np.array([entry['mean'] for entry in data])
    
    # Normalize the values
    #values = (values - values.mean(axis=0)) / values.std(axis=0)
    
    value_sums = values.sum(axis=1)
    correlation = np.corrcoef(value_sums, mean_values)[0,1]
    print(f"Correlation between sum of values and mean value: {correlation:.3f}")
    
    # Now get both 2D and 3D PCA results
    pca_2d = PCA(n_components=2)
    pca_3d = PCA(n_components=3)
    pca_result_2d = pca_2d.fit_transform(values)
    pca_result_3d = pca_3d.fit_transform(values)
    
    print(f"Explained variance ratios (2D): {pca_2d.explained_variance_ratio_}")
    print(f"Explained variance ratios (3D): {pca_3d.explained_variance_ratio_}")
    
    return pca_result_2d, pca_result_3d, mean_values, pca_2d.explained_variance_ratio_, pca_3d.explained_variance_ratio_

def perform_pca_analysis(data):
    values = np.array([entry['values'] for entry in data])
    mean_values = np.array([entry['mean'] for entry in data])
    
    value_sums = values.sum(axis=1)
    correlation = np.corrcoef(value_sums, mean_values)[0,1]
    print(f"Correlation between sum of values and mean value: {correlation:.3f}")
    
    pca = PCA(n_components=2)
    pca_result_2d = pca.fit_transform(values)
    pca_3d = PCA(n_components=3)
    pca_result_3d = pca_3d.fit_transform(values)
    
    print(f"Explained variance ratios (2D): {pca.explained_variance_ratio_}")
    print(f"Explained variance ratios (3D): {pca_3d.explained_variance_ratio_}")
    
    return pca_result_2d, pca_result_3d, mean_values, pca.explained_variance_ratio_, pca_3d.explained_variance_ratio_

def plot_pca_results(pca_result, mean_values, exp_var_ratio):    
    plt.figure(figsize=(12, 10))
    
    colors = ['blue', 'white', 'red']
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1],
                         c=mean_values, cmap=cmap, 
                         vmin=0, vmax=1, alpha=0.6)
    
    plt.colorbar(scatter, label='Mean Value')
    plt.xlabel(f'First Principal Component ({exp_var_ratio[0]:.1%} variance explained)')
    plt.ylabel(f'Second Principal Component ({exp_var_ratio[1]:.1%} variance explained)')
    plt.title('PCA of Value Vectors')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{VISUALISATIONS_DIR}/experiment_appendix/exp_val_correlation.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_pca_3d(pca_result_3d, mean_values, exp_var_ratio):
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['blue', 'white', 'red']
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
    scatter = ax.scatter(pca_result_3d[:, 0], 
                        pca_result_3d[:, 1], 
                        pca_result_3d[:, 2],
                        c=mean_values, 
                        cmap=cmap,
                        alpha=0.6,
                        vmin=0, vmax=1)
    
    plt.colorbar(scatter, label='Mean Value')
    
    ax.set_xlabel(f'PC1 ({exp_var_ratio[0]:.1%})')
    ax.set_ylabel(f'PC2 ({exp_var_ratio[1]:.1%})')
    ax.set_zlabel(f'PC3 ({exp_var_ratio[2]:.1%})')
    
    plt.title('3D PCA of Value Vectors')
    ax.view_init(elev=20, azim=45)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{VISUALISATIONS_DIR}/experiment_appendix/exp_val_correlation_3d.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_improvement_distribution(mean_values):
    plt.figure(figsize=(10, 6))
    
    density = gaussian_kde(mean_values)
    xs = np.linspace(-1, 1, 200)
    plt.plot(xs, density(xs))
    
    plt.xlabel('Mean Value')
    plt.ylabel('Density')
    plt.title('Distribution of Mean Values')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    mean = np.mean(mean_values)
    median = np.median(mean_values)
    plt.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.3f}')
    plt.axvline(median, color='g', linestyle='--', label=f'Median: {median:.3f}')
    plt.legend()

    plt.xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{VISUALISATIONS_DIR}/experiment_appendix/exp_val_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def filter_by_balls(data):
    # Filter for valid game states
    filtered_data = []
    mean_values = []
    
    for k, entry in data.items():
        # positions = entry['end_state']['positions']
        # has_target = any(ball in positions for ball in ['red', 'blue', 'yellow'])
        # has_color = any(ball in positions for ball in ['green', 'black', 'pink'])
        
        # if has_target and has_color:
        #     filtered_data.append(entry)
        #     mean_values.append(entry['mean'])
    
        mean_values.append(entry['value_estimate'])
        filtered_data.append(entry)


    print(f"Filtered out {len(data) - len(filtered_data)} entries")
    return filtered_data, mean_values

def save_examples(good_entries, bad_entries):

    env = Pool()

    os.makedirs(f"{VISUALISATIONS_DIR}/experiment_appendix/examples", exist_ok=True)

    print("Saving example shots...")

    for i, entry in enumerate(good_entries):
        env.save_shot_gif(entry['starting_state'], entry['params'], f"{VISUALISATIONS_DIR}/experiment_appendix/examples/good_{i}.gif")
        print(f"Good example {i} saved")

    for i, entry in enumerate(bad_entries):
        env.save_shot_gif(entry['starting_state'], entry['params'], f"{VISUALISATIONS_DIR}/experiment_appendix/examples/bad_{i}.gif")
        print(f"Bad example {i} saved")


def main():
    data = load_json_data(f"{DATA_DIR}/shot_task_dataset.json")
    data, mean_values = filter_by_balls(data)
    # pca_result_2d, pca_result_3d, mean_values, exp_var_ratio_2d, exp_var_ratio_3d = perform_pca_analysis(data)

    # # Create visualizations
    # plot_pca_results(pca_result_2d, mean_values, exp_var_ratio_2d)
    # plot_pca_3d(pca_result_3d, mean_values, exp_var_ratio_3d)
    plot_improvement_distribution(mean_values)

    # Select examples based on filtered mean values
    sorted_indices = np.argsort(mean_values)
    good_entries = [data[i] for i in sorted_indices[-3:]]
    bad_entries = [data[i] for i in sorted_indices[:3]]
    save_examples(good_entries, bad_entries)
    

if __name__ == '__main__':
    main()
