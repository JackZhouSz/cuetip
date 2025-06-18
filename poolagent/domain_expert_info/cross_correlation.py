import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from poolagent.path import DATA_DIR
from poolagent.domain_expert_info.training import VALUE_INPUT_SIZE, DIFFICULTY_INPUT_SIZE

def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data['train']

def extract_functions(data):
    value_functions = [[] for _ in range(VALUE_INPUT_SIZE)]
    difficulty_functions = [[] for _ in range(DIFFICULTY_INPUT_SIZE)]

    for entry in data:
        for s in range(10):
            for i in range(VALUE_INPUT_SIZE):
                value_functions[i].append(entry['values'][s][i])
            for i in range(DIFFICULTY_INPUT_SIZE):
                difficulty_functions[i].append(entry['difficulties'][s][i])

    return value_functions, difficulty_functions

def calculate_correlation_matrix(functions):
    n = len(functions)
    correlation_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            correlation = pearsonr(functions[i], functions[j]).statistic
            correlation_matrix[i, j] = correlation if not np.isnan(correlation) else 0

    return correlation_matrix

def plot_heatmap(correlation_matrix, title, xlabel, ylabel):
    plt.figure(figsize=(10, 8))
    im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, label='Correlation Coefficient')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add value annotations
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            plt.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                     ha='center', va='center', color='black', fontsize=8)

    plt.xticks(range(correlation_matrix.shape[1]), range(1, correlation_matrix.shape[1] + 1))
    plt.yticks(range(correlation_matrix.shape[0]), range(1, correlation_matrix.shape[0] + 1))

    plt.tight_layout()
    plt.show()

def print_top_correlations(correlation_matrix, func_type1, func_type2):
    if func_type1 == func_type2:
        # For same-type correlations, we only want the upper triangle
        mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        correlations = correlation_matrix[mask]
    else:
        correlations = correlation_matrix.flatten()

    top_indices = np.argsort(np.abs(correlations))[-5:][::-1]
    print(f"\nTop 5 highest absolute correlations between {func_type1} and {func_type2}:")
    for idx in top_indices:
        if func_type1 == func_type2:
            i, j = np.unravel_index(idx, correlation_matrix.shape)
            i, j = i + 1, j + 1  # Adding 1 for 1-based indexing
        else:
            i, j = divmod(idx, correlation_matrix.shape[1])
            i, j = i + 1, j + 1  # Adding 1 for 1-based indexing
        print(f"{func_type1} {i} and {func_type2} {j}: {correlation_matrix[i-1, j-1]:.4f}")

# Main execution
filename = f'{DATA_DIR}/training_data.json'
data = load_data(filename)

value_functions, difficulty_functions = extract_functions(data)

# Calculate correlation matrices
val_to_val = calculate_correlation_matrix(value_functions)
diff_to_diff = calculate_correlation_matrix(difficulty_functions)
val_to_diff = calculate_correlation_matrix(value_functions + difficulty_functions)[:VALUE_INPUT_SIZE, VALUE_INPUT_SIZE:]

# Plot heatmaps
plot_heatmap(val_to_val, 'Correlation: Value Functions to Value Functions', 'Value Function Index', 'Value Function Index')
plot_heatmap(diff_to_diff, 'Correlation: Difficulty Functions to Difficulty Functions', 'Difficulty Function Index', 'Difficulty Function Index')
plot_heatmap(val_to_diff, 'Cross-correlation: Value Functions to Difficulty Functions', 'Difficulty Function Index', 'Value Function Index')

# Print top correlations
print_top_correlations(val_to_val, 'Value Function', 'Value Function')
print_top_correlations(diff_to_diff, 'Difficulty Function', 'Difficulty Function')
print_top_correlations(val_to_diff, 'Value Function', 'Difficulty Function')
