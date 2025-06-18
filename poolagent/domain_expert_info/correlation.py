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

def calculate_correlations(data):
    value_correlations = np.zeros(VALUE_INPUT_SIZE + 1)  # +1 for sum
    difficulty_correlations = np.zeros(DIFFICULTY_INPUT_SIZE + 1)  # +1 for sum

    value_function_values = [[] for _ in range(VALUE_INPUT_SIZE + 1)]
    difficulty_function_values = [[] for _ in range(DIFFICULTY_INPUT_SIZE + 1)]
    visit_distributions = []

    for entry in data:
        visit_distributions.extend(entry['visit_distribution'])
        
        for s in range(10):

            # Process value functions
            for i in range(VALUE_INPUT_SIZE):
                value_function_values[i].append(entry['values'][s][i])
            
            # Calculate sum for each set of VALUE_INPUT_SIZE values
            value_function_values[VALUE_INPUT_SIZE].append(sum(entry['values'][s]))

            # Process difficulty functions
            for i in range(DIFFICULTY_INPUT_SIZE):
                difficulty_function_values[i].append(entry['difficulties'][s][i])
        
            # Calculate sum for DIFFICULTY_INPUT_SIZE values
            difficulty_function_values[DIFFICULTY_INPUT_SIZE].append(sum(entry['difficulties'][s]))

    # Calculate correlations for value functions
    for i in range(VALUE_INPUT_SIZE + 1):
        correlation = pearsonr(value_function_values[i], visit_distributions).statistic
        value_correlations[i] = correlation if not np.isnan(correlation) else 0

    # Calculate correlations for difficulty functions
    for i in range(DIFFICULTY_INPUT_SIZE + 1):
        correlation = pearsonr(difficulty_function_values[i], visit_distributions).statistic
        difficulty_correlations[i] = correlation if not np.isnan(correlation) else 0

    return value_correlations, difficulty_correlations

def plot_correlations(value_correlations, difficulty_correlations):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Value correlations
    ax1.bar(range(1, len(value_correlations) + 1), value_correlations)
    ax1.set_xlabel('Value Function Index (last = Sum of all functions)')
    ax1.set_ylabel('Correlation Coefficient')
    ax1.set_title('Correlation between Value Function and Visit Distribution')
    ax1.set_xticks(range(1, len(value_correlations) + 1))
    ax1.set_xticklabels([*range(1, len(value_correlations)), 'Sum'])
    
    # Difficulty correlations
    ax2.bar(range(1, len(difficulty_correlations) + 1), difficulty_correlations)
    ax2.set_xlabel('Difficulty Function Index (last = Sum of all functions)')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_title('Correlation between Difficulty Function and Visit Distribution')
    ax2.set_xticks(range(1, len(difficulty_correlations) + 1))
    ax2.set_xticklabels([*range(1, len(difficulty_correlations)), 'Sum'])
    
    plt.tight_layout()
    
    # Print correlations
    print("Value Function Correlations:")
    for i, corr in enumerate(value_correlations, 1):
        if isinstance(corr, str) or np.isnan(corr):
            corr = 0
        if i < len(value_correlations):
            print(f"Function {i}: {corr:.4f}")
        else:
            print(f"Sum of all functions: {corr:.4f}")
    
    print("\nDifficulty Function Correlations:")
    for i, corr in enumerate(difficulty_correlations, 1):
        if isinstance(corr, str) or np.isnan(corr):
            corr = 0
        if i < len(difficulty_correlations):
            print(f"Function {i}: {corr:.4f}")
        else:
            print(f"Sum of all functions: {corr:.4f}")

    plt.show()

# Main execution
filename = f'{DATA_DIR}/training_data.json'
data = load_data(filename)

value_correlations, difficulty_correlations = calculate_correlations(data)

# Plot both correlations in the same figure
plot_correlations(value_correlations, difficulty_correlations)
