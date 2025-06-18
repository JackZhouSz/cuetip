import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_csv_file(filepath):
    df = pd.read_csv(filepath)
    # Calculate std dev assuming min/max represent 95% confidence interval
    # In a normal distribution, 95% CI is approximately ±1.96 std devs from mean
    std = (df['Grouped runs - test_loss__MAX'] - df['Grouped runs - test_loss__MIN']) / (2 * 1.96)
    
    return {
        'steps': df['Step'].astype(int),
        'mean': df['Grouped runs - test_loss'],
        'std': std
    }

def plot_with_error_bars(model_data):
    plt.figure(figsize=(12, 8))
    
    # Use a more muted color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for (label, filepath), color in zip(model_data, colors):
        data = read_csv_file(filepath)
        
        # Plot the mean line
        plt.plot(data['steps'], data['mean'], label=label, color=color, linewidth=2)
        
        # Add ±1 standard deviation band
        plt.fill_between(data['steps'],
                        data['mean'] - data['std'],
                        data['mean'] + data['std'],
                        color=color,
                        alpha=0.15)  # More transparent fill

    # Customize the plot
    plt.xlabel('Steps', fontsize=12, labelpad=10)
    plt.ylabel('Test Loss', fontsize=12, labelpad=10)
    plt.title('Test Loss Over Time (±1σ)', fontsize=14, pad=20)
    
    # Add subtle grid only on the y-axis
    plt.grid(True, axis='y', alpha=0.2, linestyle='--')
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Customize legend
    plt.legend(frameon=True, 
              fancybox=True, 
              framealpha=0.95,
              edgecolor='none',
              bbox_to_anchor=(1.05, 1), 
              loc='upper left')
    
    # Adjust layout and margins
    plt.tight_layout()
    
    return plt.gcf()

model_data = [
    ('Attention Combined', 'distribution_model_attention_combined.csv'),
    ('Attention Separate', 'distribution_model_attention_separate.csv'),
    ('MLP Combined', 'distribution_model_mlp_combined.csv'),
    ('MLP Separate', 'distribution_model_mlp_separate.csv')
]

# Create and show the plot
fig = plot_with_error_bars(model_data)
plt.show()