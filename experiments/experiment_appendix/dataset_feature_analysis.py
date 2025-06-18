import json
import math
from collections import defaultdict
from typing import List, Dict, Any
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os
from scipy.ndimage import gaussian_filter

from poolagent.path import DATA_DIR, VISUALISATIONS_DIR
from poolagent import LIMITS
LIMITS['theta'] = (0, 40)

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as f:
        return json.load(f)["train"]

def calculate_distance(pos1: List[float], pos2: List[float]) -> float:
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def generalize_event(event: str) -> str:
    parts = event.split('-')
    if len(parts) > 3 and parts[1] != 'ball':
        return '-'.join(parts[:3])
    return event

def analyze_events(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    event_stats = defaultdict(lambda: {'count': 0, 'total_ev': 0, 'total_entropy': 0})
    for entry in data:
        ev = entry['expected_value']
        entropy = entry['entropy']
        for event in entry['events']:
            event_name = generalize_event(event[0])
            event_stats[event_name]['count'] += 1
            event_stats[event_name]['total_ev'] += ev
            event_stats[event_name]['total_entropy'] += entropy
    
    for event in event_stats:
        count = event_stats[event]['count']
        event_stats[event]['avg_ev'] = event_stats[event]['total_ev'] / count
        event_stats[event]['avg_entropy'] = event_stats[event]['total_entropy'] / count
    
    return event_stats

def bin_distance(distance: float, bin_size: float = 0.1) -> float:
    return round(distance / bin_size) * bin_size

def analyze_state_features(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    feature_stats = {
        'num_balls': defaultdict(lambda: {'count': 0, 'total_ev': 0}),
        'num_target_balls_potted': defaultdict(lambda: {'count': 0, 'total_ev': 0}),
        'total_distance_to_pockets': defaultdict(lambda: {'count': 0, 'total_ev': 0})
    }
    
    pockets = [(0, 0), (0.5, 0), (1, 0), (0, 2), (0.5, 2), (1, 2)]
    target_balls = ['red', 'blue', 'yellow']
    
    for entry in data:
        state = entry['state']['positions']
        ev = entry['expected_value']
        
        num_balls = sum(1 for ball, pos in state.items() if pos != ["infinity", "infinity"])
        feature_stats['num_balls'][num_balls]['count'] += 1
        feature_stats['num_balls'][num_balls]['total_ev'] += ev
        
        num_target_balls_potted = sum(1 for ball in target_balls if state.get(ball) == ["infinity", "infinity"])
        feature_stats['num_target_balls_potted'][num_target_balls_potted]['count'] += 1
        feature_stats['num_target_balls_potted'][num_target_balls_potted]['total_ev'] += ev
        
        total_distance = 0
        for ball in target_balls:
            if ball in state and state[ball] != ["infinity", "infinity"]:
                ball_pos = state[ball]
                min_distance = min(calculate_distance(ball_pos, pocket) for pocket in pockets)
                total_distance += min_distance
        
        distance_bucket = bin_distance(total_distance)
        feature_stats['total_distance_to_pockets'][distance_bucket]['count'] += 1
        feature_stats['total_distance_to_pockets'][distance_bucket]['total_ev'] += ev
    
    for feature in feature_stats:
        for value in feature_stats[feature]:
            count = feature_stats[feature][value]['count']
            feature_stats[feature][value]['avg_ev'] = feature_stats[feature][value]['total_ev'] / count
    
    return feature_stats

def bin_action_value(key: str, value: float) -> float:
    if key == 'V0':
        return round(value * 2) / 2  # Bin V0 into 0.5 increments
    elif key in ['theta', 'phi']:
        return round(value / 5) * 5  # Bin angles into 5-degree increments
    elif key in ['a', 'b']:
        return round(value * 40) / 40  # Bin a and b into 0.025 increments
    else:
        return round(value, 3)  # Keep 3 decimal places for other values

def analyze_action_stats(data: List[Dict[str, Any]]) -> Dict[str, Dict[float, Dict[str, float]]]:
    action_stats = defaultdict(lambda: defaultdict(lambda: {'ev_sum': 0, 'count': 0}))
    for entry in data:
        action = entry['action']
        ev = entry['expected_value']
        for key, value in action.items():
            binned_value = bin_action_value(key, value)
            action_stats[key][binned_value]['ev_sum'] += ev
            action_stats[key][binned_value]['count'] += 1
    
    for action in action_stats:
        for value in action_stats[action]:
            count = action_stats[action][value]['count']
            action_stats[action][value]['avg_ev'] = action_stats[action][value]['ev_sum'] / count
    
    return action_stats

def print_sorted_data(data: List[tuple], headers: List[str], title: str, reverse: bool = True):
    print(f"\n{title}:")
    
    sort_key = 'avg_ev'
    sorted_data = sorted(data, key=lambda x: x[1].get(sort_key, 0), reverse=reverse)[:5]
    
    table = []
    for k, values in sorted_data:
        row = [k]
        for header in headers[1:]:
            header_key = header.lower().replace(' ', '_')
            if header_key in values:
                row.append(f"{values[header_key]:.3f}")
            else:
                row.append("N/A")
        table.append(row)
    
    print(tabulate(table, headers=headers, tablefmt="grid"))

def main(file_path: str):
    data = load_json_data(file_path)
    
    event_stats = analyze_events(data)
    state_features = analyze_state_features(data)
    action_stats = analyze_action_stats(data)
    
    print_sorted_data(event_stats.items(), ["Event", "Avg EV", "Avg Entropy", "Count"], "Events most correlated with high expected value")
    print_sorted_data(event_stats.items(), ["Event", "Avg EV", "Avg Entropy", "Count"], "Events most correlated with low expected value", reverse=False)
    
    print_sorted_data(event_stats.items(), ["Event", "Avg Entropy", "Avg EV", "Count"], "Events most correlated with high entropy")
    print_sorted_data(event_stats.items(), ["Event", "Avg Entropy", "Avg EV", "Count"], "Events most correlated with low entropy", reverse=False)
    
    for feature, stats in state_features.items():
        print_sorted_data(stats.items(), [feature.capitalize(), "Avg EV", "Count"], f"State feature '{feature}' correlated with high expected value")
    
    for action, values in action_stats.items():
        print_sorted_data([(str(k), v) for k, v in values.items()], 
                          [f"{action} Value", "Avg EV", "Count"], 
                          f"Action '{action}' values correlated with high expected value")
        
    # Plot action correlations
    plot_action_correlations(action_stats)
    
    # Plot circular heatmap for a and b values
    plot_circular_heatmap(data)
    
    print("\nPlots have been saved as 'action_correlations.png' and 'ab_circular_heatmap.png'")

def plot_action_correlations(action_stats: Dict[str, Dict[float, Dict[str, float]]]):
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Action Parameter Correlations with Expected Value', fontsize=16)
    
    for idx, (action, values) in enumerate(action_stats.items()):
        row = idx // 3
        col = idx % 3
        ax = axs[row, col]
        
        x = [float(k) for k in values.keys()]
        y = [v['avg_ev'] for v in values.values()]
        
        ax.scatter(x, y)
        ax.set_title(f'{action} vs Expected Value')
        ax.set_xlabel(action)
        ax.set_ylabel('Average Expected Value')
        
        # Set fixed scales based on LIMITS
        ax.set_xlim(LIMITS[action][0], LIMITS[action][1])
        
        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", alpha=0.8)
    
    # Remove the unused subplot
    fig.delaxes(axs[1, 2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALISATIONS_DIR, 'action_correlations.png'))
    plt.close()

def plot_circular_heatmap(data: List[Dict[str, Any]]):
    a_values = []
    b_values = []
    ev_values = []
    
    for entry in data:
        a_values.append(entry['action']['a'])
        b_values.append(entry['action']['b'])
        ev_values.append(entry['expected_value'])
    
    # Set the limits for a and b
    LIMITS = {'a': (-0.25, 0.25), 'b': (-0.25, 0.25)}
    
    # Increase the number of bins for smoother results
    num_bins = 200
    
    # Create a 2D histogram
    hist, xedges, yedges = np.histogram2d(a_values, b_values, bins=num_bins, weights=ev_values,
                                          range=[LIMITS['a'], LIMITS['b']])
    counts, _, _ = np.histogram2d(a_values, b_values, bins=num_bins,
                                  range=[LIMITS['a'], LIMITS['b']])
    
    # Avoid division by zero
    counts[counts == 0] = 1
    
    # Calculate average EV for each bin
    hist_avg = hist / counts
    
    # Apply Gaussian filter for smoothing
    hist_avg_smooth = gaussian_filter(hist_avg, sigma=0.8)
    
    # Create coordinate meshgrid
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)
    
    # Create custom colormap
    colors = ['blue', 'green', 'yellow', 'red']
    n_bins = 256  # Increase color resolution
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use pcolormesh for the heatmap
    c = ax.pcolormesh(X, Y, hist_avg_smooth, cmap=cmap, shading='gouraud')
    ax.set_title('2D Heatmap of a and b values vs Expected Value', fontsize=16)
    
    # Set axis labels
    ax.set_xlabel('a value', fontsize=12)
    ax.set_ylabel('b value', fontsize=12)
    
    # Set tick labels
    ax.set_xticks(np.linspace(-0.25, 0.25, 5))
    ax.set_yticks(np.linspace(-0.25, 0.25, 5))
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add colorbar
    cbar = fig.colorbar(c, ax=ax, orientation='vertical', label='Average Expected Value')
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_ylabel('Average Expected Value', fontsize=12)
    
    # Ensure the aspect ratio is equal
    ax.set_aspect('equal', 'box')
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALISATIONS_DIR, 'ab_2d_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main(f"{DATA_DIR}/poolmaster_training_data.json")