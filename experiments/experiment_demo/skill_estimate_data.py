import json
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poolagent.path import DATA_DIR

CHOOSE_FROM = -1
N = 50
TRAINING_DATA = "poolmaster_training_data.json"

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)['train']

    if CHOOSE_FROM > 0 and len(data) > CHOOSE_FROM:
        data = np.random.choice(data, CHOOSE_FROM, replace=False)

    return data

def flatten_shot(shot_data):
    flattened = []
    flattened.extend(shot_data['difficulties'])
    return flattened

def filter_shot(shot):
    events = [e[0] for e in shot['events']]
    target_balls = ['red', 'blue', 'yellow']
    
    target_ball_potted = any(any(e.startswith(f'ball-pocket-{color}') for e in events) for color in target_balls)
    non_target_ball_potted = any(e.startswith('ball-pocket-') and not any(e.startswith(f'ball-pocket-{color}') for color in target_balls) for e in events)
    
    return target_ball_potted and not non_target_ball_potted

def extract_shots(entries):
    all_shots = []
    for entry_idx, entry in enumerate(entries):
        shot_data = {
            'difficulties': entry['difficulties'],
        }

        individual_shot_entry = {
            "starting_state": entry["state"],
            'difficulties': entry['difficulties'],
            "params": entry["action"],
            "events": entry["events"],
        }

        if not filter_shot(individual_shot_entry):
            continue

        all_shots.append({
            'entry_idx': entry_idx,
            'data': shot_data,
            'entry': individual_shot_entry
        })
    return all_shots

def perform_kmeans(shots):
    flattened_shots = [flatten_shot(shot['data']) for shot in shots]
    scaler = StandardScaler()
    normalized_shots = scaler.fit_transform(flattened_shots)

    kmeans = KMeans(n_clusters=N, init='k-means++', n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_shots)
    
    return cluster_labels, normalized_shots

def calculate_pca(normalized_shots):
    pca = PCA(n_components=2)  # We'll use 2 components for visualization
    pca_result = pca.fit_transform(normalized_shots)
    return pca_result, pca.explained_variance_ratio_

def select_diverse_shots(shots, cluster_labels, normalized_shots):
    selected_indices = []
    for cluster in range(N):
        cluster_points = normalized_shots[cluster_labels == cluster]
        cluster_center = np.mean(cluster_points, axis=0)
        distances = np.sum((cluster_points - cluster_center) ** 2, axis=1)
        closest_point_idx = np.argmin(distances)
        selected_indices.append(np.where(cluster_labels == cluster)[0][closest_point_idx])
    
    return selected_indices, [shots[i] for i in selected_indices]

def visualize_clusters_pca(pca_result, cluster_labels, selected_indices, selected_shots):
    plt.figure(figsize=(12, 10))
    
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, alpha=0.5, cmap='viridis')
    
    selected_pca = pca_result[selected_indices]
    plt.scatter(selected_pca[:, 0], selected_pca[:, 1], color='red', s=100, edgecolors='black', linewidth=2, label='Selected Shots')

    plt.title('PCA Visualization of KMeans Clusters')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.colorbar(scatter, label='Cluster')

    for i, (x, y) in enumerate(selected_pca):
        plt.annotate(f"({selected_shots[i]['entry_idx']})", 
                     (x, y), 
                     xytext=(5, 5), 
                     textcoords='offset points',
                     fontsize=8,
                     alpha=0.8)

    plt.tight_layout()
    plt.show()

def main(json_file_path, output_file_path):
    entries = load_json(json_file_path)
    all_shots = extract_shots(entries)

    cluster_labels, normalized_shots = perform_kmeans(all_shots)
    pca_result, explained_variance_ratio = calculate_pca(normalized_shots)
    
    selected_indices, selected_shots = select_diverse_shots(all_shots, cluster_labels, normalized_shots)
    
    visualize_clusters_pca(pca_result, cluster_labels, selected_indices, selected_shots)

    data = {
        f"shot_{idx}": v for idx, v in  enumerate([shot['entry'] for shot in selected_shots])
    }

    with open(output_file_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Selected {len(selected_shots)} diverse entries. Results saved to {output_file_path}")
    print(f"Explained variance ratio of first two PCA components: {explained_variance_ratio}")

if __name__ == "__main__":
    json_file_path = f"{DATA_DIR}/{TRAINING_DATA}"
    output_file_path = f"{DATA_DIR}/shot_task_dataset1.json"
    main(json_file_path, output_file_path)