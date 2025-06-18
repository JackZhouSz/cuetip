import json
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from collections import defaultdict
import os
from pathlib import Path

from poolagent.path import ROOT_DIR

def load_reasoning_from_json(data):
    """
    Extract reasoning text from a loaded JSON structure, separating chooser and suggester
    
    Parameters:
    data (dict): Loaded JSON data
    
    Returns:
    dict: Dictionary with 'chooser' and 'suggester' lists of reasoning entries
    """
    reasoning_entries = {
        'chooser': [],
        'suggester': []
    }
    
    if 'entries' not in data:
        print(f"Warning: No entries found in data for agent {data.get('agent_name')}")
        return reasoning_entries
    
    for entry in data['entries']:
        # Extract chooser reasoning
        if 'lm_chooser' in entry and 'response' in entry['lm_chooser']:
            reasoning = entry['lm_chooser']['response']['reasoning']
            metadata = {
                'game_file': entry['game_file'],
                'shot': entry['shot'],
                'agent': entry['agent'],
                'llm': entry['llm'],
                'chosen_shot': entry['lm_chooser']['chosen_shot'],
                'component': 'chooser'
            }
            reasoning_entries['chooser'].append({
                'reasoning': reasoning,
                'metadata': metadata
            })
        
        # Extract suggester reasoning
        if 'lm_suggester' in entry and 'response' in entry['lm_suggester']:
            reasoning = entry['lm_suggester']['response']['reasoning']
            metadata = {
                'game_file': entry['game_file'],
                'shot': entry['shot'],
                'agent': entry['agent'],
                'llm': entry['llm'],
                'suggested_shots': entry['lm_suggester']['response'].get('shots', ''),
                'component': 'suggester'
            }
            reasoning_entries['suggester'].append({
                'reasoning': reasoning,
                'metadata': metadata
            })
    
    return reasoning_entries

def load_all_json_files(directory):
    """
    Load all JSON files from a directory and organize by agent name and component
    
    Parameters:
    directory (str): Path to directory containing JSON files
    
    Returns:
    dict: Dictionary of agent_name -> {'chooser': [...], 'suggester': [...]}
    """
    agent_data = defaultdict(lambda: {'chooser': [], 'suggester': []})
    
    # Iterate through all JSON files in directory
    for file_path in Path(directory).glob('*.json'):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            agent_name = data.get('agent_name')
            if not agent_name:
                print(f"Warning: No agent_name found in {file_path}")
                continue
                
            entries = load_reasoning_from_json(data)
            agent_data[agent_name]['chooser'].extend(entries['chooser'])
            agent_data[agent_name]['suggester'].extend(entries['suggester'])
            
            print(f"Loaded {len(entries['chooser'])} chooser and {len(entries['suggester'])} suggester entries from {file_path} for agent {agent_name}")
            
        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON file {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    return agent_data

def analyze_all_similarities(agent_data, method='embeddings', n_examples=3):
    """
    Analyze similarities for each agent's reasoning, separated by component
    
    Parameters:
    agent_data (dict): Dictionary of agent_name -> {'chooser': [...], 'suggester': [...]}
    method (str): 'embeddings' or 'tfidf' for similarity calculation
    n_examples (int): Number of distinct examples to return per agent/component
    
    Returns:
    dict: Analysis results for each agent and component
    """
    results = {}
    
    for agent_name, components in agent_data.items():
        results[agent_name] = {}
        
        for component in ['chooser', 'suggester']:
            entries = components[component]
            print(f"\nAnalyzing {len(entries)} {component} entries for agent {agent_name}")
            
            if not entries:
                print(f"Warning: No {component} entries to analyze for agent {agent_name}")
                continue
                
            # Use the existing analyze_agent_similarities function for each component
            component_results = analyze_agent_similarities({agent_name: entries}, method, n_examples)
            results[agent_name][component] = component_results[agent_name]
    
    return results


def analyze_agent_similarities(agent_data, method='embeddings', n_examples=3):
    """
    Analyze similarities for each agent's reasoning
    
    Parameters:
    agent_data (dict): Dictionary of agent_name -> list of reasoning entries
    method (str): 'embeddings' or 'tfidf' for similarity calculation
    n_examples (int): Number of distinct examples to return per agent
    
    Returns:
    dict: Analysis results for each agent
    """
    results = {}
    
    for agent_name, entries in agent_data.items():
        print(f"\nAnalyzing {len(entries)} entries for agent {agent_name}")
        
        if not entries:
            print(f"Warning: No entries to analyze for agent {agent_name}")
            continue
            
        paragraphs = [entry['reasoning'] for entry in entries]
        
        # Calculate similarities
        if method == 'embeddings':
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(paragraphs)
            similarity_matrix = cosine_similarity(embeddings)
        else:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(paragraphs)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Convert similarities to distances and ensure non-negative values
        distances = np.clip(1 - similarity_matrix, 0, 2)  # Clip to valid range [0, 2]
        
        # Adjust DBSCAN parameters based on the data
        # Calculate mean distance to determine eps
        mean_dist = np.mean(distances)
        eps = mean_dist * 0.5  # Use half of mean distance as eps
        
        # Cluster similar paragraphs
        clustering = DBSCAN(
            eps=eps,
            min_samples=max(2, len(paragraphs) // 20),  # Adaptive min_samples
            metric='precomputed'
        )
        
        try:
            clusters = clustering.fit_predict(distances)
        except Exception as e:
            print(f"Clustering failed for agent {agent_name}: {str(e)}")
            print("Falling back to larger eps value...")
            clustering = DBSCAN(
                eps=mean_dist,  # Use mean distance as eps
                min_samples=2,
                metric='precomputed'
            )
            clusters = clustering.fit_predict(distances)
        
        # Organize paragraphs by cluster
        cluster_dict = defaultdict(list)
        for idx, cluster_id in enumerate(clusters):
            cluster_dict[cluster_id].append(idx)
        
        # Find distinct examples
        distinct_examples = []
        sorted_clusters = sorted(
            [(k, v) for k, v in cluster_dict.items() if k != -1],
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        # Get central examples from top clusters
        for cluster_id, cluster_indices in sorted_clusters[:n_examples]:
            cluster_similarities = similarity_matrix[cluster_indices][:, cluster_indices]
            centrality_scores = np.mean(cluster_similarities, axis=1)
            central_idx = cluster_indices[np.argmax(centrality_scores)]
            
            distinct_examples.append({
                'paragraph': paragraphs[central_idx],
                'metadata': entries[central_idx]['metadata'],
                'cluster_size': len(cluster_indices),
                'cluster_id': cluster_id,
                'average_similarity': float(np.mean(cluster_similarities))  # Add average similarity
            })
        
        results[agent_name] = {
            'similarity_matrix': similarity_matrix,
            'clusters': clusters.tolist(),
            'distinct_examples': distinct_examples,
            'cluster_sizes': {k: len(v) for k, v in cluster_dict.items()},
            'total_entries': len(entries),
            'clustering_params': {
                'eps': eps,
                'min_samples': clustering.min_samples,
                'mean_distance': float(mean_dist)
            }
        }
    
    return results

def print_analysis_results(results):
    """Pretty print the analysis results for all agents and components"""
    for agent_name, components in results.items():
        print(f"\n\n=== Analysis for {agent_name} ===")
        
        for component in ['chooser', 'suggester']:
            if component in components:
                component_results = components[component]
                print(f"\n--- {component.title()} Component ---")
                print(f"Total entries analyzed: {component_results['total_entries']}")
                print(f"Number of clusters: {len(component_results['cluster_sizes']) - 1}")  # Excluding noise
                
                # Print clustering parameters
                params = component_results['clustering_params']
                print(f"\nClustering parameters:")
                print(f"  eps: {params['eps']:.3f}")
                print(f"  min_samples: {params['min_samples']}")
                print(f"  mean distance: {params['mean_distance']:.3f}")
                
                print("\nCluster sizes:")
                for cluster_id, size in component_results['cluster_sizes'].items():
                    label = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
                    print(f"{label}: {size} paragraphs")
                
                print("\nDistinct Examples:")
                for i, example in enumerate(component_results['distinct_examples'], 1):
                    print(f"\nExample {i} (from cluster {example['cluster_id']}, "
                          f"cluster size: {example['cluster_size']}, "
                          f"avg similarity: {example['average_similarity']:.3f}):")
                    print("Metadata:")
                    for key, value in example['metadata'].items():
                        print(f"  {key}: {value}")
                    print(f"Text: {example['paragraph'][:200]}...")

def convert_to_serializable(obj):
    """
    Convert numpy types and other non-serializable objects to Python native types
    
    Parameters:
    obj: Any object that needs to be converted
    
    Returns:
    Object with only JSON-serializable types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj

def save_analysis_results(results, output_dir=f"{ROOT_DIR}/experiments/experiment_appendix/reasoning_data"):
    """
    Save analysis results to a JSON file with nice formatting
    
    Parameters:
    results (dict): Analysis results for all agents and components
    output_dir (str): Directory to save the results
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare results for JSON serialization
    serializable_results = {}
    
    for agent_name, components in results.items():
        serializable_results[agent_name] = {}
        
        for component, component_results in components.items():
            # Convert numpy arrays and other non-serializable types
            clean_results = {
                'total_entries': component_results['total_entries'],
                'cluster_sizes': convert_to_serializable(component_results['cluster_sizes']),
                'clustering_params': {
                    'eps': float(component_results['clustering_params']['eps']),
                    'min_samples': int(component_results['clustering_params']['min_samples']),
                    'mean_distance': float(component_results['clustering_params']['mean_distance'])
                },
                'distinct_examples': []
            }
            
            # Clean up distinct examples
            for example in component_results['distinct_examples']:
                clean_example = {
                    'paragraph': example['paragraph'],
                    'metadata': convert_to_serializable(example['metadata']),
                    'cluster_size': int(example['cluster_size']),
                    'cluster_id': int(example['cluster_id']),
                    'average_similarity': float(example['average_similarity'])
                }
                clean_results['distinct_examples'].append(clean_example)
            
            serializable_results[agent_name][component] = clean_results
    
    # Add analysis metadata
    output_data = {
        'analysis_metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_agents': len(results),
            'components_analyzed': ['chooser', 'suggester']
        },
        'results': serializable_results
    }
    
    # Save to file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'reasoning_analysis_{timestamp}.json'
    
    # Convert any remaining non-serializable types
    final_output = convert_to_serializable(output_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    return output_file

def main():
    """Main function to run the analysis"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Analyze similarity between reasoning paragraphs across multiple agents')
    parser.add_argument('--directory', default=f'{ROOT_DIR}/experiments/experiment_appendix/reasoning_data/',
                      help='Path to the directory containing JSON files')
    parser.add_argument('--method', choices=['embeddings', 'tfidf'], default='embeddings',
                      help='Method to use for similarity calculation')
    parser.add_argument('--n-examples', type=int, default=3,
                      help='Number of distinct examples to show per agent/component')
    parser.add_argument('--output-dir', default=f'{ROOT_DIR}/experiments/experiment_appendix/reasoning_data/',
                      help='Directory to save the analysis results')
    
    args = parser.parse_args()
    
    print(f"Analyzing files in directory: {args.directory}")
    print(f"Using method: {args.method}")
    
    # Load all JSON files and organize by agent and component
    agent_data = load_all_json_files(args.directory)
    print(f"\nFound {len(agent_data)} different agents")
    
    # Analyze similarities for each agent and component
    results = analyze_all_similarities(
        agent_data,
        method=args.method,
        n_examples=args.n_examples
    )
    
    # Print results
    print_analysis_results(results)
    
    # Save results to JSON file
    output_file = save_analysis_results(results, args.output_dir)

if __name__ == '__main__':
    main()