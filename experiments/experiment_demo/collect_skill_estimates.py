import json
import glob
import os
from typing import Dict, List, Any
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poolagent.path import DATA_DIR, ROOT_DIR

def add_model_data(all_results: Dict[str, Any], task_id: str, model_id: str, model_data: Dict[str, Any]) -> Dict[str, Any]:

    current_model_data = all_results[task_id]['LanguageAgent_language']['models'][model_id] 

    current_model_data['attempts'] += model_data['attempts']

    all_results[task_id]['LanguageAgent_language']['models'][model_id] = current_model_data

    return all_results

def load_results_files(base_path: str) -> Dict[str, Any]:
    """
    Load and combine all results files from the specified path pattern.
    
    Args:
        base_path: Base directory containing the logs
    
    Returns:
        Dictionary mapping task_id to their full result entries
    """
    all_results = {}
    
    # Find all json files matching the pattern
    pattern = os.path.join(base_path, "logs", "*", "all_results.json")
    result_files = glob.glob(pattern)
    
    if not result_files:
        raise FileNotFoundError(f"No result files found matching pattern: {pattern}")
    
    # Load and combine all results
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for task_id, task_data in data.items():
                        if task_id in all_results:
                            print(f"Warning: Duplicate task_id found in {file_path}: {task_id}")
                            for model_id, model_data in task_data['LanguageAgent_language']['models'].items():
                                if model_id in all_results[task_id]['LanguageAgent_language']['models']:
                                    print(f"Warning: Duplicate model_id found in {file_path}: {model_id}")
                                    all_results = add_model_data(all_results, task_id, model_id, model_data)
                                else:
                                    all_results[task_id]['LanguageAgent_language']['models'][model_id] = model_data
                        else:
                            all_results[task_id] = task_data
                else:
                    print(f"Warning: Unexpected format in {file_path} - expected dict")
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse JSON from {file_path}")
        except Exception as e:
            print(f"Warning: Error processing {file_path}: {str(e)}")
    
    return all_results

def load_and_annotate_dataset(data_dir: str, results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Load and annotate the skill estimates dataset with language agent data.
    
    Args:
        data_dir: Directory containing skill_estimate_dataset.json
        results: Dictionary of results from load_results_files
    
    Returns:
        Annotated dataset list
    """
    dataset_path = os.path.join(data_dir, "skill_estimate_dataset.json")
    
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
            
        if not isinstance(dataset, list):
            raise ValueError("skill_estimate_dataset.json should contain a list")
        
        # Annotate each entry with corresponding language agent data
        for idx, entry in enumerate(dataset):
            task_id = f'task_shot_{idx}'
            if task_id in results:
                dataset[idx]['LanguageAgent_language'] = results[task_id].get('LanguageAgent_language')
            else:
                print(f"Warning: No matching result found for task_id: {task_id}")
                
        return dataset
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in dataset file: {dataset_path}")

def save_annotated_dataset(dataset: List[Dict[str, Any]], output_path: str):
    """
    Save the annotated dataset to a JSON file.
    
    Args:
        dataset: Annotated dataset to save
        output_path: Path where to save the output file
    """
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

def main(base_path: str, output_path: str):
    """
    Main function to process and combine the JSON files.
    
    Args:
        base_path: Base directory containing both logs and skill_estimate_dataset.json
        output_path: Path where to save the annotated dataset
    """
    try:
        # Load all results files
        print("Loading results files...")
        results = load_results_files(base_path)
        print(f"Found {len(results)} result entries")
        
        # Load and annotate the dataset
        print("Loading and annotating dataset...")
        annotated_dataset = load_and_annotate_dataset(DATA_DIR, results)
        print(f"Processed {len(annotated_dataset)} dataset entries")
        
        # Save the annotated dataset
        print(f"Saving annotated dataset to {output_path}...")
        save_annotated_dataset(annotated_dataset, output_path)
        print("Processing complete!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    BASE_PATH = ROOT_DIR + "/experiments/experiment_demo"
    OUTPUT_PATH = DATA_DIR + "/skill_estimate_results.json"
    main(BASE_PATH, OUTPUT_PATH)