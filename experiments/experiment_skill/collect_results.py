import os
import json
from glob import glob
from pathlib import Path
from datetime import datetime
from statistics import mean

# Minimum number of attempts required to consider the results valid
MIN_ATTEMPTS_REQUIRED = 5
MAX_TASKS = 50

def merge_results(existing_data, new_data):
    """
    Merge two result dictionaries by combining their attempts and recalculating statistics
    """
    if not existing_data or not new_data:
        return new_data
    
    # Combine attempts lists
    merged_attempts = existing_data['attempts'] + new_data['attempts']
    
    # Recalculate statistics
    values = [attempt[0] for attempt in merged_attempts]
    successes = [attempt[1] for attempt in merged_attempts]
    
    return {
        'attempts': merged_attempts,
        'mean_val': mean(values),
        'success_rate': sum(successes) / len(successes)
    }

def is_valid_result(results):
    """
    Validate that results meet our requirements:
    - Has 'attempts' list
    - Has minimum number of attempts
    - Each attempt is properly formatted [value, boolean]
    """
    if not isinstance(results.get('attempts'), list):
        return False
        
    if len(results['attempts']) < MIN_ATTEMPTS_REQUIRED:
        return False
        
    # Check format of each attempt
    for attempt in results['attempts']:
        if not isinstance(attempt, list) or len(attempt) != 2:
            return False
        if not isinstance(attempt[0], (int, float)) or not isinstance(attempt[1], bool):
            return False
            
    return True

def combine_results():
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Dictionary to store all results
    all_results = {}
    
    # Find all results.json files in the logs directory
    pattern = 'logs/*/tasks/*/results.json'
    result_files = glob(pattern)
    
    for result_file in result_files:
        # Parse the path to get the task name
        path_parts = Path(result_file).parts
        task_name = path_parts[path_parts.index('tasks') + 1]

        if int(task_name.split('_')[-1]) + 1 > MAX_TASKS:
            continue
        
        try:
            # Read the results.json file
            with open(result_file, 'r') as f:
                results = json.load(f)
                
            # Validate the data structure and minimum attempts
            if not is_valid_result(results):
                print(f"Warning: Skipping {result_file} - invalid data structure or fewer than {MIN_ATTEMPTS_REQUIRED} attempts")
                continue
                
            # If task_name already exists, merge the results
            if task_name in all_results:
                all_results[task_name] = merge_results(all_results[task_name], results)
            else:
                all_results[task_name] = results
                
            print(f"Processed {result_file} ({len(results['attempts'])} attempts)")
            
        except json.JSONDecodeError as e:
            print(f"Error reading {result_file}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {result_file}: {e}")
    
    # Write combined results to all_results.json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'results/{timestamp}.json'
    
    if not all_results:
        print("\nNo valid results files found. No output file created.")
        return
        
    try:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSuccessfully created {output_file}")
        
        # Print summary
        print("\nSummary:")
        for task, data in all_results.items():
            print(f"{task}:")
            print(f"  Number of attempts: {len(data['attempts'])}")
            print(f"  Mean value: {data['mean_val']:.3f}")
            print(f"  Success rate: {data['success_rate']:.3f}")
            
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

if __name__ == "__main__":
    combine_results()