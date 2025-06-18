import os
import json
from glob import glob
from pathlib import Path
from datetime import datetime

def merge_results(existing_data, new_data):
    """
    Merge two result dictionaries by combining their games lists and recalculating winrate
    """
    if not existing_data or not new_data:
        return new_data
    
    # Combine games lists
    merged_games = existing_data['games'] + new_data['games']
    
    # Recalculate winrate for all games
    merged_winrate = sum(merged_games) / len(merged_games)
    
    return {
        'games': merged_games,
        'winrate': merged_winrate
    }

def combine_results():
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Dictionary to store all results
    all_results = {}
    
    # Find all results.json files in the logs directory
    pattern = 'full_logs/*/tasks/*/results.json'
    result_files = glob(pattern)
    
    for result_file in result_files:
        # Parse the path to get the task name
        path_parts = Path(result_file).parts
        task_name = path_parts[path_parts.index('tasks') + 1]
        
        try:
            # Read the results.json file
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            # If task_name already exists, overwrite
            all_results[task_name] = results
                
            print(f"Processed {result_file}")
            
        except json.JSONDecodeError as e:
            print(f"Error reading {result_file}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {result_file}: {e}")
    
    # Write combined results to all_results.json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'full_results/{timestamp}.json'
    try:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSuccessfully created {output_file}")
        
        # # Print summary
        # print("\nSummary:")
        # for task, data in all_results.items():
        #     print(f"{task}: {len(data['games'])} games, winrate: {data['winrate']:.3f}")
            
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

if __name__ == "__main__":
    combine_results()