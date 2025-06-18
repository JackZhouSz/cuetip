import os
import json
from glob import glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def get_ball_differential(game_file):
    try:
        with open(game_file, 'r') as f:
            game_data = json.load(f)
        
        if not game_data:
            return 0, 0
        
        game_length = len(game_data)
            
        final_shot_key = list(game_data.keys())[-1]
        final_shot = game_data[final_shot_key]
        positions = final_shot['end_state']['positions']
        
        p1_balls = sum(1 for color in ['red', 'blue', 'yellow'] if color in positions)
        p2_balls = sum(1 for color in ['green', 'black', 'pink'] if color in positions)
        
        return p2_balls - p1_balls, game_length
    except Exception as e:
        print(f"Error processing game file {game_file}: {e}")
        return 0, 0

def merge_results(existing_data, new_data):
    if not existing_data or not new_data:
        return new_data
    
    merged_games = existing_data['games'] + new_data['games']
    merged_differentials = existing_data.get('ball_differentials', []) + new_data.get('ball_differentials', [])
    merged_winning_diffs = existing_data.get('winning_differentials', []) + new_data.get('winning_differentials', [])
    merged_losing_diffs = existing_data.get('losing_differentials', []) + new_data.get('losing_differentials', [])
    merged_game_lengths = existing_data.get('game_lengths', []) + new_data.get('game_lengths', [])
    
    return {
        'games': merged_games,
        'game_lengths': merged_game_lengths,
        'avg_game_length': sum(merged_game_lengths) / len(merged_game_lengths) if merged_game_lengths else 0,
        'winrate': sum(merged_games) / len(merged_games),
        'ball_differentials': merged_differentials,
        'avg_ball_differential': sum(merged_differentials) / len(merged_differentials) if merged_differentials else 0,
        'winning_differentials': merged_winning_diffs,
        'losing_differentials': merged_losing_diffs,
        'avg_ball_differential_winning': sum(merged_winning_diffs) / len(merged_winning_diffs) if merged_winning_diffs else 0,
        'avg_ball_differential_losing': sum(merged_losing_diffs) / len(merged_losing_diffs) if merged_losing_diffs else 0
    }

def process_noise_level(noise_level_path, all_results):    
    pattern = f'{noise_level_path}/tasks/*/results.json'
    result_files = glob(pattern)
    
    for result_file in result_files:
        path_parts = Path(result_file).parts
        task_name = path_parts[path_parts.index('tasks') + 1]
        task_dir = str(Path(result_file).parent)
        
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            game_files = glob(f"{task_dir}/game_*.json")
            ball_differentials = []
            winning_differentials = []
            losing_differentials = []
            game_lengths = []
            
            for i, game_file in enumerate(game_files):
                diff, game_length = get_ball_differential(game_file)
                game_lengths.append(game_length)
                ball_differentials.append(diff)
                if diff >= 0:  # P1 win
                    winning_differentials.append(diff)
                else:  # P1 loss
                    losing_differentials.append(diff)

                if results['games'][i] == 1 and diff < 0:
                    print(f"Game {game_file} has a negative differential: {diff} and P1 win")
                elif results['games'][i] == 0 and diff > 0:
                    print(f"Game {game_file} has a positive differential: {diff} and P1 loss")
            
            results.update({
                'game_lengths': game_lengths,
                'avg_game_length': sum(game_lengths) / len(game_lengths) if game_lengths else 0,
                'ball_differentials': ball_differentials,
                'avg_ball_differential': sum(ball_differentials) / len(ball_differentials) if ball_differentials else 0,
                'winning_differentials': winning_differentials,
                'losing_differentials': losing_differentials,
                'avg_ball_differential_winning': sum(winning_differentials) / len(winning_differentials) if winning_differentials else 0,
                'avg_ball_differential_losing': sum(losing_differentials) / len(losing_differentials) if losing_differentials else 0
            })

            results['games'] = []
            for diff in ball_differentials:
                if diff > 0:
                    results['games'].append(1)
                elif diff == 0:
                    results['games'].append(0.5)
                else:
                    results['games'].append(0)
            results['winrate'] = sum(results['games']) / len(results['games'])
             
            if task_name in all_results:
                all_results[task_name] = merge_results(all_results[task_name], results)
            else:
                all_results[task_name] = results
                            
        except json.JSONDecodeError as e:
            print(f"Error reading {result_file}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {result_file}: {e}")
    
    return all_results

def combine_results():
    os.makedirs('noise_results', exist_ok=True)
    
    # Define noise levels
    noise_levels = ['no_noise', 'pro', 'amateur', 'novice']
    combined_results = defaultdict(dict)
    
    # Process each noise level
    paths = glob('noise_logs/*')
    for path in paths:
        for noise_level in noise_levels:
            noise_path = f'{path}/{noise_level}'
            print(f"\nProcessing {noise_path}...")
            if os.path.exists(noise_path):
                print(f"Found {noise_level} data at {noise_path}")
                combined_results[noise_level] = process_noise_level(noise_path, combined_results[noise_level])
    
    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'noise_results/{timestamp}.json'
    try:
        with open(output_file, 'w') as f:
            json.dump(combined_results, f, indent=2)
        print(f"\nSuccessfully created {output_file}")
        
        print("\nSummary:")
        for noise_level, noise_data in combined_results.items():
            print(f"\n{noise_level.upper()}:")
            for task, data in noise_data.items():
                print(f"\n{task}:")
                print(f"  Games: {len(data['games'])}, Winrate: {data['winrate']:.3f}")
                print(f"  Avg game length: {data.get('avg_game_length', 0):.2f}")
                print(f"  Avg ball diff: {data.get('avg_ball_differential', 0):.2f}")
                print(f"  Avg ball diff (wins): {data.get('avg_ball_differential_winning', 0):.2f}")
                print(f"  Avg ball diff (losses): {data.get('avg_ball_differential_losing', 0):.2f}")
            
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

if __name__ == "__main__":
    combine_results()