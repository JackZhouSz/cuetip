import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

from poolagent.path import ROOT_DIR

def collect_game_data(root_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Collect game data from JSON files with specific LLM-related entries.
    
    Args:
        root_dir: Root directory containing experiment folders
        
    Returns:
        Dictionary with agent names as keys and lists of their LLM interactions as values
    """
    # Store collected data by agent
    agent_data = defaultdict(list)
    
    # Walk through the directory structure
    root_path = Path(root_dir)
    for timestamp_dir in root_path.glob("experiment_one/logs/*/"):
        for matchup_dir in timestamp_dir.glob("*/"):  # matchups or tasks
            if matchup_dir.name not in ['matchups', 'tasks']:
                continue
                
            for agent_matchup_dir in matchup_dir.glob("*---*"):
                # Split agent names from directory name
                agent1, agent2 = agent_matchup_dir.name.split('---')
                
                # Process each game JSON file
                for game_file in agent_matchup_dir.glob("game_*.json"):
                    try:
                        with open(game_file, 'r') as f:
                            game_data = json.load(f)
                            
                        # Process each shot in the game
                        for shot_key, shot_data in game_data.items():
                            if not shot_key.startswith('shot_'):
                                continue
                                
                            # Get basic shot information
                            player = shot_data.get('player')
                            if not player:
                                continue
                                
                            # Determine current agent
                            current_agent = agent1 if player == 'one' else agent2
                            agent_name = current_agent.split('_')[0]
                            llm_name = current_agent.split('_')[1]
                            
                            # Only collect shots with LM chooser or suggester
                            if 'lm_chooser' in shot_data or 'lm_suggester' in shot_data:
                                entry = {
                                    'game_file': str(game_file),
                                    'shot': shot_key,
                                    'agent': agent_name,
                                    'llm': llm_name,
                                    'start_state': shot_data.get('start_state'),
                                    'params': shot_data.get('params'),
                                    'lm_chooser': shot_data.get('lm_chooser'),
                                    'lm_suggester': shot_data.get('lm_suggester')
                                }
                                agent_data[current_agent].append(entry)
                                
                    except json.JSONDecodeError as e:
                        print(f"Error reading {game_file}: {e}")
                    except Exception as e:
                        print(f"Unexpected error processing {game_file}: {e} at line {e.__traceback__.tb_lineno}")
    
    return dict(agent_data)

def save_results(data: Dict[str, List[Dict[str, Any]]], output_dir: str):
    """
    Save collected data to JSON files, one per agent.
    
    Args:
        data: Collected agent data
        output_dir: Directory to save the output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for agent_name, agent_data in data.items():
        # Create a safe filename from agent name
        safe_name = "".join(c if c.isalnum() else "_" for c in agent_name)
        output_file = os.path.join(output_dir, f"{safe_name}_data.json")
        
        with open(output_file, 'w') as f:
            json.dump({
                'agent_name': agent_name,
                'total_entries': len(agent_data),
                'entries': agent_data
            }, f, indent=2)
        
        print(f"Saved {len(agent_data)} entries for agent {agent_name} to {output_file}")

def main():
    # Configure these paths as needed
    root_directory = ROOT_DIR
    output_directory = ROOT_DIR + "/experiment_appendix/reasoning_data"
    
    # Collect data
    print("Collecting game data...")
    agent_data = collect_game_data(root_directory)
    
    # Print summary
    print("\nData collection summary:")
    for agent, data in agent_data.items():
        print(f"Agent {agent}: {len(data)} entries")
    
    # Save results
    print("\nSaving results...")
    save_results(agent_data, output_directory)
    print("Done!")

if __name__ == "__main__":
    main()