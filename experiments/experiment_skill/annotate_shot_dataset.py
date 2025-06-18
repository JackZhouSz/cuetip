import json
import os
import sys
from typing import Dict

# Add the parent directory to the path to import poolagent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poolagent.pool import PoolGame
from poolagent.utils import State, SKILL_LEVELS
from poolagent.value_data.gen_mcts_data import propose_shot
from poolagent.path import DATA_DIR

ROLL_OUTS = 1000

def estimate_shot_values(shots_data: Dict) -> Dict:
    """
    Estimates the value of the starting state for each shot in the dataset.
    
    Args:
        shots_data: Dictionary containing shot data entries
        
    Returns:
        Updated dictionary with value estimates added
    """
    # Initialize the pool game environment
    env = PoolGame(visualizable=False)
    
    # Process each shot
    for shot_id, shot_data in shots_data.items():
        print(f"Processing shot {shot_id}...")
        
        # Load the starting state
        state = State().from_json(shot_data['starting_state'])
        
        # Reset environment and set to starting state
        env.reset()
        env.from_state(state)
        
        # Estimate the value using the same method as in the original code
        value_estimate = env.get_value_estimate(
            lambda g: propose_shot(g, eps=0, skill_level=SKILL_LEVELS.BASELINE),
            initial_roll_outs=ROLL_OUTS
        )
        
        # Add the value estimate to the shot data
        shot_data['value_estimate'] = float(value_estimate)
        
        print(f"Value estimate for shot {shot_id}: {value_estimate}")
    
    return shots_data

def main():
    # Load the shot task dataset
    dataset_path = os.path.join(DATA_DIR, "shot_task_dataset.json")
    print(f"Loading dataset from {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        shots_data = json.load(f)
    
    # Process the shots and add value estimates
    updated_shots = estimate_shot_values(shots_data)
    
    # Save the updated dataset
    output_path = os.path.join(DATA_DIR, "shot_task_dataset.json")
    print(f"Saving updated dataset to {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(updated_shots, f, indent=4)
    
    print("Done!")

if __name__ == "__main__":
    main()