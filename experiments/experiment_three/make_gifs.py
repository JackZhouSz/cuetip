import json
from pathlib import Path

import sys
import os
from turtle import st
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poolagent.path import DATA_DIR
from poolagent.pool import Pool
from poolagent.utils import State

RENAME_IDS = {
    'red': '1',
    'blue': '2',
    'yellow': '3',
    'green': '4',
    'black': '5',
    'pink': '6'
}

def main():
    # Check if shot index is provided
    if len(sys.argv) != 2:
        print("Usage: python make_gifs.py <shot_index>")
        sys.exit(1)
    
    shot_index = int(sys.argv[1])
    
    # Create output directory if it doesn't exist
    output_dir = Path("gifs")
    output_dir.mkdir(exist_ok=True)
    
    # Load the dataset
    with open(f"{DATA_DIR}/shot_task_dataset.json", "r") as f:
        dataset = json.load(f)
    
    # Initialize pool environment
    env = Pool()
    
    # Get the specific shot data
    items = list(dataset.items())
    if shot_index >= len(items):
        print(f"Error: Shot index {shot_index} is out of range. Dataset has {len(items)} shots.")
        sys.exit(1)
        
    k, data = items[shot_index]
    print(f"Generating GIF for shot {shot_index}...")

    # Load the starting state
    starting_state = data["starting_state"]

    for id in RENAME_IDS:
        if id in starting_state:
            starting_state = starting_state.replace(id, RENAME_IDS[id])

    state = State().from_json(starting_state)
    
    # Save the shot as a GIF
    output_path = output_dir / f"shot_{shot_index+1}.gif"
    env.save_shot_gif(state, data["params"], str(output_path))
    
    print(f"Saved GIF to {output_path}")

if __name__ == "__main__":
    main()
