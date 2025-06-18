import json
import os
import yaml
from typing import Dict, List
from datetime import datetime
from rich.console import Console
from rich.progress import track

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poolagent.path import ROOT_DIR, DATA_DIR
from poolagent.experiment_manager import LLM
from poolagent.pool import Pool
from poolagent.agents import LanguageFunctionAgent
from poolagent.utils import State

def generate_shots(env: Pool, state: Dict, model: str) -> Dict:
    state = State(state['positions'])
    env.from_state(state)
    llm = LLM(model)
    agent = LanguageFunctionAgent(target_balls=['red', 'blue', 'yellow'])
    shot = agent.take_shot(env, state, llm.llm)

    return {
        "params": shot,
        "events": [e.to_json() for e in env.get_events()],
        "difficulties": agent.record['function_chooser']['difficulties'][agent.record['chosen_shot']]
    }

def process_dataset(model_type: str):
    console = Console()
    experiments_config = yaml.safe_load(open(ROOT_DIR + '/experiments/experiments_config.yaml'))
    
    assert model_type in ["api", "local", "together", "custom"], "Invalid model type"
    lm_size = os.getenv("SIZE", "SMALL")
    assert lm_size in ['SMALL', 'MEDIUM', 'LARGE'], "Invalid SIZE env variable"
    
    if model_type == "api":
        models = experiments_config['models']['api']
    elif model_type == "together":
        models = experiments_config['models']['together']['text']
    elif model_type == "local":
        models = experiments_config['models']['local']['text'][lm_size]
    else:
        models = experiments_config['models']['custom']
    
    input_path = os.path.join(DATA_DIR, "skill_estimate_dataset.json")
    output_path = os.path.join(DATA_DIR, "model_shot_dataset.json")
    
    # Load existing results if available
    existing_results = {}
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing_data = json.load(f)
            for entry in existing_data:
                state_key = json.dumps(entry['starting_state'], sort_keys=True)
                existing_results[state_key] = entry['models']
    
    with open(input_path, 'r') as f:
        dataset = json.load(f)

    env = Pool()
    processed_dataset = []
    
    for entry in track(dataset, description="Processing entries"):
        processed_entry = entry.copy()
        state_key = json.dumps(entry['starting_state'], sort_keys=True)
        
        # Initialize or load existing models data
        if state_key in existing_results:
            processed_entry['models'] = existing_results[state_key]
        else:
            processed_entry['models'] = {}
            
        # Process only missing models
        for model in models:
            if model not in processed_entry['models']:
                console.print(f"Processing model: {model}")
                results = generate_shots(env=env, state=entry['starting_state'], model=model)
                processed_entry['models'][model] = results
            else:
                console.print(f"Skipping existing model: {model}")
        
        processed_dataset.append(processed_entry)
        
        # Save intermediate results
        with open(output_path, 'w') as f:
            json.dump(processed_dataset, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, 
                       choices=["api", "local", "together", "custom"],
                       help="Type of models to use")
    
    args = parser.parse_args()
    process_dataset(args.model_type)