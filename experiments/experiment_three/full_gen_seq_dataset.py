import json, dspy, os, logging, random
import numpy as np
import yaml
import weave
from datetime import datetime
from typing import List
from rich.console import Console
from rich.table import Table
from rich import box
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poolagent import explain_shots
from poolagent.path import DATA_DIR, ROOT_DIR
from poolagent.pool import Pool
from poolagent.agents import FunctionChooser
from poolagent.utils import State
from poolagent.experiment_manager import LLM

def explain_shot_func(
    env: Pool,
    model_id: str,
    starting_state: State,
    shot: dict,
) -> dict:

    shots = [shot] + [env.random_params() for _ in range(1,4)]
    random.shuffle(shots)

    return explain_shots(
        env=env,
        model_id=model_id,
        state=starting_state,
        shot_params=shots,
        target_balls=['red', 'blue', 'yellow'],
        unconditional=False
    )

def explain_shot_func_no_functions(
    env: Pool,
    model_id: str,
    starting_state: State,
    shot: dict,
) -> dict:

    shots = [shot] + [env.random_params() for _ in range(1,4)]
    random.shuffle(shots)

    return explain_shots(
        env=env,
        model_id=model_id,
        state=starting_state,
        shot_params=shots,
        target_balls=['red', 'blue', 'yellow'],
        unconditional=True
    )

class ExplanationGenerator:
    def __init__(self, model_ids: List[str], k_explanations: int):
        self.model_ids = model_ids
        self.k_explanations = k_explanations
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.console = Console()
        
        # Initialize components
        self.env = Pool()
        self.chooser = FunctionChooser(target_balls=['red', 'blue', 'yellow'])
        
        # Setup logging
        self.log_dir = os.path.join(ROOT_DIR, "experiments", "experiment_three", "full_logs", self.timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = self.setup_logging()
        
        # Load data
        self.explanations = {}
        self.load_data()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.log_dir}/main.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger()

    def load_data(self):
        
        self.training_data = []
        with open(f"{DATA_DIR}/model_shot_dataset.json") as f:
            self.training_data = json.load(f)

    def load_model(self, model_id: str, temperature: float = 0.2, max_tokens: int = 2048, repetition_penalty: float = 1) -> LLM:
        return LLM(model_id, temperature=temperature, max_tokens=max_tokens, repetition_penalty=repetition_penalty)

    def save_results(self, results_dir: str, explanation_data: dict):
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, "results.json")

        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                current_results = json.load(f)
        else:
            current_results = []

        current_results.append(explanation_data)

        with open(results_file, "w") as f:
            json.dump(current_results, f, indent=2)

    def print_progress(self, completed_tasks: int, total_tasks: int):
        table = Table(title="Explanation Generation Progress", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Completed Tasks", f"{completed_tasks}/{total_tasks}")
        table.add_row("Progress", f"{(completed_tasks/total_tasks)*100:.1f}%")
        
        self.console.print(table)
        self.console.print("\n" + "="*50 + "\n")

    def generate_explanations(self, model_id: str, temperature: float = 0.2, max_tokens: int = 2048, repetition_penalty: float = 1):
        model_key = model_id
        if '/' in model_id:
            model_key = model_id.split('/')[-1]
        
        if model_key not in self.explanations:
            self.explanations[model_key] = []

        existing_count = len(self.explanations[model_key])
        remaining_entries = self.training_data
        
        if not remaining_entries:
            self.logger.info(f"No remaining entries for model {model_id}")
            return

        self.logger.info(f"Generating explanations for model {model_id}")

        for i, data_entry in enumerate(remaining_entries):
            self.logger.info(f"Processing entry {existing_count + i + 1}/{len(remaining_entries)} for model {model_id}")
            
            start_state = State.from_json(data_entry['starting_state'])
            entry = data_entry['models'][model_id]
            action = entry['params']

            self.env.from_state(start_state)
            self.env.strike(**action)
            events = self.env.get_events()
            end_state = self.env.get_state()

            _, _, _, raw_values, raw_difficulties = self.chooser.evaluate_shots(
                start_state, [action], [events], [end_state]
            )
            raw_values, raw_difficulties = raw_values[0], raw_difficulties[0]
            normalized_values, normalized_difficulties = self.chooser.normalise(
                raw_values, raw_difficulties
            )

            explanation_data = {
                "state": start_state.to_json(),
                "end_state": end_state.to_json(),
                "shot_params": action,
                "events": [e.to_json() for e in events],
                "raw_values": raw_values.tolist(),
                "raw_difficulty": raw_difficulties.tolist(),
                "normalized_values": normalized_values.tolist(),
                "normalized_difficulty": normalized_difficulties.tolist(),
            }

            @weave.op()
            def generate_explanation_variants(starting_state, shot_params):
                return {
                    "with_functions": [
                        explain_shot_func(self.env, model_id, starting_state, shot_params)
                        for i in range(self.k_explanations)
                    ],
                    "without_functions": [
                        explain_shot_func_no_functions(self.env, model_id, starting_state, shot_params)
                        for i in range(self.k_explanations)
                    ]
                }

            explanation_data["explanations"] = generate_explanation_variants(
                start_state,
                action,
            )
            
            self.explanations[model_key].append(explanation_data)
            
            # Save intermediate results
            results_dir = f"{self.log_dir}/tasks/{model_key}/"
            self.save_results(results_dir, explanation_data)
            
            self.print_progress(existing_count + i + 1, len(remaining_entries))

def load_model_config() -> dict:
    """Load model configuration from yaml file."""
    config_path = os.path.join(ROOT_DIR, 'experiments/experiments_config.yaml')
    with open(config_path) as f:
        return yaml.safe_load(f)

def generate_dataset(
    model_name: str,
    k_explanations: int,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    repetition_penalty: float = 1
):

    # Initialize generator with single model
    weave.init(f"experiment-three-{model_name.split('/')[-1]}")
    generator = ExplanationGenerator([model_name], k_explanations)

    # Process just the specified model
    generator.generate_explanations(model_name, temperature, max_tokens, repetition_penalty)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run simulation with API or local models")
    parser.add_argument("--model", type=str, required=True, help="Name of the specific model to use")
    parser.add_argument("--k_explanations", type=int, default=3, help="Number of explanations to generate per shot")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature to use LLMs with.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max number of tokens to use when generating tokens with LLMs.")
    parser.add_argument("--repetition_penalty", type=float, default=1, help="Repetition penalty to use when generating tokens with LLMs.")
    args = parser.parse_args()

    generate_dataset(
        model_name=args.model,
        k_explanations=args.k_explanations,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty
    )