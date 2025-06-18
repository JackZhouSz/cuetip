import json, re
import os
import argparse
import logging
from datetime import datetime
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich import box

from poolagent.pool import Pool, Fouls
from poolagent.path import DATA_DIR, ROOT_DIR
from poolagent.utils import Event, EventType, SKILL_LEVELS, State, blur_shot
from poolagent.agents import LanguageAgent, VisionLanguageAgent
from poolagent.experiment_manager import MODELS, LLM

# Constants
N_SHOTS = 250
GPU_SIZE = 40
MAX_MODEL_SIZE_FOR_SINGLE_GPU = GPU_SIZE // 2

class PoolShot:
    """Represents a single pool shot to be evaluated"""
    def __init__(self, shot_data: dict, model: str):
        self.shot_data = shot_data
        self.shot_message = self._generate_message(shot_data)
        self.description = self._generate_description(shot_data, model)
        self.models = [model] if model else []
        self.assigned_llms = {}

    def _generate_message(self, shot_data: dict) -> str:
        """Generate the message used for the shot task"""
        state = State().from_json(shot_data['starting_state'])
        req_events = get_required_events(shot_data['events'])
        return f"Suggest a shot that causes the following events to occur: [{', '.join([event.encoding for event in req_events])}]"

    def _generate_description(self, shot_data: dict, model: str) -> str:
        """Generate a description that uniquely identifies this task"""
        shot_id = shot_data.get('shot_id', 'unknown_shot')
        return f"{model}-{shot_id}" if model else f"nomodel-{shot_id}"

def get_required_events(events) -> List[Event]:
    """Extract required events from event list"""
    required_events = []
    for event in events:
        event = Event.from_encoding(event[0], event[1])
        if event.event_type in [EventType.BALL_POCKET]:
            required_events.append(event)
    return required_events

class SequentialPoolExperiment:
    """Manages sequential execution of pool shot evaluations"""
    def __init__(self, experiment_name: str, input_file: str, vision: bool = False, gpu_ids: List[int] = None):
        self.experiment_name = experiment_name
        self.input_file = input_file
        self.vision = vision
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.console = Console()
        self.gpu_ids = gpu_ids
        self.current_results = self.load_current_results()
        
        # Setup environment and agent
        self.target_balls = ['red', 'blue', 'yellow']
        self.env = Pool(visualizable=False)
        self.agent = VisionLanguageAgent(self.target_balls) if vision else LanguageAgent(self.target_balls)
        
        # Setup logging and results directory
        self.setup_logging()
        self.results = {}
        
    def setup_logging(self):
        """Initialize logging"""
        self.log_dir = f"{ROOT_DIR}/experiments/{self.experiment_name}/logs/{self.timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.log_dir}/experiment.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()

    def load_current_results(self) -> Dict[str, dict]:
        """Load results from all dated experiment directories under logs"""
        base_dir = os.path.join(ROOT_DIR, "experiments", "experiment_demo", "logs")
        if not os.path.exists(base_dir):
            print(f"Base directory not found: {base_dir}")
            return {}

        print(f"\nLoading results from {base_dir}")
        all_results = {}

        for exp_dir in os.listdir(base_dir):
            exp_path = os.path.join(base_dir, exp_dir)
            if not os.path.isdir(exp_path):
                continue
                
            results_file = os.path.join(exp_path, "all_results.json")
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                        for task_id, task_results in results.items():
                            all_results[task_id] = task_results
                        print(f"Loaded results from: {results_file}")
                except json.JSONDecodeError:
                    print(f"Warning: Could not load results from {results_file}")
                except Exception as e:
                    print(f"Error loading {results_file}: {str(e)}")

        if not all_results:
            print("No results files found")
            
        return all_results
    
    def load_shots(self) -> List[dict]:
        """Load shot data from input file"""
        with open(self.input_file, 'r') as f:
            shots = json.load(f)
        self.logger.info(f"Loaded {len(shots)} shots from {self.input_file}")
        return shots

    def evaluate_shot(self, shot: PoolShot, llm) -> float:
        """Evaluate a single pool shot attempt"""
        state = State().from_json(shot.shot_data['starting_state'])
        req_events = get_required_events(shot.shot_data['events'])
        
        # Get shot suggestion from agent
        self.env.reset()
        self.env.from_state(state)
        suggested_shot = self.agent.take_shot(
            self.env, 
            state, 
            lm=llm.llm, 
            message=shot.shot_message,
            parallel=False
        )
        
        # Evaluate shot N_SHOTS times
        successes = 0
        for i in range(N_SHOTS):
            current_req_events = req_events.copy()
            self.env.from_state(state)
            
            # Execute shot
            foul = self.env.strike(
                **blur_shot(suggested_shot, skill=SKILL_LEVELS.BASELINE),
                check_rules=True,
                target_balls=self.target_balls
            )
            
            if foul != Fouls.NONE:
                continue
                
            # Check if required events occurred
            events = self.env.get_events()
            for event in events:
                if current_req_events and event == current_req_events[0]:
                    current_req_events.pop(0)
                if len(current_req_events) == 0:
                    successes += 1
                    break
                    
            if (i + 1) % 50 == 0:
                self.logger.info(f"Completed {i + 1}/{N_SHOTS} shots, current success rate: {successes/(i+1):.3f}")
                
        return successes / N_SHOTS

    def get_model_size(self, model_id):
        match = re.search(r'-(\d+)b', model_id.lower())
        return int(match.group(1)) if match else 0
    
    def get_required_gpus(self, model_size):
        return 1 if model_size < MAX_MODEL_SIZE_FOR_SINGLE_GPU else (model_size - 1) // MAX_MODEL_SIZE_FOR_SINGLE_GPU + 1

    def save_results(self, shot: PoolShot, model_id: str, success_rate: float):
        """Save results for a shot attempt"""
        task_id = f"task_{shot.shot_data['shot_id']}"
        task_dir = os.path.join(self.log_dir, "tasks", task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        results_file = os.path.join(task_dir, "results.json")
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            results = {
                "task_id": task_id,
                "timestamp": self.timestamp,
                "message": shot.shot_message,
                "starting_state": shot.shot_data['starting_state']
            }
            
        # Update results
        agent_type = "vision" if self.vision else "language"
        agent_id = f"{self.agent.__class__.__name__}_{agent_type}"
        
        if agent_id not in results:
            results[agent_id] = {
                "agent_config": {
                    "class": self.agent.__class__.__name__,
                    "vision": self.vision,
                    "target_balls": self.target_balls,
                    "N_candidates": self.agent.N
                },
                "models": {}
            }
            
        if model_id not in results[agent_id]["models"]:
            results[agent_id]["models"][model_id] = {
                "attempts": [],
                "N_SHOTS": N_SHOTS
            }
            
        results[agent_id]["models"][model_id]["attempts"].append(success_rate)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        self.results[task_id] = results

    def print_progress(self, completed: int, total: int):
        """Display progress information"""
        table = Table(title="Experiment Progress", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Completed Shots", f"{completed}/{total}")
        table.add_row("Progress", f"{(completed/total)*100:.1f}%")
        
        self.console.print(table)
        self.console.print("\n" + "="*50 + "\n")

    def run(self, models: List[str], n_trials: int, gpu_size: int):
        """Run the experiment sequentially"""
        shots = self.load_shots()
        total_tasks = len(shots)  
        completed = 0
        previous_lm = None
        
        for i, shot_data in enumerate(shots):
            shot_data['shot_id'] = f"shot_{i}"
            
            for model_id in models:
                self.logger.info(f"\nProcessing shot {i+1}/{len(shots)} with model {model_id}")
                shot = PoolShot(shot_data, model_id)

                if shot.description in self.current_results and model_id in self.current_results[shot.description]['LanguageAgent_language']['models']:
                    self.logger.info("Skipping shot, already evaluated")
                    completed += 1
                    continue
                
                for trial in range(n_trials):
                    self.logger.info(f"\nStarting trial {trial+1}/{n_trials}")
                    
                    # Load model
                    if model_id:
                        
                        model_size = self.get_model_size(model_id)
                        required_gpus = self.get_required_gpus(model_size)
                        gpu_ids = self.gpu_ids[:required_gpus]

                        if previous_lm:
                            if model_id != previous_lm.model_id:
                                previous_lm.delete()
                                previous_lm = None

                                shot.assigned_llms[model_id] = LLM(
                                    model_id, 
                                    gpu_size=gpu_size,
                                    gpu_ids=gpu_ids
                                )

                            else:
                                shot.assigned_llms[model_id] = previous_lm
                        else:
                            shot.assigned_llms[model_id] = LLM(
                                model_id, 
                                gpu_size=gpu_size,
                                gpu_ids=gpu_ids
                            )
                    
                    # Evaluate shot
                    try:
                        success_rate = self.evaluate_shot(shot, shot.assigned_llms[model_id])
                        self.logger.info(f"Trial completed with success rate: {success_rate:.3f}")
                        
                        # Save results
                        self.save_results(shot, model_id, success_rate)
                        
                        previous_lm = shot.assigned_llms[model_id].delete()
                            
                    except Exception as e:
                        self.logger.error(f"Error in trial: {str(e)}")
                        continue
                    
                completed += 1
                self.print_progress(completed, total_tasks)
        
            # Save final results
            with open(f"{self.log_dir}/all_results.json", "w") as f:
                json.dump(self.results, f, indent=4)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run sequential pool shot experiment')
    parser.add_argument('--model_type', choices=['api', 'local'], required=True,
                      help='Whether to use API or local models')
    parser.add_argument('--vision', action='store_true',
                      help='Use vision models instead of text models')
    parser.add_argument('--num_trials', type=int, default=10,
                      help='Number of trials per shot')
    parser.add_argument('--gpu_size', type=int, default=40,
                      help='GPU size in GB')
    parser.add_argument('--num_gpus', type=int, default=1,
                      help='Number of GPUs to use')
    args = parser.parse_args()
    
    # Setup experiment
    input_file = f"{DATA_DIR}/skill_estimate_dataset.json"
    experiment_name = "experiment_demo"
    
    # Get available models
    if args.model_type == 'api':
        models = MODELS['api']
    else:
        model_category = "vision" if args.vision else "text"
        models = [model_id for model_id, _ in MODELS['local'][model_category]]
    
    gpu_ids = [f"gpu_{i+1}" for i in range(args.num_gpus)]

    # Run experiment
    experiment = SequentialPoolExperiment(
        experiment_name=experiment_name,
        input_file=input_file,
        vision=args.vision,
        gpu_ids=gpu_ids
    )
    
    experiment.run(
        models=models, 
        n_trials=args.num_trials,
        gpu_size=args.gpu_size
    )

if __name__ == "__main__":
    from huggingface_hub import login
    assert "HUGGINGFACE_TOKEN" in os.environ, "Please set HUGGINGFACE_TOKEN"
    login(os.environ["HUGGINGFACE_TOKEN"])
    main()