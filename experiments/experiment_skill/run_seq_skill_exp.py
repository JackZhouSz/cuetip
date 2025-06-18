import argparse, sys, os, logging, time, json, yaml
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from typing import Dict, List
import weave

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poolagent.pool import PoolGame
from poolagent.utils import Event, EventType, SKILL_LEVELS, State
from poolagent.path import DATA_DIR, ROOT_DIR
from poolagent.agents import *
from poolagent.value_data.gen_mcts_data import propose_shot
from poolagent.experiment_manager import LLM

from annotate_shot_dataset import ROLL_OUTS

AGENTS = {
    "ProRandomBallAgent": ProRandomBallAgent,
    "LanguageAgent": LanguageAgent,
    "LanguageDEFAgent": LanguageDEFAgent,
    "LanguageFunctionAgent": LanguageFunctionAgent,
    "VisionLanguageAgent": VisionLanguageAgent,
    "VisionLanguageDEFAgent": VisionLanguageDEFAgent,
    "VisionLanguageFunctionAgent": VisionLanguageFunctionAgent,
    "FunctionAgent": FunctionAgent
}

def gen_key(model, agent, shot_id):

    model_id = model.split('/')[-1] if '/' in model else model

    return f"{agent}_{model_id}_{shot_id}"

class PoolShot:
    def __init__(self, shot_data: dict, model: str, agent: str):
        self.agent = agent
        self.shot_data = shot_data
        self.required_events = [Event.from_encoding(e[0], e[1]) for e in shot_data['events']
                         if Event.from_encoding(e[0], e[1]).event_type == EventType.BALL_POCKET]
        self.shot_message = self._generate_message()
        shot_id = shot_data.get('shot_id', 'unknown_shot')
        self.description = gen_key(model, agent, shot_id)
        self.model = model
        self.assigned_llm = None

    def _generate_message(self) -> str:
        events_str = ', '.join([event.encoding for event in self.required_events])
        return f"Suggest a shot that causes the following events to occur: [{events_str}], and sets up the best follow up shot."

class SequentialPoolExperiment:
    def __init__(self, experiment_name: str, gpu_ids: List[str], gpu_size: int = 40):
        self.experiment_name = experiment_name
        self.gpu_size = gpu_size
        self.gpu_ids = gpu_ids
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.console = Console()
        self.env = PoolGame(visualizable=False)
        
        # Setup logging
        self.log_dir = os.path.join(ROOT_DIR, "experiments", "experiment_skill", "logs", self.timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = self.setup_logging()
        
        self.results = self.load_previous_results()
        self.logger.info(f"Loaded previous results: {len(self.results)} tasks")

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

    def load_previous_results(self) -> Dict:
        results_dir = os.path.join(ROOT_DIR, "experiments", "experiment_skill", "results")
        
        json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if not json_files:
            return {}
            
        most_recent_file = max(json_files, key=lambda f: os.path.getmtime(os.path.join(results_dir, f)))
        with open(os.path.join(results_dir, most_recent_file), 'r') as f:
            print(f"Loading results from {most_recent_file}")
            return json.load(f)

    def is_api_model(self, model_id: str) -> bool:
        return any([name in model_id for name in ['gpt', 'together']])

    def load_model(self, model_id: str) -> LLM:
        if self.is_api_model(model_id):
            return LLM(model_id)
        
        model_size = int(model_id.split('-')[1][:-1]) if '-' in model_id else 0
        required_gpus = 1 if model_size < 20 else (model_size - 1) // 20 + 1
        assigned_gpus = self.gpu_ids[:required_gpus]
        
        return LLM(model_id, assigned_gpus, self.gpu_size)

    def evaluate_shot(self, task: PoolShot) -> tuple:
        self.logger.info(f"Evaluating shot for {task.description}")
        
        state = State().from_json(task.shot_data['starting_state'])
        self.env.reset()
        self.env.from_state(state)
        
        agent = AGENTS[task.agent](['red', 'blue', 'yellow'])
        
        try:
            start_time = time.time()
            shot = agent.take_shot(
                self.env, 
                state, 
                lm=task.assigned_llm.llm if task.assigned_llm else None,
                message=task.shot_message,
                parallel=False
            )
            end_time = time.time()
            
            self.logger.info(f"Shot found in {end_time - start_time:.2f} seconds: {shot}")
            
            self.env.reset()
            self.env.from_state(state)
            self.env.take_shot('one', shot)
            
            events = self.env.get_events()
            success = all([event in events for event in task.required_events])
            
            self.logger.info(f"Required events: {task.required_events}")
            self.logger.info(f"Events: {events}")
            self.logger.info(f"Success: {success}")
            
            shot_value = self.env.get_value_estimate(
                lambda g: propose_shot(g, eps=0, skill_level=SKILL_LEVELS.BASELINE),
                initial_roll_outs=ROLL_OUTS
            )
            
            self.logger.info(f"Shot value: {shot_value}")
            
            return shot_value, success
            
        except Exception as e:
            self.logger.error(f"Error evaluating shot: {str(e)}")
            return 0.0, False

    def run_task(self, task: PoolShot):
        self.logger.info(f"Starting task: {task.description} -- for {task.num_trials} trials")
        
        num_trials = task.num_trials

        if task.model:
            task.assigned_llm = self.load_model(task.model)
        
        attempts = []
        for i in range(num_trials):
            self.logger.info(f"Trial {i + 1}/{num_trials}")
            shot_value, success = self.evaluate_shot(task)
            attempts.append((shot_value, success))
        
        results = {
            "attempts": attempts,
            "mean_val": sum([a[0] for a in attempts]) / len(attempts),
            "success_rate": sum([a[1] for a in attempts]) / len(attempts)
        }
        
        # Save results
        results_dir = os.path.join(self.log_dir, "tasks", task.description)
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Cleanup
        if task.assigned_llm:
            task.assigned_llm.delete()
        
        return results

    def print_state(self, completed_tasks: int, total_tasks: int):
        table = Table(title="Experiment Progress", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Completed Tasks", f"{completed_tasks}/{total_tasks}")
        table.add_row("Progress", f"{(completed_tasks/total_tasks)*100:.1f}%")
        
        self.console.print(table)
        self.console.print("\n" + "="*50 + "\n")

def run_experiment(config: dict, args):
    lm_size = os.getenv("SIZE", "SMALL")
    assert lm_size in ['SMALL', 'MEDIUM', 'LARGE']
    
    if args.model_type == "api":
        model_ids = config['models']['api']
    elif args.model_type == "together":
        model_ids = config['models']['together']['text'] if not args.vision else config['models']['together']['vision']
    else:
        model_ids = config['models']['local']['text'][lm_size]
        if args.vision:
            model_ids.extend(config['models']['local']['vision'][lm_size])
    
    gpu_ids = [f"gpu_{i}" for i in range(1, args.n_gpus+1)]

    exp_description = f"{args.model_type}_{'vision' if args.vision else 'text'}_{lm_size}"
    weave.init(f"ExperimentSkill-{exp_description}") 

    experiment = SequentialPoolExperiment("experiment_skill", gpu_ids, args.gpu_size)
    
    # Load shot data
    with open(os.path.join(DATA_DIR, "shot_task_dataset.json"), 'r') as f:
        shots = json.load(f)

    # Calculate total possible tasks
    agent_types = ["VisionLanguageAgent", "VisionLanguageDEFAgent", "VisionLanguageFunctionAgent"] if args.vision else ["LanguageAgent", "LanguageDEFAgent", "LanguageFunctionAgent"]
    total_tasks = len(model_ids) * len(agent_types) * len(shots)  # + 2 * len(shots) for baseline agents

    def completed_matchup_trials(results: dict, task: PoolShot) -> int:
        if task.description not in results:
            return 0
        return len(results[task.description]['attempts'])

    # Generate all possible tasks
    all_tasks = []
    for model in model_ids:
        for k, shot in shots.items():
            shot['shot_id'] = k
            for agent_type in agent_types:
                task = PoolShot(shot, model, agent_type)
                task_trials = args.num_trials - completed_matchup_trials(experiment.results, task)
                if task_trials > 0:
                    task.num_trials = task_trials
                    all_tasks.append(task)

    # Calculate split range
    total_tasks = len(all_tasks)
    start_idx = int((total_tasks * args.index / args.splits) if args.index >= 0 else 0)
    end_idx = int((total_tasks * (args.index + 1) / args.splits) if args.index >= 0 else total_tasks)
    
    # Get tasks for this split
    tasks = all_tasks[start_idx:end_idx]

    print(f"Running split {args.index + 1}/{args.splits} with {len(tasks)} tasks")
    print(f"Tasks range: {start_idx} to {end_idx} of {total_tasks} total possible tasks")
    
    # Run tasks for this split
    for i, task in enumerate(tasks):
        experiment.logger.info(f"\nRunning task {i + 1}/{len(tasks)} (Global task {start_idx + i + 1}/{total_tasks})")
        results = experiment.run_task(task)
        experiment.print_state(i + 1, len(tasks))
        
        # Save results for this split
        split_results_path = f"{experiment.log_dir}/results_split_{args.index}.json"
        with open(split_results_path, "w") as f:
            json.dump(experiment.results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sequential pool shot simulation")
    parser.add_argument("--model_type", choices=["api", "together", "local"], help="Choose 'api' for GPT models, 'local' or 'together' for Hugging Face models", default="together")
    parser.add_argument("--vision", action="store_true", help="Use vision models", default=False)
    parser.add_argument("--gpu_size", type=int, help="Size in GB of the GPU in use", default=40)
    parser.add_argument("--n_gpus", type=int, default=0, help="Number of GPUs to use")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials per task")
    parser.add_argument("--splits", type=int, default=1, help="Number of splits to run")
    parser.add_argument("--index", type=int, default=-1, help="Index of split to run (0 to splits-1)")
    args = parser.parse_args()

    # Load config and authenticate
    config = yaml.safe_load(open(os.path.join(ROOT_DIR, 'experiments/experiments_config.yaml')))
    if "HUGGINGFACE_TOKEN" in os.environ:
        from huggingface_hub import login
        login(os.environ["HUGGINGFACE_TOKEN"])
    
    run_experiment(config, args)