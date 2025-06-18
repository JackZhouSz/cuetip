import json, glob, os, argparse, yaml
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poolagent.pool import PoolGame, Fouls
from poolagent.path import DATA_DIR, ROOT_DIR
from poolagent.utils import Event, EventType, SKILL_LEVELS, State
from poolagent.agents import *
from poolagent.value_data.gen_mcts_data import propose_shot
from poolagent.experiment_manager import ExperimentManager, Task, Experiment

N_SHOTS = 250

AGENTS = {
    "ProRandomBallAgent": ProRandomBallAgent,
    "LanguageAgent": LanguageAgent,
    "LanguageFunctionAgent": LanguageFunctionAgent,
    "VisionLanguageAgent": VisionLanguageAgent,
    "VisionLanguageFunctionAgent": VisionLanguageFunctionAgent,
    "FunctionAgent": FunctionAgent
}

def get_model_config(model_type: str, config: dict, num_gpus: int) -> Tuple[List[str], List[str]]:
    """Get model and GPU configuration"""
    lm_size = os.getenv("SIZE", "SMALL")
    assert lm_size in ['SMALL', 'MEDIUM', 'LARGE']
    
    model_ids = (config['models']['api'] if model_type == "api" 
                else config['models']['local']['text'][lm_size])
    gpu_ids = [f"gpu_{i}" for i in range(1, num_gpus+1)]


    ### TEMP
    if model_type == "api":
        model_ids = ["gpt-4o-mini"]
    ### TEMP
    
    return model_ids, gpu_ids

class PoolShot(Task):
    """Task class representing a single pool shot to be evaluated"""
    def __init__(self, shot_data: dict, model: str, agent: str):
        self.agent = agent
        self.shot_data = shot_data
        self.required_events = [Event.from_encoding(e[0], e[1]) for e in shot_data['events']
                         if Event.from_encoding(e[0], e[1]).event_type == EventType.BALL_POCKET]
        self.shot_message = self._generate_message()
        description = f"{agent}-{model}-{shot_data.get('shot_id', 'unknown_shot')}"
        super().__init__(description, [model])

    def _generate_message(self) -> str:
        """Generate the message for the shot task"""
        events_str = ', '.join([event.encoding for event in self.required_events])
        return f"Suggest a shot that causes the following events to occur: [{events_str}], and sets up the best follow up shot."

class PoolExperiment(Experiment):
    """Experiment class for running pool shot evaluations"""
    def __init__(self, input_file: str, num_threads: int, models: List[str], vision: bool = False):
        super().__init__()
        self.num_threads = num_threads
        self.vision = vision
        self.environments = [PoolGame(visualizable=False) for _ in range(num_threads)]
        self.current_tasks = {}

        self.previous_results = self.load_previous_results()
        
        # Load shots and create tasks
        with open(input_file, 'r') as f:
            shots = json.load(f)
        
        self.tasks = []
        for model in models:
            for k, shot in shots.items():
                shot['shot_id'] = k

                task = PoolShot(shot, model, "LanguageAgent" if not vision else "VisionLanguageAgent")
                if task.description not in self.previous_results:
                    self.tasks.append(task)

                task = PoolShot(shot, model, "LanguageFunctionAgent" if not vision else "VisionLanguageFunctionAgent")
                if task.description not in self.previous_results:
                    self.tasks.append(task)

                task = PoolShot(shot, "none", "FunctionAgent")
                if task.description not in self.previous_results:
                    self.tasks.append(task)

                task = PoolShot(shot, "none", "ProRandomBallAgent")
                if task.description not in self.previous_results:
                    self.tasks.append(task)

    def load_previous_results(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Load previous results from file"""
        results = {}
        results_dir = os.path.join(ROOT_DIR, "experiments", "experiment_skill", "logs")
        
        # Get newest timestamp
        timestamps = [os.path.basename(f) for f in glob.glob(os.path.join(results_dir, "*"))]
        if not timestamps:
            return results
        
        for timestamp in timestamps:
            all_results_path = os.path.join(results_dir, timestamp, "all_results.json")
            if os.path.exists(all_results_path):
                with open(all_results_path, 'r') as f:
                    t_results = json.load(f)
                    results.update(t_results)

        return results

    def evaluate_shot(self, thread_id: int, task: PoolShot, llm, logger=None) -> float:
        """Evaluate a single pool shot attempt"""
        required_events = task.required_events
        state = State().from_json(task.shot_data['starting_state'])
        env = self.environments[thread_id]
        
        env.reset()
        env.from_state(state)
        agent = AGENTS[task.agent](['red', 'blue', 'yellow'])

        if logger:
            logger.info(f"Taking shot {task.shot_data['shot_id']} with agent {task.agent}")
        shot = agent.take_shot(
            env, state, lm=llm.llm, 
            message=task.shot_message, 
            parallel=False
        )

        if logger:
            logger.info(f"Shot found: {shot}")
        
        env.reset()
        env.from_state(state)
        env.take_shot('one', shot)
        
        events = env.get_events()
        success = all([event in events for event in required_events])

        if logger:
            logger.info(f"Required events: {required_events}")
            logger.info(f"Events: {events}")
            logger.info(f"Success: {success}")
        
        if logger:
            logger.info("Evaluating shot value")

        return env.get_value_estimate(
            lambda g: propose_shot(g, eps=0, skill_level=SKILL_LEVELS.BASELINE),
            initial_roll_outs=150
        ), success

    def run_task(self, thread_id: int, task: PoolShot, timestamp: str, N: int = 1, logger=None):
        """Run a pool shot task N times with each model"""

        if task.description in self.previous_results:
            if logger:
                logger.info(f"Skipping task {task.description} - already evaluated")
            return

        results = {}
        agent = task.agent
        self.current_tasks[task.description] = []
        
        for model_id, llm in task.assigned_llms.items():
            attempts = []
            
            # Run N evaluations
            for i in range(N):
                if logger:
                    logger.info(f"Model {model_id} - Agent {agent} - Shot {task.shot_data['shot_id']} - Attempt {i + 1}/{N}")
                shot_evaluation, success = self.evaluate_shot(thread_id, task, llm, logger)
                self.current_tasks[task.description].append(success)
                attempts.append(( shot_evaluation, success ))
                
                if logger:
                    logger.info(f"Shot value: {shot_evaluation} - Success: {success}")
            
                # Store results
                results = {
                    "attempts": attempts,
                    "mean_val": sum([a[0] for a in attempts]) / len(attempts),
                    "success_rate": sum([a[1] for a in attempts]) / len(attempts)
                }
                
                self._save_results(task, results, timestamp)

        del self.current_tasks[task.description]

    def _save_results(self, task: PoolShot, results: dict, timestamp: str):
        """Save task results to file"""
        results_dir = os.path.join(ROOT_DIR, "experiments", "experiment_skill", "logs", 
                                 timestamp, "tasks", task.description)
        os.makedirs(results_dir, exist_ok=True)

        with open(os.path.join(results_dir, "results.json"), 'w') as f:
                json.dump(results, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Run pool shot simulation")
    parser.add_argument("model_type", choices=["api", "local"])
    parser.add_argument("--vision", action="store_true", default=False)
    parser.add_argument("--gpu_size", type=int, default=40)
    parser.add_argument("--n_gpus", type=int, default=3)
    parser.add_argument("--n_threads", type=int, default=3)
    parser.add_argument("--n_trials", type=int, default=10)
    args = parser.parse_args()

    # Load config and authenticate
    config = yaml.safe_load(open(os.path.join(ROOT_DIR, 'experiments/experiments_config.yaml')))
    if "HUGGINGFACE_TOKEN" in os.environ:
        from huggingface_hub import login
        login(os.environ["HUGGINGFACE_TOKEN"])

    # Setup and run experiment
    model_ids, gpu_ids = get_model_config(args.model_type, config, args.n_gpus)
    experiment = PoolExperiment(
        os.path.join(DATA_DIR, "shot_task_dataset.json"),
        args.n_threads,
        model_ids,
        vision=args.vision
    )
    
    manager = ExperimentManager(
        experiment_name="experiment_skill",
        model_ids=model_ids,
        gpu_ids=gpu_ids,
        experiment=experiment,
        max_concurrent_threads=args.n_threads,
        N=args.n_trials,
        gpu_size=args.gpu_size,
        temperature=0.2
    )
    
    manager.run()

if __name__ == "__main__":
    main()