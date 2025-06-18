import json, glob, os, argparse
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from poolagent.pool import Pool, Fouls
from poolagent.path import DATA_DIR, ROOT_DIR
from poolagent.utils import Event, EventType, SKILL_LEVELS, State, dspy_setup, blur_shot
from poolagent.agents import LanguageAgent, VisionLanguageAgent
from poolagent.experiment_manager import ExperimentManager, MODELS, Task, Experiment

N_SHOTS = 250

def parse_args():
    parser = argparse.ArgumentParser(description='Run pool shot experiment with specified model type')
    parser.add_argument('--model_type', choices=['api', 'local'], required=True,
                      help='Whether to use API or local models')
    parser.add_argument('--vision', action='store_true',
                      help='Use vision models instead of text models')
    parser.add_argument('--num_threads', type=int, default=2,
                      help='Number of concurrent threads')
    parser.add_argument('--num_trials', type=int, default=10,
                      help='Number of trials per shot')
    parser.add_argument('--gpu_size', type=int, default=40,
                      help='GPU size in GB')
    parser.add_argument('--num_gpus', type=int, default=0,
                      help='Number of GPUs to use')
    args = parser.parse_args()
    print(f"\nParsed arguments:")
    print(f"Model type: {args.model_type}")
    print(f"Vision enabled: {args.vision}")
    print(f"Number of threads: {args.num_threads}")
    print(f"Number of trials: {args.num_trials}")
    print(f"GPU size: {args.gpu_size}")
    print(f"Number of GPUs: {args.num_gpus}")
    return args

def get_available_models(model_type: str, vision: bool) -> List[str]:
    """Get list of model IDs based on type and vision capability"""
    if model_type == 'api':
        models = MODELS['api']
    else:
        model_category = "vision" if vision else "text"
        models = [model_id for model_id, size in MODELS['local'][model_category]]
    
    print(f"\nAvailable models for {model_type} {'vision' if vision else 'text'} mode:")
    print(f"Found {len(models)} models: {models}")
    return models

def get_model_config(model_type: str, vision: bool, num_gpus: int) -> Tuple[List[str], List[str]]:
    """Get model and GPU configuration"""
    model_ids = get_available_models(model_type, vision)
    
    if model_type == 'api':
        gpu_ids = []
    else:
        gpu_ids = [f"GPU_{i}" for i in range(1, num_gpus + 1)]
    
    print(f"\nModel configuration:")
    print(f"Number of models: {len(model_ids)}")
    print(f"Number of GPUs: {len(gpu_ids)}")
    return model_ids, gpu_ids
    
class PoolShot(Task):
    """Task class representing a single pool shot to be evaluated"""
    def __init__(self, shot_data: dict, model: str):
        self.shot_data = shot_data
        self.shot_message = self._generate_message(shot_data)
        description = self._generate_description(shot_data, model)
        super().__init__(description, [model])
        print(f"\nInitialized PoolShot task:")
        print(f"Shot ID: {shot_data.get('shot_id', 'unknown')}")
        print(f"Model: {model}")
        print(f"Message length: {len(self.shot_message)}")
        
    def _generate_message(self, shot_data: dict) -> str:
        """Generate the message used for the shot task"""
        state = State().from_json(shot_data['starting_state'])
        req_events = get_required_events(shot_data['events'])
        events_str = ', '.join([event.encoding for event in req_events])
        print(f"Required events: {len(req_events)}")
        return f"Suggest a shot that causes the following events to occur: [{events_str}]"
    
    def _generate_description(self, shot_data: dict, model: str) -> str:
        """Generate a description that uniquely identifies this task: model_id-shot_id"""
        shot_id = shot_data.get('shot_id', 'unknown_shot')
        return f"{model}-{shot_id}" if model else f"nomodel-{shot_id}"

def get_required_events(events) -> List[Event]:
    """Extract required events from event list"""
    required_events = []
    for event in events:
        event = Event.from_encoding(event[0], event[1])
        if event.event_type in [EventType.BALL_POCKET]:
            required_events.append(event)
    print(f"Extracted {len(required_events)} required events from {len(events)} total events")
    return required_events

class PoolExperiment(Experiment):
    """Experiment class for running pool shot evaluations with consolidated results handling"""
    def __init__(self, N_attempts: int, input_file: str, output_file: str, num_threads: int, models: List[str], vision: bool = False):
        super().__init__()
        print(f"\nInitializing PoolExperiment:")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Number of threads: {num_threads}")
        print(f"Number of models: {len(models)}")
        print(f"Vision mode: {vision}")
        
        self.num_threads = num_threads
        self.input_file = input_file
        self.output_file = output_file
        self.results = {}
        self.target_balls = ['red', 'blue', 'yellow']
        self.vision = vision
        self.agent = LanguageAgent(self.target_balls) if not vision else VisionLanguageAgent(self.target_balls)
        self.agent_config = {
            "class": self.agent.__class__.__name__,
            "vision": vision,
            "target_balls": self.target_balls,
            "N_candidates": self.agent.N
        }
        self.environments = [Pool(visualizable=False) for _ in range(self.num_threads)]
        self.current_tasks = {}
        
        print(f"Created {len(self.environments)} pool environments")
        print(f"Agent type: {self.agent.__class__.__name__}")
        
        self.latest_results = self.load_all_results()
        print(f"Loaded {len(self.latest_results)} existing results")
        self.initialize_tasks(N_attempts, models)
        print(f"Initialized {len(self.tasks)} tasks")

    def evaluate_shot(self, thread_id: int, task: PoolShot, llm) -> float:
        """Evaluate a single pool shot attempt"""
        state = State().from_json(task.shot_data['starting_state'])
        req_events = get_required_events(task.shot_data['events'])

        env = self.environments[thread_id]
        env.reset()
        env.from_state(state)
        shot = self.agent.take_shot(env, state, lm=llm.llm, message=task.shot_message, parallel=False)
        
        successes = 0
        print(f"\nEvaluating shot for thread {thread_id}:")
        print(f"Required events: {len(req_events)}")
        
        for i in range(N_SHOTS):
            current_req_events = req_events.copy()
            env.from_state(state)
            foul = env.strike(**blur_shot(shot, skill=SKILL_LEVELS.BASELINE), check_rules=True, target_balls=self.target_balls)
            events = env.get_events()

            if foul != Fouls.NONE:
                continue

            for event in events:
                if current_req_events and event == current_req_events[0]:
                    current_req_events.pop(0)
                if len(current_req_events) == 0:
                    successes += 1
                    break
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{N_SHOTS} shots, current success rate: {successes/(i+1):.3f}")

        return successes / N_SHOTS

    def load_results(self) -> dict:
        """Load existing results or create new results dictionary"""
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r') as f:
                results = json.load(f)
                print(f"Loaded {len(results)} existing results from {self.output_file}")
                return results
        print(f"No existing results found at {self.output_file}")
        return {}

    def load_all_results(self) -> Dict:
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

    def get_completed_attempts(self, task: PoolShot, model_id: str) -> List[float]:
        """Get list of completed attempts for a given task and model"""
        task_id = f"task_{task.shot_data['shot_id']}"
        if task_id not in self.latest_results:
            print(f"No results found for task {task_id}")
            return []
            
        agent_type = "vision" if self.vision else "language"
        agent_id = f"{self.agent_config['class']}_{agent_type}"
        
        task_results = self.latest_results[task_id]
        if agent_id not in task_results:
            print(f"No results found for agent {agent_id}")
            return []
            
        model_results = task_results[agent_id].get("models", {}).get(model_id, {})
        attempts = model_results.get("attempts", [])
        print(f"Found {len(attempts)} completed attempts for task {task_id}, model {model_id}")
        return attempts

    def initialize_tasks(self, N: int, models: List[str]):
        """Create PoolShot tasks from input file, skipping completed ones"""
        with open(self.input_file, 'r') as f:
            shots = json.load(f)
        
        print(f"\nInitializing tasks from {len(shots)} shots")
        skipped = 0
        created = 0
        
        for model_id in models:
            for i, shot in enumerate(shots):
                shot['shot_id'] = f"shot_{i}"
                task = PoolShot(shot, model_id)
                if self.should_skip_task(task, N):
                    skipped += 1
                else:
                    self.tasks.append(task)
                    created += 1
        
        print(f"Created {created} new tasks")
        print(f"Skipped {skipped} completed tasks")

    def should_skip_task(self, task: PoolShot, N: int) -> bool:
        """Check if task is already complete for all models"""
        for model_id in task.models:
            completed_attempts = self.get_completed_attempts(task, model_id)
            print(f"Found {len(completed_attempts)} completed attempts for model {model_id}")
            if len(completed_attempts) >= N:
                return True
        return False

    def save_task_results(self, task: PoolShot, task_results: dict, timestamp: str):
        """Save consolidated results for a task in a single results.json file"""
        task_id = f"task_{task.shot_data['shot_id']}"
        task_dir = os.path.join(ROOT_DIR, "experiments", "experiment_demo", "logs", timestamp, "tasks", task_id)
        os.makedirs(task_dir, exist_ok=True)
        print(f"\nSaving results for {task_id} to {task_dir}")
        
        results_file = os.path.join(task_dir, "results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                all_results = json.load(f)
                print(f"Loaded existing results from {results_file}")
        else:
            all_results = {
                "task_id": task_id,
                "timestamp": timestamp,
                "message": task.shot_message,
                "starting_state": task.shot_data['starting_state']
            }
            print("Created new results structure")

        agent_type = "vision" if self.vision else "language"
        agent_id = f"{self.agent_config['class']}_{agent_type}"
        
        if agent_id not in all_results:
            all_results[agent_id] = {
                "agent_config": self.agent_config,
                "models": {}
            }
            
        for model_id, model_results in task_results.items():
            if model_id not in all_results[agent_id]["models"]:
                all_results[agent_id]["models"][model_id] = {
                    "attempts": [],
                    "N_SHOTS": N_SHOTS
                }
            
            model_attempts = model_results["shots"][task.shot_data['shot_id']]["attempts"]
            all_results[agent_id]["models"][model_id]["attempts"] = model_attempts
            print(f"Updated results for model {model_id}: {len(model_attempts)} attempts")

        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"Saved results to {results_file}")

    def run_task(self, thread_id: int, task: PoolShot, timestamp: str, N: int = 1, logger=None):
        """Run a pool shot task N times with each model, continuing from previous attempts"""
        print(f"\nRunning task {task.description} on thread {thread_id}")
        print(f"Requested attempts: {N}")
        
        self.current_tasks[task.description] = []
        results = {}
        
        for model_id, llm in task.assigned_llms.items():
            print(f"\nProcessing model: {model_id}")
            
            # Initialize results structure
            results[model_id] = {
                "shots": {
                    task.shot_data['shot_id']: {
                        "attempts": self.get_completed_attempts(task, model_id),
                        "message": task.shot_message
                    }
                },
                "timestamp": timestamp,
                "N_SHOTS": N_SHOTS
            }

            self.current_tasks[task.description] = [1 for _ in results[model_id]["shots"][task.shot_data['shot_id']]["attempts"]]
            
            # Get number of remaining attempts
            existing_attempts = results[model_id]["shots"][task.shot_data['shot_id']]["attempts"]
            remaining_attempts = N - len(existing_attempts)
            print(f"Found {len(existing_attempts)} existing attempts")
            print(f"Remaining attempts to run: {remaining_attempts}")
            
            # Run remaining attempts
            for attempt in range(remaining_attempts):
                print(f"\nStarting attempt {attempt + 1}/{remaining_attempts}")
                success_rate = self.evaluate_shot(thread_id, task, llm)
                results[model_id]["shots"][task.shot_data['shot_id']]["attempts"].append(success_rate)
                
                current_attempt = len(existing_attempts) + attempt
                self.current_tasks[task.description].append(1)
                
                if logger:
                    logger.info(f"Model {model_id} - Shot {task.shot_data['shot_id']} - "
                              f"Attempt {current_attempt + 1}/{N}: "
                              f"Success rate {success_rate:.3f}")
                print(f"Completed attempt with success rate: {success_rate:.3f}")
        
        # Save consolidated task results
        print(f"\nSaving results for task {task.description}")
        self.save_task_results(task, results, timestamp)
        
        if all(self.current_tasks[task.description]):
            print(f"Task {task.description} completed and removed from current tasks")
            del self.current_tasks[task.description]

def main():
    args = parse_args()
    
    # Configuration
    input_file = f"{DATA_DIR}/skill_estimate_dataset.json"
    output_file = f"{DATA_DIR}/skill_estimate_results.json"
    experiment_name = "experiment_demo"
    
    print(f"\nStarting experiment with configuration:")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Experiment name: {experiment_name}")
    
    # Get model and GPU configuration based on arguments
    model_ids, gpu_ids = get_model_config(args.model_type, args.vision, args.num_gpus)
    
    # Print configuration
    print(f"\nExperiment Configuration:")
    print(f"Model Type: {args.model_type}")
    print(f"Vision Models: {args.vision}")
    print(f"Selected Models: {model_ids}")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"GPU Configuration: {gpu_ids if gpu_ids else 'No GPUs (API mode)'}")
    print(f"Number of Threads: {args.num_threads}")
    print(f"Trials per Shot: {args.num_trials}\n")
    
    # Create and run experiment
    print("\nInitializing experiment...")
    experiment = PoolExperiment(args.num_trials, input_file, output_file, args.num_threads, model_ids, vision=args.vision)
    
    print("\nInitializing experiment manager...")
    manager = ExperimentManager(
        experiment_name=experiment_name,
        model_ids=model_ids,
        gpu_ids=gpu_ids,
        experiment=experiment,
        max_concurrent_threads=args.num_threads,
        N=args.num_trials,
        gpu_size=args.gpu_size
    )
    
    # Run the experiment
    print("\nStarting experiment execution...")
    manager.run()
    print("\nExperiment completed")
    
if __name__ == "__main__":
    main()

