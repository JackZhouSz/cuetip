import argparse, sys, os, logging, time, dspy, traceback, json, re, dotenv, torch, yaml
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

import weave

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from poolagent.path import ROOT_DIR
from poolagent.agents import *
AGENTS = {
    "ProRandomBallAgent": ProRandomBallAgent,
    "LanguageAgent": LanguageAgent,
    "LanguageDEFAgent": LanguageDEFAgent,
    "LanguageFunctionAgent": LanguageFunctionAgent,
    "VisionLanguageAgent": VisionLanguageAgent,
    "VisionLanguageDEFAgent": VisionLanguageDEFAgent,
    "VisionLanguageFunctionAgent": VisionLanguageFunctionAgent,
    "FunctionAgent": FunctionAgent,
    "PoolMasterAgent": PoolMasterAgent
}
from poolagent.pool import PoolGame
from poolagent.utils import blur_shot, SKILL_LEVELS

NOISE_LEVELS = {
    'no_noise': SKILL_LEVELS.NONE,
    'novice': SKILL_LEVELS.NOVICE,
    'amateur': SKILL_LEVELS.AMATEUR,
    'pro': SKILL_LEVELS.PRO
}

# Constants
GPU_SIZE = 40
MAX_MODEL_SIZE_FOR_SINGLE_GPU = GPU_SIZE // 2
PARALLEL = False

from poolagent.experiment_manager import LLM

def gen_key(agent_one, agent_two):

    model_id_one = agent_one.model_id
    if model_id_one:
        model_id_one = agent_one.model_id.split('/')[-1] if '/' in model_id_one else model_id_one

    model_id_two = agent_two.model_id
    if model_id_two:
        model_id_two = agent_two.model_id.split('/')[-1] if '/' in model_id_two else model_id_two

    return f"{agent_one.name}_{model_id_one}---{agent_two.name}_{model_id_two}"

class Agent:
    def __init__(self, name, model_id=None):
        self.name = name
        self.model_id = model_id

class Task:
    def __init__(self, description, models):
        self.description = description
        self.models = models
        self.assigned_llms = {}

class Matchup(Task):
    def __init__(self, description, agent_one, agent_two, noise_level, num_trials=10):
        models = [model_id for model_id in [agent_one.model_id, agent_two.model_id] if model_id]
        super().__init__(description, models)
        self.agent_one = agent_one
        self.agent_two = agent_two
        self.num_trials = num_trials
        self.noise_level = noise_level

class SequentialExperimentManager:
    def __init__(self, experiment_name, gpu_ids, gpu_size=40):
        self.experiment_name = experiment_name
        self.gpu_size = gpu_size
        self.gpu_ids = gpu_ids
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.console = Console()
        self.env = PoolGame()
        
        # Setup logging
        self.log_dir = os.path.join(ROOT_DIR, "experiments", "experiment_one", "noise_logs", self.timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = self.setup_logging()

        self.results = self.load_previous_results()
        self.logger.info(f"Loaded previous results: {len(self.results)} tasks")
        
    def load_previous_results(self):
        results_folders = ROOT_DIR + "/experiments/experiment_one/noise_results"
        json_files = [f for f in os.listdir(results_folders) if f.endswith('.json')]
        most_recent_file = max(json_files, key=lambda f: os.path.getmtime(os.path.join(results_folders, f)))
        with open(os.path.join(results_folders, most_recent_file), 'r') as f:
            return json.load(f)

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
    
    def gen_key(self, agent_one, agent_two):

        model_id_one = agent_one.model_id
        if model_id_one:
            model_id_one = agent_one.model_id.split('/')[-1] if '/' in model_id_one else model_id_one

        model_id_two = agent_two.model_id
        if model_id_two:
            model_id_two = agent_two.model_id.split('/')[-1] if '/' in model_id_two else model_id_two

        return f"{agent_one.name}_{model_id_one}---{agent_two.name}_{model_id_two}"

    def get_model_size(self, model_id):
        match = re.search(r'-(\d+)b', model_id.lower())
        return int(match.group(1)) if match else 0
    
    def get_required_gpus(self, model_size):
        return 1 if model_size < MAX_MODEL_SIZE_FOR_SINGLE_GPU else (model_size - 1) // MAX_MODEL_SIZE_FOR_SINGLE_GPU + 1
    
    def is_api_model(self, model_id):
        return any([name in model_id for name in ['gpt', 'together']])

    def bot_shot(self, env, state, player):
        return player.take_shot(env, state)
    def language_agent_shot(self, env, state, player, logger, lm):
        return player.take_shot(env, state, lm.llm, logger=logger, parallel=PARALLEL)
    def language_def_agent_shot(self, env, state, player, logger, lm):
        return player.take_shot(env, state, lm.llm, logger=logger, parallel=PARALLEL)
    def language_function_agent_shot(self, env, state, player, logger, lm):
        return player.take_shot(env, state, lm.llm, logger=logger, parallel=PARALLEL)
    def function_agent_shot(self, env, state, player, logger):
        return player.take_shot(env, state, logger=logger, parallel=PARALLEL)

    def get_agent_shot(self, env, state, player, current_settings, logger, lm):
        if isinstance(player, ProRandomBallAgent):
            return self.bot_shot(env, state, player)
        elif isinstance(player, LanguageAgent):
            return self.language_agent_shot(env, state, player, logger, lm)
        elif isinstance(player, LanguageDEFAgent):
            return self.language_def_agent_shot(env, state, player, logger, lm)
        elif isinstance(player, LanguageFunctionAgent):
            return self.language_function_agent_shot(env, state, player, logger, lm)
        elif isinstance(player, FunctionAgent):
            return self.function_agent_shot(env, state, player, logger)
        else:
            logger.error(f"Unknown agent type: {player.__class__.__name__}")
            return None

    def load_model(self, model_id):
        if self.is_api_model(model_id):
            return LLM(model_id)
        
        model_size = self.get_model_size(model_id)
        required_gpus = self.get_required_gpus(model_size)
        assigned_gpus = self.gpu_ids[:required_gpus]
        
        return LLM(model_id, assigned_gpus, self.gpu_size)
    
    def save_results(self, matchup_dir, reward):
        results_file = os.path.join(matchup_dir, "results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            results['games'].append(reward)
            results['winrate'] = sum(results['games']) / len(results['games'])
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            with open(results_file, 'w') as f:
                json.dump({
                    "games": [reward],
                    "winrate": reward
                }, f, indent=2)  

    def run_game(self, matchup, game_num, noise_level):
        self.logger.info(f"Starting matchup simulation: {matchup.agent_one.name} vs {matchup.agent_two.name}")

        llm_map = {
            'one': matchup.assigned_llms.get(matchup.agent_one.model_id, None),
            'two': matchup.assigned_llms.get(matchup.agent_two.model_id, None)
        }

        matchup_key = self.gen_key(matchup.agent_one, matchup.agent_two)
        matchup_dir = os.path.join(ROOT_DIR, "experiments", "experiment_one", "noise_logs", self.timestamp, noise_level, "tasks", matchup_key)
        os.makedirs(matchup_dir, exist_ok=True)

        m_agent_one = matchup.agent_one
        m_agent_two = matchup.agent_two
        agent_one = AGENTS[m_agent_one.name](target_balls=['red', 'blue', 'yellow'])
        agent_two = AGENTS[m_agent_two.name](target_balls=['black', 'pink', 'green'])

        players = {
            'one': agent_one,
            'two': agent_two
        }

        self.env.reset()
        state = self.env.get_state()

        self.env.current_player = 'one'
        self.env.double_shot = False

        turn_count = 0
        game_log = {}
        MAX_TURNS = 20
        ended_in_draw = False

        while not self.env.check_win():
            turn_count += 1

            if turn_count > MAX_TURNS:
                self.logger.info(f"Game {game_num + 1} finished in draw after {MAX_TURNS} turns, no winner")
                ended_in_draw = True
                break

            current_settings = {
                'current_player': self.env.current_player,
                'double_shot': self.env.double_shot
            }

            player = players[current_settings['current_player']]
            current_llm = llm_map[current_settings['current_player']]

            self.logger.info(f"Turn {turn_count}: Player {current_settings['current_player']} ({player.__class__.__name__})")

            max_retries = 3
            for retry in range(max_retries):
                try:
                    self.logger.info(f"Taking shot for {current_settings['current_player']}")
                    start_time = time.time()
                    start_state = self.env.get_state()
                    shot = self.get_agent_shot(self.env, start_state, player, current_settings, self.logger, current_llm)
                    end_time = time.time()
                    self.logger.info(f"Shot taken in {end_time - start_time:.2f} seconds")
                    
                    self.env.from_state(start_state)
                    self.env.current_player = current_settings['current_player']
                    self.env.double_shot = current_settings['double_shot']
                    self.env.take_shot(current_settings['current_player'], blur_shot(shot, skill=NOISE_LEVELS[noise_level]))
                    end_state = self.env.get_state()
                    self.logger.info(f"Shot taken in environment")
                    
                    game_log[f'shot_{turn_count}'] = {
                        'player': current_settings['current_player'],
                        'shot': shot,
                        'double_shot': current_settings['double_shot'],
                        'llm': current_llm.model_id if current_llm else None,
                        'gpu_ids': current_llm.gpu_ids if current_llm else None,
                        'execution_time': float(end_time - start_time),
                        **player.record
                    }

                    events = self.env.get_events()

                    game_log[f'shot_{turn_count}']['events'] = [e.to_json() for e in events]
                    game_log[f'shot_{turn_count}']['start_state'] = start_state.to_json()
                    game_log[f'shot_{turn_count}']['end_state'] = end_state.to_json()

                    self.logger.info(f"Shot taken: {shot}")
                    events_str = ", ".join([e.encoding for e in events])
                    self.logger.info(f"  Events: {events_str}")
                    break
                except Exception as e:
                    self.logger.error(f"Error in {current_settings['current_player']}: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    if retry < max_retries - 1:
                        self.logger.warning(f"Error in {current_settings['current_player']} - retrying")
                        time.sleep(1)  # Wait a bit before retrying
                    else:
                        self.logger.error(f"Error in {current_settings['current_player']} - failed after {max_retries} attempts")
                        self.logger.error(traceback.format_exc())
                        game_log[f'shot_{turn_count}'] = {
                            'player': current_settings['current_player'],
                            'error': str(e)
                        }
                        self.env.current_player = 'one' if self.env.current_player == 'two' else 'two'
                        self.env.double_shot = True

        def calc_draw_reward(env):

            state = env.get_state()
            balls = state.ball_positions

            player_one_balls = ['red', 'blue', 'yellow']
            player_two_balls = ['black', 'pink', 'green']
            
            num_player_one_balls = 0
            num_player_two_balls = 0

            for ball, pos in balls.items():
                if isinstance(pos[0], str):
                    continue
                if ball in player_one_balls:
                    num_player_one_balls += 1
                elif ball in player_two_balls:
                    num_player_two_balls += 1

            if num_player_one_balls == num_player_two_balls:
                return 0.5

            if num_player_one_balls > num_player_two_balls:
                return 1
            
            if num_player_two_balls > num_player_one_balls:
                return 0

        game_reward = self.env.reward()[0] if not ended_in_draw else calc_draw_reward(self.env)

        self.logger.info(f"Game {game_num + 1} finished. Reward: {game_reward}")

        game_log_file = os.path.join(matchup_dir, f"game_{game_num + 1}.json")
        with open(game_log_file, 'w') as f:
            json.dump(game_log, f, indent=2)

        self.save_results(matchup_dir, game_reward)

        return game_reward
    
    @weave.op()
    def run_matchup(self, matchup):
        self.logger.info(f"Starting matchup: {matchup.description} -- running {matchup.num_trials} games")
        
        n_games = matchup.num_trials
        noise_level = matchup.noise_level

        # Load models for the matchup
        for model_id in matchup.models:
            if model_id:
                matchup.assigned_llms[model_id] = self.load_model(model_id)
        
        matchup_dir = f"{self.log_dir}/{noise_level}/tasks/{matchup.description}"
        os.makedirs(matchup_dir, exist_ok=True)
        
        rewards = []
        for game_num in range(n_games):
            self.logger.info(f"Starting game {game_num + 1}/{n_games}")
            
            try:
                game_result = self.run_game(matchup, game_num, noise_level)
                rewards.append(game_result)
                    
            except Exception as e:
                self.logger.error(f"Error in game {game_num + 1}: {str(e)}")
                self.logger.error(traceback.format_exc())
        
        # Clean up models
        for llm in matchup.assigned_llms.values():
            llm.delete()
        
        # Save matchup results
        results = {
            'games': rewards,
            'winrate': sum(rewards) / len(rewards) if rewards else 0
        }
        with open(f"{matchup_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        self.results[matchup.description] = results
        return results
    
    def print_state(self, completed_tasks, total_tasks):
        table = Table(
            title="Experiment Progress",
            box=box.ROUNDED
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Completed Tasks", f"{completed_tasks}/{total_tasks}")
        table.add_row("Progress", f"{(completed_tasks/total_tasks)*100:.1f}%")
        
        self.console.print(table)
        self.console.print("\n" + "="*50 + "\n")

def run_experiment(experiments_config, args):

    model_type = args.model_type
    vision = args.vision
    gpu_size = args.gpu_size
    num_gpus = args.n_gpus
    matchup_splits = args.splits
    matchup_index = args.index
    num_trials = args.num_trials

    assert model_type in ["api", "local", "together", "custom"], "Invalid model type. Choose 'api', 'together', 'local', or 'custom."
    
    # Initialize experiment manager
    gpu_ids = [f"gpu_{i}" for i in range(1, num_gpus+1)]
    manager = SequentialExperimentManager("experiment_one", gpu_ids, gpu_size)
    
    if os.getenv("SIZE"):
        lm_size_for_experiment = os.getenv("SIZE")
        assert lm_size_for_experiment in ['SMALL', 'MEDIUM', 'LARGE'], "Invalid experiment SIZE variable. Choose 'SMALL', 'MEDIUM', or 'LARGE'."
    else:
        lm_size_for_experiment = 'SMALL'

    api_model_ids = experiments_config['models']['api'] 

    together_model_ids = experiments_config['models']['together']['text']
    together_vision_model_ids = experiments_config['models']['together']['vision']

    local_model_ids = experiments_config['models']['local']['text'][lm_size_for_experiment]
    local_vision_model_ids = experiments_config['models']['local']['vision'][lm_size_for_experiment]

    custom_model_ids = experiments_config['models']['custom']

    gpu_ids = [f"gpu_{i}" for i in range(1, num_gpus+1)]

    exp_description = f"{model_type}_{'vision' if vision else 'text'}_{lm_size_for_experiment}"
    weave.init(f"ExperimentOne-noise-{exp_description}") 

    # Create agents
    agents = []
    final_model_ids = []
    if model_type == "api":
        for model_id in api_model_ids:
            agents.append(Agent(f"LanguageAgent", model_id))
            agents.append(Agent(f"LanguageDEFAgent", model_id))
            agents.append(Agent(f"LanguageFunctionAgent", model_id))
            final_model_ids.append(model_id)
    elif model_type == "together":
        for model_id in together_model_ids:
            agents.append(Agent(f"LanguageAgent", model_id))
            agents.append(Agent(f"LanguageDEFAgent", model_id))
            agents.append(Agent(f"LanguageFunctionAgent", model_id))
            final_model_ids.append(model_id)
        if vision:
            for model_id in together_vision_model_ids:
                agents.append(Agent(f"VisionLanguageAgent", model_id))
                agents.append(Agent(f"VisionLanguageDEFAgent", model_id))
                agents.append(Agent(f"VisionLanguageFunctionAgent", model_id))
                final_model_ids.append(model_id)
    elif model_type == "local":
        for model_id in local_model_ids:
            agents.append(Agent(f"LanguageAgent", model_id))
            agents.append(Agent(f"LanguageDEFAgent", model_id))
            agents.append(Agent(f"LanguageFunctionAgent", model_id))
            final_model_ids.append(model_id)
        if vision:
            for model_id in local_vision_model_ids:
                agents.append(Agent(f"VisionLanguageAgent", model_id))
                agents.append(Agent(f"VisionLanguageDEFAgent", model_id))
                agents.append(Agent(f"VisionLanguageFunctionAgent", model_id))
                final_model_ids.append(model_id)
    elif model_type == "custom":
        for model_id in custom_model_ids:
            agents.append(Agent(f"LanguageAgent", model_id))
            agents.append(Agent(f"LanguageDEFAgent", model_id))
            agents.append(Agent(f"LanguageFunctionAgent", model_id))
            final_model_ids.append(model_id)

    # Add baseline agents
    agents.extend([
        Agent("FunctionAgent"),
        Agent("PoolMasterAgent")
        ])
    
    total_matchups = len(NOISE_LEVELS.keys()) * (len(agents)**2 - len(agents))
    start_idx = int((total_matchups * matchup_index / matchup_splits) if matchup_index >= 0 else 0)
    end_idx = int((total_matchups * (matchup_index + 1) / matchup_splits) if matchup_index >= 0 else total_matchups)

    def completed_matchup_trials(results, noise_level, key):
        results = results.get(noise_level, {})
        if not key in results:
            return 0
        return len(results[key]['games'])

    matchups = []
    index = 0
    for noise_level in NOISE_LEVELS.keys():
        for i, a1 in enumerate(agents):
            for j, a2 in enumerate(agents):
                if i != j:
                    if index >= start_idx and index < end_idx:
                        key = gen_key(a1, a2)
                        matchup_trials = num_trials - completed_matchup_trials(manager.results, noise_level, key)
                        if matchup_trials > 0:
                            matchup = Matchup(key, a1, a2, noise_level, num_trials=matchup_trials)
                            matchups.append(matchup)
                    index += 1

    print(f"Running {len(matchups)}/{total_matchups} matchups")
    
    # Run matchups sequentially
    for i, matchup in enumerate(matchups):

        manager.logger.info(f"\nRunning matchup {i}/{len(matchups)}")
        results = manager.run_matchup(matchup)
        manager.print_state(i, len(matchups))
    
        # Save final results
        with open(f"{manager.log_dir}/all_results.json", "w") as f:
            json.dump(manager.results, f, indent=4)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Run sequential simulation with API or local models")
    parser.add_argument("--model_type", choices=["api", "together", "local", "custom"], help="Choose 'api' for GPT models, 'local' or 'together' for Hugging Face models, or 'custom' for a custom list of models", default="together")
    parser.add_argument("--vision", action="store_true", help="Use vision models", default=False)
    parser.add_argument("--num_trials", type=int, default=5, help="Number of trials to run for each matchup")
    parser.add_argument("--gpu_size", type=int, help="Size in GB of the GPU in use", default=40)
    parser.add_argument("--n_gpus", type=int, default=0, help="Number of GPUs to use")
    parser.add_argument("--splits", type=int, default=1, help="Number of splits to run")
    parser.add_argument("--index", type=int, default=-1, help="Index of matchups to run i.e. index/splits")
    args = parser.parse_args()

    if args.model_type == "local":
        from huggingface_hub import login
        assert "HUGGINGFACE_TOKEN" in os.environ, "Please set the HUGGINGFACE_TOKEN environment variable"
        login(os.environ["HUGGINGFACE_TOKEN"])

    experiments_config = data = yaml.safe_load(open(ROOT_DIR + '/experiments/experiments_config.yaml'))
    
    run_experiment(experiments_config, args)