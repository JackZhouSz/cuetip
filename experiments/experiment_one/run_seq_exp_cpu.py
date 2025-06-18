import argparse, sys, os, logging, time, dspy, traceback, json, re, dotenv, torch
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from poolagent.path import ROOT_DIR
from poolagent.agents import *
AGENTS = {
    "ProRandomBallAgent": ProRandomBallAgent,
    "LanguageAgent": LanguageAgent,
    "LanguageFunctionAgent": LanguageFunctionAgent,
    "VisionLanguageAgent": VisionLanguageAgent,
    "VisionLanguageFunctionAgent": VisionLanguageFunctionAgent,
    "FunctionAgent": FunctionAgent
}
from poolagent.pool import PoolGame
from poolagent.utils import blur_shot, SKILL_LEVELS

# Constants
GPU_SIZE = 40
MAX_MODEL_SIZE_FOR_SINGLE_GPU = GPU_SIZE // 2
PARALLEL = False

MODELS = {
    'api': [
        "gpt-4o-mini",
        "gpt-4o",
    ],
    'local': {
        "vision": [
            ("llava-hf/llava-v1.6-mistral-7b-hf", "SMALL"),
            ("microsoft/Phi-3.5-vision-instruct", "SMALL"),
            ("llava-hf/llava-v1.6-vicuna-13b-hf", "MEDIUM"),
            ("llava-hf/llava-v1.6-34b-hf", "MEDIUM"),
            ("meta-llama/Llama-3.2-11B-Vision-Instruct", "MEDIUM"),
            ("meta-llama/Llama-3.2-90B-Vision-Instruct", "LARGE"),
        ],
        "text": [
            ("meta-llama/Llama-3.2-3B-Instruct", "SMALL"),
            ("meta-llama/Llama-3.1-8B-Instruct", "MEDIUM"),
            ("meta-llama/Llama-3.1-70B-Instruct", "LARGE"),
        ]
    }
}

class LLM:
    def __init__(self, model_id, gpu_ids=None, gpu_size=40, temperature=0.0, max_tokens=4096):
        self.model_id = model_id
        self.gpu_ids = gpu_ids
        self.gpu_size = gpu_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = self.setup_model_dspy()
        
    def setup_model_dspy(self):
        if not self.gpu_ids:
            # (Azure) OpenAI setup
            dotenv.load_dotenv()
            assert os.getenv("API_KEY") is not None, "API_KEY not found in .env file"
            use_azure = os.getenv("API_BASE") is not None

            config = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            if use_azure:
                #, "API_BASE not found in .env file"
            
                llm = dspy.AzureOpenAI(
                    model=self.model_id,
                    api_base=os.getenv("API_BASE"),
                    api_version='2024-06-01',
                    api_key=os.getenv("API_KEY"),
                    **config,
                )
            else:
                llm = dspy.LM(
                    model=f"openai/{self.model_id}",
                    api_key=os.getenv("API_KEY"),
                    **config,
                )
        else:
            # Hugging Face setup
            gpu_idx = [int(gpu_id.split("_")[1]) - 1 for gpu_id in self.gpu_ids]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_idx))
            
            llm = dspy.HFModel(
                model=self.model_id,
                hf_device_map="auto",
                model_kwargs={
                    "do_sample": False,
                    "torch_dtype": torch.float16,
                }
            )

        return llm
    
    def delete(self):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        del self.llm

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
    def __init__(self, description, agent_one, agent_two):
        models = [model_id for model_id in [agent_one.model_id, agent_two.model_id] if model_id]
        super().__init__(description, models)
        self.agent_one = agent_one
        self.agent_two = agent_two

class SequentialExperimentManager:
    def __init__(self, experiment_name, gpu_ids, gpu_size=40, temperature=0.0, max_tokens=4096):
        self.experiment_name = experiment_name
        self.gpu_size = gpu_size
        self.gpu_ids = gpu_ids
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.console = Console()
        self.results = {}
        self.env = PoolGame()
        
        # Setup logging
        self.log_dir = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/{experiment_name}/logs/{self.timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = self.setup_logging()
        
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
    
    def get_model_size(self, model_id):
        match = re.search(r'-(\d+)b', model_id.lower())
        return int(match.group(1)) if match else 0
    
    def get_required_gpus(self, model_size):
        return 1 if model_size < MAX_MODEL_SIZE_FOR_SINGLE_GPU else (model_size - 1) // MAX_MODEL_SIZE_FOR_SINGLE_GPU + 1
    
    def is_api_model(self, model_id):
        return '/' not in model_id
    
    def gen_key(self, agent_one, agent_two):

        model_id_one = agent_one.model_id
        if model_id_one:
            model_id_one = agent_one.model_id.split('/')[-1] if '/' in model_id_one else model_id_one

        model_id_two = agent_two.model_id
        if model_id_two:
            model_id_two = agent_two.model_id.split('/')[-1] if '/' in model_id_two else model_id_two

        return f"{agent_one.name}_{model_id_one}---{agent_two.name}_{model_id_two}"

    def bot_shot(self, env, state, player):
        return player.take_shot(env, state)
    def language_agent_shot(self, env, state, player, logger, lm):
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
        elif isinstance(player, LanguageFunctionAgent):
            return self.language_function_agent_shot(env, state, player, logger, lm)
        elif isinstance(player, FunctionAgent):
            return self.function_agent_shot(env, state, player, logger)
        else:
            logger.error(f"Unknown agent type: {player.__class__.__name__}")
            return None

    def load_model(self, model_id):
        if self.is_api_model(model_id):
            return LLM(model_id, temperature=self.temperature, max_tokens=self.max_tokens)
        
        model_size = self.get_model_size(model_id)
        required_gpus = self.get_required_gpus(model_size)
        assigned_gpus = self.gpu_ids[:required_gpus]
        
        return LLM(model_id, assigned_gpus, self.gpu_size)
    
    def run_game(self, matchup, game_num):
        env = self.env

        self.logger.info(f"Starting matchup simulation: {matchup.agent_one.name} vs {matchup.agent_two.name}")

        llm_map = {
            'one': matchup.assigned_llms.get(matchup.agent_one.model_id, None),
            'two': matchup.assigned_llms.get(matchup.agent_two.model_id, None)
        }

        matchup_key = self.gen_key(matchup.agent_one, matchup.agent_two)
        matchup_dir = os.path.join(ROOT_DIR, "experiments", "experiment_one", "logs", self.timestamp, "tasks", matchup_key)
        os.makedirs(matchup_dir, exist_ok=True)

        m_agent_one = matchup.agent_one
        m_agent_two = matchup.agent_two
        agent_one = AGENTS[m_agent_one.name](target_balls=['red', 'blue', 'yellow'])
        agent_two = AGENTS[m_agent_two.name](target_balls=['black', 'pink', 'green'])

        players = {
            'one': agent_one,
            'two': agent_two
        }

        self.logger.info(f"Starting matchup simulation: {m_agent_one.name} vs {m_agent_two.name}")

        rewards = []
        starting_state = env.get_state().copy()

        env.from_state(starting_state)
        state = env.get_state()

        env.current_player = 'one'
        env.double_shot = False

        turn_count = 0
        game_log = {}
        MAX_TURNS = 20
        ended_in_draw = False

        while not env.check_win():
            turn_count += 1

            if turn_count > MAX_TURNS:
                self.logger.info(f"Game {game_num + 1} finished in draw after {MAX_TURNS} turns, no winner")
                ended_in_draw = True
                break

            current_settings = {
                'current_player': env.current_player,
                'double_shot': env.double_shot
            }

            player = players[current_settings['current_player']]
            current_llm = llm_map[current_settings['current_player']]

            self.logger.info(f"Turn {turn_count}: Player {current_settings['current_player']} ({player.__class__.__name__})")

            max_retries = 3
            for retry in range(max_retries):
                try:
                    self.logger.info(f"Taking shot for {current_settings['current_player']}")
                    start_time = time.time()
                    shot = self.get_agent_shot(env, state, player, current_settings, self.logger, current_llm)
                    end_time = time.time()
                    self.logger.info(f"Shot taken in {end_time - start_time:.2f} seconds")
                    
                    env.from_state(state)
                    env.current_player = current_settings['current_player']
                    env.double_shot = current_settings['double_shot']
                    env.take_shot(current_settings['current_player'], blur_shot(shot, skill=SKILL_LEVELS.BASELINE))
                    self.logger.info(f"Shot taken in environment")
                    
                    game_log[f'shot_{turn_count}'] = {
                        'player': current_settings['current_player'],
                        'shot': shot,
                        'double_shot': current_settings['double_shot'],
                        'llm': current_llm.model_id if current_llm else None,
                        'gpu_ids': current_llm.gpu_ids if current_llm else None,
                        'execution_time': end_time - start_time,
                        **player.record
                    }

                    self.logger.info(f"Shot taken: {shot}")
                    events = env.get_events()
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
                        env.current_player = 'one' if env.current_player == 'two' else 'two'
                        env.double_shot = True

        game_reward = env.reward()[0] if not ended_in_draw else 0.5

        self.logger.info(f"Game {game_num + 1} finished. Reward: {game_reward}")

        game_log_file = os.path.join(matchup_dir, f"game_{game_num + 1}.json")
        with open(game_log_file, 'w') as f:
            json.dump(game_log, f, indent=2)

        self.save_results(matchup_dir, rewards)

        return game_reward
    
    def run_matchup(self, matchup, n_games=10):
        self.logger.info(f"Starting matchup: {matchup.description}")
        
        # Load models for the matchup
        for model_id in matchup.models:
            if model_id:
                matchup.assigned_llms[model_id] = self.load_model(model_id)
        
        matchup_dir = f"{self.log_dir}/tasks/{matchup.description}"
        os.makedirs(matchup_dir, exist_ok=True)
        
        rewards = []
        for game_num in range(n_games):
            self.logger.info(f"Starting game {game_num + 1}/{n_games}")
            
            try:
                game_result = self.run_game(matchup, game_num)
                rewards.append(game_result)
                
                # Save game results
                with open(f"{matchup_dir}/game_{game_num + 1}.json", 'w') as f:
                    json.dump({
                        'game_number': game_num + 1,
                        'result': game_result
                    }, f, indent=2)
                    
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

def run_experiment(model_type, num_gpus, gpu_size, vision=False, temperature=0.0, max_tokens=4096):
    assert model_type in ["api", "local"], "Invalid model type. Choose 'api' or 'local'."
    
    # Initialize experiment manager
    gpu_ids = [f"gpu_{i}" for i in range(1, num_gpus+1)]
    manager = SequentialExperimentManager("experiment_one", gpu_ids, gpu_size, temperature=temperature, max_tokens=max_tokens)
    
    # Create agents based on model type
    agents = []
    if model_type == "api":
        for model_id in MODELS[model_type]:
            agents.append(Agent("LanguageAgent", model_id))
            agents.append(Agent("LanguageFunctionAgent", model_id))
    elif model_type == "local":
        size = os.getenv("SIZE", "MEDIUM")
        for model_id, model_size in MODELS[model_type]['text']:
            if model_size == size:
                agents.append(Agent("LanguageAgent", model_id))
                agents.append(Agent("LanguageFunctionAgent", model_id))
        if vision:
            for model_id, model_size in MODELS[model_type]['vision']:
                if model_size == size:
                    agents.append(Agent("VisionLanguageAgent", model_id))
                    agents.append(Agent("VisionLanguageFunctionAgent", model_id))
    
    # Add baseline agents
    agents.extend([
        Agent("ProRandomBallAgent"),
        Agent("FunctionAgent")
    ])
    
    # Create all possible matchups
    matchups = [
        Matchup(f"{a1.name}_{a1.model_id}---{a2.name}_{a2.model_id}", a1, a2)
        for i, a1 in enumerate(agents)
        for j, a2 in enumerate(agents)
        if i != j
    ]
    
    # Run matchups sequentially
    for i, matchup in enumerate(matchups, 1):
        manager.logger.info(f"\nRunning matchup {i}/{len(matchups)}")
        results = manager.run_matchup(matchup, n_games=10)
        manager.print_state(i, len(matchups))
    
    # Save final results
    with open(f"{manager.log_dir}/all_results.json", "w") as f:
        json.dump(manager.results, f, indent=4)

if __name__ == "__main__":
    from huggingface_hub import login
    assert "HUGGINGFACE_TOKEN" in os.environ, "Please set the HUGGINGFACE_TOKEN environment variable"
    login(os.environ["HUGGINGFACE_TOKEN"])
    
    parser = argparse.ArgumentParser(description="Run sequential simulation with API or local models")
    parser.add_argument("model_type", choices=["api", "local"], help="Choose 'api' for GPT models or 'local' for Hugging Face models")
    parser.add_argument("--vision", action="store_true", help="Use vision models", default=False)
    parser.add_argument("--n_gpus", type=int, default=0, help="Number of GPUs to use")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature to use LLMs with.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max number of tokens to use when generating tokens with LLMs.")
    args = parser.parse_args()
    
    run_experiment(args.model_type, args.n_gpus, 0, vision=args.vision)
