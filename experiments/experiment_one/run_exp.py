import argparse, sys, json, os, logging, time, dspy, traceback, yaml

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poolagent.agents import *
AGENTS = {
    "ProRandomBallAgent": ProRandomBallAgent,
    "LanguageAgent": LanguageAgent,
    "LanguageFunctionAgent": LanguageFunctionAgent,
    "VisionLanguageAgent": VisionLanguageAgent,
    "VisionLanguageFunctionAgent": VisionLanguageFunctionAgent,
    "FunctionAgent": FunctionAgent
}
from poolagent.path import ROOT_DIR
from poolagent.experiment_manager import ExperimentManager, Experiment, Task
from poolagent.pool import PoolGame
from poolagent.utils import blur_shot

### Specific Experiment and Task ###
   
class Agent:
    def __init__(self, name, model_id=None):
        self.name = name
        self.model_id = model_id

class Matchup(Task):
    def __init__(self, description, agent_one, agent_two):
        models = [model_id for model_id in [agent_one.model_id, agent_two.model_id] if model_id != ""]
        super().__init__(description, models)

        self.agent_one = agent_one
        self.agent_two = agent_two

class MatchupManager(Experiment):
    def __init__(self, agents, num_threads, num_games=5, eidf_save_intermediate_results=False, parallel_optimsation=False):
        super().__init__()
        self.num_games = num_games
        self.tasks = [Matchup( self.gen_key(agents[i], agents[j]), agents[i], agents[j]) for i in range(len(agents)) for j in range(len(agents)) if i != j]
        self.previous_results = self.get_previous_results()
        self.current_tasks = {}
        self.environments = [PoolGame() for _ in range(num_threads)]
        self.matchup_times = {}
        self.eidf_save_intermediate_results = eidf_save_intermediate_results
        self.parallel = parallel_optimsation

    def get_previous_results(self):
        """
        Load previous results from file.

        Returns:
            dict: Previous results, or None if no results file exists.
        """
        results_files = [f for f in os.listdir(f"{ROOT_DIR}/experiments/experiment_one/results/") if f.endswith(".json")]
        if not results_files:
            return
        recent_results_file = max(results_files)
        with open(f"{ROOT_DIR}/experiments/experiment_one/results/{recent_results_file}", "r") as f:
            previous_results = json.load(f)

        filtered_matchups = []
        for matchup in self.tasks:
            matchup_key = self.gen_key(matchup.agent_one, matchup.agent_two)
            if matchup_key not in previous_results:
                filtered_matchups.append(matchup)
            elif len(previous_results[matchup_key]["games"]) < self.num_games:
                filtered_matchups.append(matchup)
            else:
                continue    

        self.tasks = filtered_matchups

        return previous_results

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
    def language_agent_shot(self, env, state, player, logger, lm, parallel=False):
        return player.take_shot(env, state, lm.llm, logger=logger, parallel=parallel)
    def language_function_agent_shot(self, env, state, player, logger, lm, parallel=False):
        return player.take_shot(env, state, lm.llm, logger=logger, parallel=parallel)
    def function_agent_shot(self, env, state, player, logger, parallel=False):
        return player.take_shot(env, state, logger=logger, parallel=parallel)

    def get_agent_shot(self, env, state, player, current_settings, logger, lm, parallel):
        if isinstance(player, ProRandomBallAgent):
            return self.bot_shot(env, state, player)
        elif isinstance(player, LanguageAgent):
            return self.language_agent_shot(env, state, player, logger, lm, parallel=parallel)
        elif isinstance(player, LanguageFunctionAgent):
            return self.language_function_agent_shot(env, state, player, logger, lm, parallel=parallel)
        elif isinstance(player, FunctionAgent):
            return self.function_agent_shot(env, state, player, logger, parallel=parallel)
        else:
            logger.error(f"Unknown agent type: {player.__class__.__name__}")
            return None

    def save_results(self, matchup_dir, rewards):
        results_file = os.path.join(matchup_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump({
                "games": rewards,
                "winrate": sum(rewards) / len(rewards)
            }, f, indent=2)

        # Save intermediate results to persistent storage, EIDF ONLY
        if self.eidf_save_intermediate_results:
            # Copy logs folder to persistent storage, EIDF ONLY
            logs_dir = os.path.join(ROOT_DIR, "experiments", "experiment_one", "logs")
            logs_dest = os.path.join("/mnt/ceph_rbd", "PoolAgent", "src", "experiment_one", "logs") 
            os.makedirs(logs_dest, exist_ok=True)
            try:
                os.system(f"cp -r {logs_dir} {logs_dest}")
            except Exception as e:
                print(f"Error copying logs to persistent storage: {str(e)}")    

    def run_task(self, thread_id, matchup, timestamp, N=25, logger=None, max_retries=3):
        env = self.environments[thread_id]

        if not logger:
            logger = logging.getLogger(f"Thread-{thread_id}")

        logger.info(f"Thread ID: {thread_id}, Number of games: {N}")
        logger.info(f"Starting matchup simulation: {matchup.agent_one.name} vs {matchup.agent_two.name}")

        llm_map = {
            'one': matchup.assigned_llms.get(matchup.agent_one.model_id, None),
            'two': matchup.assigned_llms.get(matchup.agent_two.model_id, None)
        }

        matchup_key = self.gen_key(matchup.agent_one, matchup.agent_two)
        matchup_dir = os.path.join(ROOT_DIR, "experiments", "experiment_one", "logs", timestamp, "tasks", matchup_key)
        os.makedirs(matchup_dir, exist_ok=True)

        self.current_tasks[matchup_key] = []

        m_agent_one = matchup.agent_one
        m_agent_two = matchup.agent_two
        agent_one = AGENTS[m_agent_one.name](target_balls=['red', 'blue', 'yellow'])
        agent_two = AGENTS[m_agent_two.name](target_balls=['black', 'pink', 'green'])

        players = {
            'one': agent_one,
            'two': agent_two
        }

        logger.info(f"Starting matchup simulation: {m_agent_one.name} vs {m_agent_two.name}")
        logger.info(f"Thread ID: {thread_id}, Number of games: {N}")

        rewards = []
        env.reset()
        starting_state = env.get_state().copy()

        for game_num in range(N):
            logger.info(f"Starting game {game_num + 1}/{N}")

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
                    logger.info(f"Game {game_num + 1} finished in draw after {MAX_TURNS} turns, no winner")
                    ended_in_draw = True
                    break

                current_settings = {
                    'current_player': env.current_player,
                    'double_shot': env.double_shot
                }

                player = players[current_settings['current_player']]
                current_llm = llm_map[current_settings['current_player']]

                logger.info(f"Turn {turn_count}: Player {current_settings['current_player']} ({player.__class__.__name__})")

                for retry in range(max_retries):
                    try:
                        logger.info(f"Taking shot for {current_settings['current_player']}")
                        start_time = time.time()
                        shot = self.get_agent_shot(env, state, player, current_settings, logger, current_llm, self.parallel)
                        end_time = time.time()
                        logger.info(f"Shot taken in {end_time - start_time:.2f} seconds")
                        
                        env.from_state(state)
                        env.current_player = current_settings['current_player']
                        env.double_shot = current_settings['double_shot']
                        env.take_shot(current_settings['current_player'], blur_shot(shot, skill=SKILL_LEVELS.BASELINE))
                        logger.info(f"Shot taken in environment")
                        
                        game_log[f'shot_{turn_count}'] = {
                            'player': current_settings['current_player'],
                            'shot': shot,
                            'double_shot': current_settings['double_shot'],
                            'llm': current_llm.model_id if current_llm else None,
                            'gpu_ids': current_llm.gpu_ids if current_llm else None,
                            'execution_time': end_time - start_time,
                            **player.record
                        }

                        logger.info(f"Shot taken: {shot}")
                        events = env.get_events()
                        events_str = ", ".join([e.encoding for e in events])
                        logger.info(f"  Events: {events_str}")
                        break
                    except Exception as e:
                        logger.error(f"Error in {current_settings['current_player']}: {str(e)}")
                        logger.error(traceback.format_exc())
                        if retry < max_retries - 1:
                            logger.warning(f"Error in {current_settings['current_player']} - retrying")
                            time.sleep(1)  # Wait a bit before retrying
                        else:
                            logger.error(f"Error in {current_settings['current_player']} - failed after {max_retries} attempts")
                            logger.error(traceback.format_exc())
                            game_log[f'shot_{turn_count}'] = {
                                'player': current_settings['current_player'],
                                'error': str(e)
                            }
                            env.current_player = 'one' if env.current_player == 'two' else 'two'
                            env.double_shot = True

                state = env.get_state()

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

            game_reward = env.reward()[0] if not ended_in_draw else calc_draw_reward(env)
            rewards.append(game_reward)

            logger.info(f"Game {game_num + 1} finished. Reward: {game_reward}")

            self.current_tasks[matchup_key].append(game_reward)

            game_log_file = os.path.join(matchup_dir, f"game_{game_num + 1}.json")
            with open(game_log_file, 'w') as f:
                json.dump(game_log, f, indent=2)

            self.save_results(matchup_dir, rewards)

        average_reward = sum(rewards) / N
        logger.info(f"Matchup simulation completed. Average reward: {average_reward}")

        del self.current_tasks[matchup_key] 

        self.save_results(matchup_dir, rewards)

        return rewards

### Specific Experiment and Task ###

def run_experiment(experiments_config, model_type, num_gpus, n_threads, gpu_size, vision=False, eidf=False):
    # Define models based on the chosen type
    assert model_type in ["api", "local", "together"], "Invalid model type. Choose 'api', 'together', or 'local'."
    
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

    gpu_ids = [f"gpu_{i}" for i in range(1, num_gpus+1)]

    # Create agents
    agents = []
    final_model_ids = []
    if model_type == "api":
        for model_id in api_model_ids:
            agents.append(Agent(f"LanguageAgent", model_id))
            agents.append(Agent(f"LanguageFunctionAgent", model_id))
            final_model_ids.append(model_id)
    elif model_type == "together":
        for model_id in together_model_ids:
            agents.append(Agent(f"LanguageAgent", model_id))
            agents.append(Agent(f"LanguageFunctionAgent", model_id))
            final_model_ids.append(model_id)
        if vision:
            for model_id in together_vision_model_ids:
                agents.append(Agent(f"VisionLanguageAgent", model_id))
                agents.append(Agent(f"VisionLanguageFunctionAgent", model_id))
                final_model_ids.append(model_id)
    elif model_type == "local":
        for model_id in local_model_ids:
            agents.append(Agent(f"LanguageAgent", model_id))
            agents.append(Agent(f"LanguageFunctionAgent", model_id))
            final_model_ids.append(model_id)
        if vision:
            for model_id in local_vision_model_ids:
                agents.append(Agent(f"VisionLanguageAgent", model_id))
                agents.append(Agent(f"VisionLanguageFunctionAgent", model_id))
                final_model_ids.append(model_id)

    # Add some agents without model_ids
    agents.extend([
        Agent("ProRandomBallAgent"),
        Agent("FunctionAgent")
    ])

    # Initialize ExpManager
    n_games = 5
    experiment = MatchupManager(
        agents, 
        num_threads=n_threads, 
        num_games=n_games, 
        eidf_save_intermediate_results=eidf, 
        parallel_optimsation=experiments_config.get("parallel_optimsation", False)
    )
    exp_manager = ExperimentManager("experiment_one", final_model_ids, gpu_ids, experiment, max_concurrent_threads=n_threads, N=n_games, gpu_size=gpu_size)

    # Run the experiment
    exp_manager.run()

    # Once finished copy the final set of results to the data directory
    with open(f"{ROOT_DIR}/experiments/experiment_one/results/{exp_manager.timestamp}.json", "w") as f:
        json.dump(experiment.results, f, indent=4)

if __name__ == "__main__":

    from huggingface_hub import login
    assert "HUGGINGFACE_TOKEN" in os.environ, "Please set the Hugging Face token as an environment variable"
    HF_TOKEN = os.environ["HUGGINGFACE_TOKEN"]
    login(HF_TOKEN)

    parser = argparse.ArgumentParser(description="Run simulation with API or local models")
    parser.add_argument("model_type", choices=["api", "together", "local"], help="Choose 'api' for GPT models, 'local' or 'together' for Hugging Face models")
    parser.add_argument("--vision", action="store_true", help="Use vision models", default=False)
    parser.add_argument("--gpu_size", type=int, help="Size in GB of the GPU in use", default=40)
    parser.add_argument("--n_gpus", type=int, default=3, help="Number of GPUs to use")
    parser.add_argument("--n_threads", type=int, default=3, help="Number of concurrent threads")    
    parser.add_argument("--eidf", action="store_true", help="Save intermediate results if on EIDF", default=False)
    args = parser.parse_args()

    experiments_config = yaml.safe_load(open(ROOT_DIR + '/experiments/experiments_config.yaml'))

    run_experiment(experiments_config, args.model_type, args.n_gpus, args.n_threads, args.gpu_size, vision=args.vision, eidf=args.eidf)