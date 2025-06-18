import numpy as np
import time
from typing import List
import os

from poolagent.path import ROOT_DIR
from poolagent.pool import Pool, Fouls
from poolagent.utils import SKILL_LEVELS, State, random_ball_shot, Agent, Event, blur_shot
from poolagent.mcts import MCTS, process_dataset

class RandomAgent(Agent):
    def take_shot(self, env : Pool, state : State) -> dict:
        env.from_state(state)
        return env.random_params()
    
class PoolMasterAgent(Agent):

    def __init__(self, target_balls, config={}):
        super().__init__(target_balls, config)
        self.depth = config.get("depth", len(target_balls))
        self.record = {}

    def take_shot(self, env : Pool, state : State) -> dict:
        shot = env.calculate_pool_master_shot(state, target_balls=self.target_balls)

        if shot is None:
            return env.random_params()

        self.record = {
            'agent': 'RandomBallAgent',
            'start_state': state.to_json(),
            'end_state': env.get_state().to_json(),
            'shot': shot,
            'events': [e.to_json() for e in env.get_events()]
        }

        return shot

class BruteForceAgent(Agent):

    def __init__(self, target_balls, config={}):
        super().__init__(target_balls, config)
        self.N = config.get("N", 3)
        self.depth = config.get("depth", len(target_balls))
        self.time_limit = config.get("time_limit", 60)
        self.gamma = config.get("gamma", 0.9)
        self.record = {}

    def check_pocket(self, events: List[Event]) -> bool:
        for e in events:
            if 'ball-pocket' in e.encoding and (not 'cue' in e.encoding):
                return True
        return False
    
    def random_params(self):
        return {
            'V0' : np.random.uniform(0.25, 2),
            'theta' : 14,
            'phi' : np.random.uniform(0, 360),
            'a': 0,
            'b': -0.1,
        }

    def take_shot(self, env: Pool, root_state: State) -> dict:
        def expand_node(state, depth, idx):
            if depth <= 0:
                return [], 0
            
            print(f"Brute Force Agent: Expanding node at depth {depth}")

            # Get possible shots from this state
            shots = self.brute_force(env, state, self.time_limit)
            if not shots:
                return [], 0

            best_path = []
            best_score = -float('inf')

            # Explore each shot's future
            for i, (shot, consistency) in enumerate(shots[:self.N]):
                env.from_state(state)

                ### TEMP
                os.makedirs(f"{ROOT_DIR}/visualisations/demo/depth_{depth}/node_{idx}", exist_ok=True)
                env.save_shot_gif(state, shot, f"{ROOT_DIR}/visualisations/demo/depth_{depth}/node_{idx}/shot_c{consistency}.gif")
                ### TEMP

                env.strike(**shot)
                next_state = env.get_state()
                
                future_path, future_score = expand_node(next_state, depth-1, i)
                total_score = consistency + future_score * self.gamma
                
                if total_score > best_score:
                    best_score = total_score
                    best_path = [shot] + future_path

            return best_path, best_score

        path, score = expand_node(root_state, self.depth, 0)

        print(f"Brute Force Agent: Found path total consistency {score}")

        shot = path[0] if path else self.random_params()
        env.from_state(root_state)
        env.strike(**shot)

        self.record = {
            'agent': 'RandomBallAgent',
            'start_state': root_state.to_json(),
            'end_state': env.get_state().to_json(),
            'shot': shot,
            'events': [e.to_json() for e in env.get_events()]
        }

        return shot

    def brute_force(self, env : Pool, state : State, time_limit: int = 30) -> List[dict]:

        t0 = time.time()
        consistency_N = 100
        shots = []
        total_time = 0
        count = 0

        # Check that there are still target balls on the table
        if all(state.is_potted(ball) for ball in self.target_balls):
            return []

        while total_time < time_limit:
            total_time = time.time() - t0

            env.from_state(state)
            shot = self.random_params()
            foul = env.strike(**shot, check_rules=True, target_balls=self.target_balls)
            count += 1

            if foul != Fouls.NONE:
                continue

            events = env.get_events()
            if self.check_pocket(events):
                consistency = 0
                for _ in range(consistency_N):
                    env.from_state(state)
                    noisy_shot = blur_shot(shot, SKILL_LEVELS.PRO)
                    env.strike(**noisy_shot)
                    events = env.get_events()
                    if self.check_pocket(events):
                        consistency += 1
                consistency /= consistency_N
                shots.append((shot, consistency))

        if len(shots) == 0:
            return []
        
        return sorted(shots, key=lambda x: x[1], reverse=True)[:self.N]

class RandomBallAgent(Agent):
    def take_shot(self, env : Pool, state : State, **kwargs) -> dict:
        env.from_state(state)
        if self.config:
            skill = self.config.get("skill", SKILL_LEVELS.PRO) 
        else:
            skill = SKILL_LEVELS.PRO
        shot = random_ball_shot(env, self.target_balls, skill)
        env.strike(**shot)
        self.record = {
            'agent': 'RandomBallAgent',
            'start_state': state.to_json(),
            'end_state': env.get_state().to_json(),
            'shot': shot,
            'events': [e.to_json() for e in env.get_events()]
        }
        return shot
    
class NoviceRandomBallAgent(RandomBallAgent):
    @staticmethod
    def default_dict():
        return {
            'skill': SKILL_LEVELS.NOVICE
        }
    
class AmateurRandomBallAgent(RandomBallAgent):
    @staticmethod
    def default_dict():
        return {
            'skill': SKILL_LEVELS.AMATEUR
        }
    
class ProRandomBallAgent(RandomBallAgent):
    @staticmethod
    def default_dict():
        return {
            'skill': SKILL_LEVELS.PRO
        }

class RandomBallMCTSAgent(Agent):
    def __init__(self, target_balls, config):
        super().__init__(target_balls, config)

        self.iterations = config.get("iterations", 10)
        self.branching_factor = config.get("branching_factor", 3)

    @staticmethod
    def default_dict():
        return {
            "iterations": 5,
            "branching_factor": 2
        }

    def take_shot(self, env : Pool, state : State) -> dict:
        
        env.from_state(state)
        mcts = MCTS(env, random_ball_shot, iterations=self.iterations, branching_factor=self.branching_factor)
        run_data = mcts.run(state)
        processed_dataset = process_dataset([run_data])
        data = processed_dataset[0]

        max_ind = np.argmax(data["visit_distribution"])
        return data["follow_up_states"][max_ind]['params']
