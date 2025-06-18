import sys
import json
import numpy as np
from scipy import stats

from poolagent.pool import PoolGame, Fouls
from poolagent.utils import TwoPlayerState, SKILL_LEVELS, blur_shot
from poolagent.value_data.gen_mcts_data import propose_shot

ROLL_OUTS = 1000

class PoolAnalysisWorker:
    def __init__(self, n_iterations, output_file, n_estimates=25, noise=SKILL_LEVELS.AMATEUR, target_balls=['red', 'yellow', 'blue']):
        self.game = PoolGame()
        self.n_iterations = n_iterations
        self.output_file = output_file
        self.n_estimates = n_estimates
        self.noise = noise
        self.target_balls = target_balls

    def run_analysis(self):
        all_data = []
        for i in range(self.n_iterations):
            print(f"Processing iteration {i+1}/{self.n_iterations}")
            state, action, initial_estimate, estimates = self._generate_single_dataset()
            
            mean = np.mean(estimates)
            std = np.std(estimates)
            iqr = stats.iqr(estimates)
            all_data.append({
                'state': state.to_json(),
                'action': action,
                'initial_estimate': float(initial_estimate),
                'estimates': estimates.tolist(),
                'mean': float(mean),
                'std': float(std),
                'iqr': float(iqr)
            })

            with open(self.output_file, 'w') as f:
                json.dump(all_data, f, indent=4)

    def _generate_single_dataset(self):
        state = TwoPlayerState().full_randomize()
        self.game.from_state(state)

        action = self._find_valid_shot(state)
        initial_estimate = self._get_initial_estimate(state)
        estimates = self._generate_estimates(state, action)

        return state, action, initial_estimate, estimates

    def _find_valid_shot(self, state):
        chance_of_ball_pot_shot = 0.3

        if np.random.rand() < chance_of_ball_pot_shot:
            while True:
                self.game.current_player = 'one'
                self.game.double_shot = False
                self.game.from_state(state)
                action = propose_shot(self.game, eps=0, skill_level=SKILL_LEVELS.NONE)
                self.game.strike(**action)
                events = self.game.get_events()
                if any('ball-pocket' in e.encoding and any(b in e.encoding for b in self.target_balls) for e in events):
                    return action
        else:
            while True:
                self.game.current_player = 'one'
                self.game.double_shot = False
                self.game.from_state(state)
                action = propose_shot(self.game, eps=0, skill_level=SKILL_LEVELS.NONE)
                foul = self.game.strike(**action, check_rules=True, target_balls=self.target_balls)
                if foul != Fouls.NONE:
                    return action

    def _get_initial_estimate(self, state):
        self.game.current_player = 'one'
        self.game.double_shot = False
        self.game.from_state(state)
        return self.game.get_value_estimate(self._propose_shot_no_noise, initial_roll_outs=ROLL_OUTS)

    def _generate_estimates(self, state, action):
        estimates = []
        for _ in range(self.n_estimates):
            self.game.current_player = 'one'
            self.game.double_shot = False
            shot = blur_shot(action, self.noise)
            self.game.from_state(state)
            self.game.take_shot(self.game.current_player, shot)
            estimate = self.game.get_value_estimate(self._propose_shot_no_noise)
            estimates.append(estimate)

            # Print the estimate to update the manager
            print(f"Estimate: {estimate}")

        return np.array(estimates)

    @staticmethod
    def _propose_shot_no_noise(game):
        return propose_shot(game, eps=0, skill_level=SKILL_LEVELS.PRO)

if __name__ == "__main__":
    n_iterations = int(sys.argv[1])
    output_file = sys.argv[2]
    worker = PoolAnalysisWorker(n_iterations, output_file)
    worker.run_analysis()
