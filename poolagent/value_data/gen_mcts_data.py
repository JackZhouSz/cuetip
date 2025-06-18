import argparse, random
import numpy as np 

from poolagent.pool import PoolGame
from poolagent.mcts import run_multiple_mcts
from poolagent.utils import TwoPlayerState, SKILL_LEVELS, SHOT_PARAM_VALUES

def gaussian_sample(param_config):
    return np.clip(
        np.random.normal(param_config["mean"], param_config["std"]),
        param_config["min"],
        param_config["max"]
    )

def random_shot():
    return {
        "V0": gaussian_sample(SHOT_PARAM_VALUES["V0"]),
        "theta": gaussian_sample(SHOT_PARAM_VALUES["theta"]),
        "phi": np.random.uniform(SHOT_PARAM_VALUES["phi"]["min"], SHOT_PARAM_VALUES["phi"]["max"]),
        "a": gaussian_sample(SHOT_PARAM_VALUES["a"]),
        "b": gaussian_sample(SHOT_PARAM_VALUES["b"])
    }

def propose_shot(game: PoolGame, eps=0.2, skill_level=SKILL_LEVELS.AMATEUR):
    if np.random.random() < eps:
        return random_shot()

    state = game.get_state()
    target_balls = game.target_balls[game.current_player]
    target_balls = [ball for ball in target_balls if not state.is_potted(ball)]

    if len(target_balls) == 0:
        return random_shot()

    ball = np.random.choice(target_balls)
    phi = game.attempt_pot_ball(ball)

    if phi == -1:
        return random_shot()

    shot = random_shot()
    
    # Adjust phi based on the attempted pot
    shot["phi"] = np.random.normal(phi, skill_level['phi'])
    shot["phi"] = shot["phi"] % 360

    return shot

if __name__ == "__main__":

    ### Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, help="Number of MCTS runs.", default=1000)
    parser.add_argument("--it", type=int, help="Number of iterations for each MCTS run.", default=500)
    parser.add_argument("--branch", type=int, help="Number of branches for each MCTS run.", default=10)
    args = parser.parse_args()

    game = PoolGame()
    num_runs = args.runs
    iterations = args.it

    np.random.seed(random.randint(0, 10000))
    states = [
        TwoPlayerState().full_randomize() for _ in range(num_runs)
    ]

    dataset = run_multiple_mcts(states, game, propose_shot, iterations, branching_factor=args.branch)

