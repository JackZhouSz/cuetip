import yaml
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import weave

from poolagent import (
    ProRandomBallAgent,
    PoolMasterAgent,
    TwoPlayerState,
    PoolGame,
    VISUALISATIONS_DIR,
    blur_shot,
    SKILL_LEVELS
)

### Set Seed
seed =  np.random.randint(0, 1000)
np.random.seed(seed)

env = PoolGame()
num_ball_level = 4
init_state = TwoPlayerState(num_ball_level=num_ball_level).randomize()
env.from_state(init_state)
player_one_balls = [str(i) for i in range(1, num_ball_level+1)]
player_two_balls = [str(i) for i in range(9, num_ball_level+9)]
target_balls = {
    'one': player_one_balls,
    'two': player_two_balls
}
env.target_balls = target_balls
N_GAMES = 100

agents = {
    "one": ProRandomBallAgent(player_one_balls),
    "two": PoolMasterAgent(player_two_balls),
}

weave.init(f"Demo-test")

@weave.op()
def take_shot(xai_agent, env, state, llm, parallel=False, message=None):
    return xai_agent.take_shot(env, state, llm, parallel=parallel, message=message)

# env.get_image().show_image()

winning_games = 0
for g in range(N_GAMES):

    print("-"*50)
    print(f"Game {g+1}/{N_GAMES}")

    env.reset()
    env.from_state(init_state)
    c = 0
    while not env.check_win():
        c+=1
        state = env.get_state()
        shot_taker = env.current_player
        shot = agents[shot_taker].take_shot(env, state)
        blurred_shot = blur_shot(shot, skill=SKILL_LEVELS.BASELINE)
        #env.save_shot_gif(state, blurred_shot, f'tmp_{g}_{c}.gif')
        env.from_state(state)
        env.take_shot(shot_taker, blurred_shot)
        print(f"Player {shot_taker} --> {[e.encoding for e in env.get_events()]}")

    winning_games += env.reward()[0]
    print(f"Winrate: {winning_games/(g+1)}")
