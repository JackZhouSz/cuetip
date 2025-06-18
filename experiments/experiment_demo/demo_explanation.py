import yaml
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import weave

from poolagent import (
    FunctionAgent,
    TwoPlayerState,
    Pool,
    VISUALISATIONS_DIR,
    explain_shots
)
from poolagent.experiment_manager import LLM


env = Pool()

state = TwoPlayerState().randomize()
target_balls = ['red', 'blue', 'yellow']

agent = FunctionAgent(target_balls)
agent.N = 5
agent.INITIAL_SEARCH = 250
agent.OPT_SEARCH = 50

weave.init(f"Demo-Full-Explanation")

env.from_state(state)
env.get_image().show_image()

model_id = 'gpt-4o'
save_dir = f"{VISUALISATIONS_DIR}/tmp"
os.makedirs(save_dir, exist_ok=True)

while True:

    env.from_state(state)

    if all(env.get_state().is_potted(ball) for ball in target_balls):
        print("All target balls potted! - Generating new state")
        state = TwoPlayerState().randomize()
        env.from_state(state)
        env.get_image().show_image()
    
    agent.take_shot(env, state)

    shot_params = agent.record['shots']
    chosen_shot = agent.record['best_shot_index']
    
    # Remove all files in save_dir
    for file in os.listdir(save_dir):
        os.remove(os.path.join(save_dir, file))
    
    # Save the shots as gifs
    for i, shot in enumerate(shot_params):
        env.save_shot_gif(state, shot, f"{save_dir}/tmp_shot_{i+1}.gif" ) 

    explanation = explain_shots(
        env=env,
        model_id=model_id,
        state=state,
        shot_params=shot_params,
        target_balls=target_balls
    )

    print("="*50)
    print(explanation)
    print("="*50)

    env.from_state(state)
    env.strike(**shot)
    state = env.get_state()

    decision = input("Press Enter to continue, type restart to restart, or exit to exit: ")

    if decision == "exit":
        break
    elif decision == "restart":
        state = TwoPlayerState().randomize()
        continue
