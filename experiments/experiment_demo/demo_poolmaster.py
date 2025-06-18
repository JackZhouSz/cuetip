import yaml
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import weave

from poolagent import (
    FunctionAgent,
    LanguageAgent,
    LanguageDEFAgent,
    LanguageFunctionAgent,
    BruteForceAgent,
    PoolMasterAgent,
    TwoPlayerState,
    Pool,
    VISUALISATIONS_DIR,
    plt_best_shot,
    plot_heatmaps
)

env = Pool()
state = TwoPlayerState().randomize()
target_balls = ['red', 'blue', 'yellow']
xai_agent = PoolMasterAgent(target_balls)
env.from_state(state)
env.get_image().show_image()

@weave.op()
def take_shot(xai_agent, env, state, llm, parallel=False, message=None):
    return xai_agent.take_shot(env, state, llm, parallel=parallel, message=message)

while True:

    env.from_state(state)

    if all(env.get_state().is_potted(ball) for ball in target_balls):
        print("All target balls potted! - Generating new state")
        state = TwoPlayerState().randomize()
        env.from_state(state)
        env.get_image().show_image()

    print("PoolMasterAgent --- taking shot")
    shot = xai_agent.take_shot(env, state)
    env.save_shot_gif(state, shot, f"{VISUALISATIONS_DIR}/tmp_shot.gif")
    print(f"Shot: {shot}")
    print(f"Events: {env.get_events()}")



    env.from_state(state)
    env.strike(**shot)
    state = env.get_state()

    decision = input("Press Enter to continue, type restart to restart, or exit to exit: ")

    if decision == "exit":
        break
    elif decision == "restart":
        state = TwoPlayerState().randomize()
        continue
