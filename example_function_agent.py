import matplotlib.pyplot as plt
import yaml
import numpy as np
import os


#
# Example of using the FunctionAgent to optimise a shot and visualise the results - there are no target events, each set of shot parameters is optimised to maximise the output of the trained neural network through simulated annealing.
#

from poolagent import (
    FunctionAgent,
    State,
    Pool,
    VISUALISATIONS_DIR,
    plot_heatmaps,
    print_distribution
)

seed = 42
np.random.seed(seed)

env = Pool()

state = State().randomize()
target_balls = ['red', 'blue', 'yellow']

if not os.path.exists(VISUALISATIONS_DIR):
    os.makedirs(VISUALISATIONS_DIR)

while True:

    env.from_state(state)
    agent = FunctionAgent(target_balls)

    # Set number of shots to optimise
    agent.N = 5
    
    shot = agent.take_shot(env, state)
    
    env.save_shot_gif(state, shot, f"{VISUALISATIONS_DIR}/tmp_shot.gif")

    plot_heatmaps(agent.record['best_shot_index'], agent.record['values'], agent.record['difficulties'], VISUALISATIONS_DIR)

    chosen_distribution = agent.record['model_distributions'][agent.record['best_shot_index']]

    print_distribution(chosen_distribution)

    env.from_state(state)
    env.strike(**shot)
    state = env.get_state()

    print(f"Chosen shot: {shot}")
    events = env.get_events()
    for e in events:
        print(f"- {e}")

    decision = input("Press Enter to continue, type restart to restart, or exit to exit: ")

    if decision == "exit":
        break
    elif decision == "restart":
        state = State().randomize()
        continue

