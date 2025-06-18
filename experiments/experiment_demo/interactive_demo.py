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
    TwoPlayerState,
    Pool,
    VISUALISATIONS_DIR,
    plt_best_shot,
    plot_heatmaps
)
from poolagent.experiment_manager import LLM

def print_reasoning(record):
        
    # Print reasoning with a header
    print("\n=== Suggest Reasoning ===")
    print(record['lm_suggester']['response']['reasoning'])

    # Print shots with a header
    print("\n=== Shots ===")
    print(record['lm_suggester']['response']['shots'])

    # Print Choosing Reasoning
    if 'lm_chooser' in record:
        print("\n=== Choosing Reasoning ===")
        print(record['lm_chooser']['response']['reasoning'])

def print_distribution(distribution):

    if isinstance(distribution[0], list):
        distribution = distribution[0]

    max_bar_width = 40  # Maximum width of the bar
    expected_value = sum([(i/10) * prob for i, prob in enumerate(distribution)])
    print("\nProbability Distribution:")
    print("-------------------------")
    
    for i, prob in enumerate(distribution):
        bar_width = int(prob * max_bar_width)
        bar = '█' * bar_width
        print(f"{i/10:.1f}: {bar} {prob:.2f}")
    print(f"\nExpected Value: {expected_value:.2f}")
    print("-------------------------")

### Set Seed
seed =  np.random.randint(0, 1000)
np.random.seed(seed)

env = Pool()

state = TwoPlayerState().randomize()
target_balls = ['red', 'blue', 'yellow']

agent_choice = input("Choose agent (function or language): ")
if agent_choice == "function":
    xai_agent = FunctionAgent(target_balls)
elif agent_choice == "language-function":
    lm = LLM('gpt-4o')
    xai_agent = LanguageFunctionAgent(target_balls)
elif agent_choice == "language":
    lm = LLM('gpt-4o')
    xai_agent = LanguageAgent(target_balls)
elif agent_choice == "language-def":
    lm = LLM('gpt-4o')
    xai_agent = LanguageDEFAgent(target_balls)
elif agent_choice == "brute-force":
    xai_agent = BruteForceAgent(target_balls)
else:
    raise ValueError("Invalid agent choice")
xai_agent.N = 3

weave.init(f"Demo-{agent_choice}-{seed}")

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
        
    if agent_choice == 'language-def':
        message = input("Enter message: ")
        shot = take_shot(xai_agent, env, state, lm.llm, parallel=False, message=message)
        values, difficulties = xai_agent.record['function_chooser']['values'], xai_agent.record['function_chooser']['difficulties']
        values, difficulties = xai_agent.function_chooser.normalise(values, difficulties)
        env.save_shot_gif(state, shot, f"{VISUALISATIONS_DIR}/tmp_shot.gif")

        best_shot_index = xai_agent.record['function_chooser']['best_shot_index']
        plot_heatmaps(best_shot_index, values, difficulties, VISUALISATIONS_DIR)

        chosen_distribution = xai_agent.record['function_chooser']['model_distributions'][best_shot_index]
        print_distribution(chosen_distribution)

        print_reasoning(xai_agent.record)

    elif agent_choice == 'brute-force':

        print("Brute force agent --- taking shot")
        shot = xai_agent.take_shot(env, state)
        env.save_shot_gif(state, shot, f"{VISUALISATIONS_DIR}/tmp_shot.gif")
        print(f"Shot: {shot}")
        print(f"Events: {env.get_events()}")

    else:
        shot = take_shot(xai_agent, env, lm.llm, state, parallel=False)
        values, difficulties = xai_agent.record['values'], xai_agent.record['difficulties']

        env.save_shot_gif(state, shot, f"{VISUALISATIONS_DIR}/tmp_shot.gif")
        values, difficulties = xai_agent.chooser.normalise(values, difficulties
                                                           )
        best_shot_index = xai_agent.record['best_shot_index']
        plot_heatmaps(best_shot_index, values, difficulties, VISUALISATIONS_DIR)

        chosen_distribution = xai_agent.record['model_distributions'][best_shot_index]
        print_distribution(chosen_distribution)

    env.from_state(state)
    env.strike(**shot)
    state = env.get_state()

    decision = input("Press Enter to continue, type restart to restart, or exit to exit: ")

    if decision == "exit":
        break
    elif decision == "restart":
        state = TwoPlayerState().randomize()
        continue
