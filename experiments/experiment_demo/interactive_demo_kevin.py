import yaml, json
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poolagent import (
    FunctionAgent,
    LanguageAgent,
    LanguageFunctionAgent,
    TwoPlayerState,
    Pool,
    VISUALISATIONS_DIR,
    plt_best_shot,
    plot_heatmaps
)
from poolagent.experiment_manager import LLM

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

agent_choice = input("Choose agent: language-function (lf) or language (l) --> ")
if agent_choice == "language" or agent_choice == "l":
    agent_choice = "language"
    lm = LLM('gpt-4o')
    xai_agent = LanguageAgent(target_balls)
elif agent_choice == "language-function" or agent_choice == "lf":
    agent_choice = "language-function"
    lm = LLM('gpt-4o')
    xai_agent = LanguageFunctionAgent(target_balls)
else:
    raise ValueError("Invalid agent choice")
xai_agent.N = 3

env.from_state(state)
env.get_image().show_image()

while True:

    env.from_state(state)

    if all(env.get_state().is_potted(ball) for ball in target_balls):
        print("All target balls potted! - Generating new state")
        state = TwoPlayerState().randomize()
        env.from_state(state)
        env.get_image().show_image()
        
    if agent_choice == 'language':
        message = input("Enter message: ")
        shot = xai_agent.take_shot(env, state, lm.llm, message=message, parallel=False)
        
        shots = xai_agent.record['shots']
        env.save_shot_gif(state, shots[0], f"{VISUALISATIONS_DIR}/demo/l_shot1.gif")
        env.save_shot_gif(state, shots[1], f"{VISUALISATIONS_DIR}/demo/l_shot2.gif")
        env.save_shot_gif(state, shots[2], f"{VISUALISATIONS_DIR}/demo/l_shot3.gif")

        suggester_response = xai_agent.record['lm_suggester']
        chooser_response = xai_agent.record['lm_chooser']

        print("\n=== Suggester ===")
        print(suggester_response['response']['reasoning'])

        print("\n=== Shots ===")
        print(suggester_response['response']['shots'])

        print("\n=== Chooser ===")
        print(chooser_response['response']['reasoning'])

        save_data = {
            'shots': shots,
            'suggester': {**xai_agent.suggester.record},
            'chooser': {**xai_agent.chooser.record}
        }
        with open(f"{VISUALISATIONS_DIR}/demo/l_data.json", 'w') as f:
            json.dump(save_data, f, indent=4)

    elif agent_choice == 'language-function':
        message = input("Enter message: ")
        shot = xai_agent.take_shot(env, state, lm.llm, message=message, parallel=False)
        values, difficulties = xai_agent.record['function_chooser']['values'], xai_agent.record['function_chooser']['difficulties']
        values, difficulties = xai_agent.chooser.normalise(values, difficulties)

        shots = xai_agent.record['shots']
        env.save_shot_gif(state, shots[0], f"{VISUALISATIONS_DIR}/demo/lf_shot1.gif")
        env.save_shot_gif(state, shots[1], f"{VISUALISATIONS_DIR}/demo/lf_shot2.gif")
        env.save_shot_gif(state, shots[2], f"{VISUALISATIONS_DIR}/demo/lf_shot3.gif")

        best_shot_index = xai_agent.record['function_chooser']['best_shot_index']
        plot_heatmaps(best_shot_index, values, difficulties, f"{VISUALISATIONS_DIR}/demo/")
        
        suggester_response = xai_agent.record['lm_suggester']

        print("\n=== Suggester ===")
        print(suggester_response['response']['reasoning'])

        print("\n=== Shots ===")
        print(suggester_response['response']['shots'])

        save_data = {
            'shots': shots,
            'suggester': {**xai_agent.suggester.record},
            'chooser': {**xai_agent.chooser.record}
        }
        with open(f"{VISUALISATIONS_DIR}/demo/lf_data.json", 'w') as f:
            json.dump(save_data, f, indent=4)

    env.from_state(state)
    env.strike(**shot)
    state = env.get_state()

    decision = input("Press Enter to continue, type restart to restart, or exit to exit: ")

    if decision == "exit":
        break
    elif decision == "restart":
        state = TwoPlayerState().randomize()
        continue
