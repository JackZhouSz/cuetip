import numpy as np
import os
import dotenv
import dspy

dotenv.load_dotenv()

#
# Example of using the LanguageFunctionAgent to optimise a shot and visualise the results - there are no target events, the LM creates N potential shots and each set of shot parameters is optimised using the trained neural network and simulated annealing.
#

from poolagent import (
    LanguageFunctionAgent,
    State,
    Pool,
    VISUALISATIONS_DIR,
    plot_heatmaps,
    print_distribution,
    explain_shot_func
)

seed = 42
np.random.seed(seed)

env = Pool()

state = State(random=True)
target_balls = ['red', 'blue', 'yellow']

# Set up LLM using DSPy and a valid OpenAI API key (local models are also supported through the DSPy API)

assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY not found in .env file, this is required for the OpenAI API"
lm = dspy.OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)
llm_config = {
    "temperature": 0.2,
    "top_k": 40,
    "top_p": 1.0,
    "max_tokens": 2048,
    "model": "gpt-4o-mini",
    "backend": "openai",
}

if not os.path.exists(VISUALISATIONS_DIR):
    os.makedirs(VISUALISATIONS_DIR)

while True:

    env.from_state(state)

    agent_config = LanguageFunctionAgent.default_dict()
    xai_agent = LanguageFunctionAgent(target_balls)
    xai_agent.N = 3
    
    shot = xai_agent.take_shot(env, state, lm)
    
    env.save_shot_gif(state, shot, f"{VISUALISATIONS_DIR}/tmp_shot.gif")

    shot_values = xai_agent.record['function_chooser']['values']
    shot_difficulties = xai_agent.record['function_chooser']['difficulties']
    plot_heatmaps(xai_agent.record['best_shot_index'], shot_values, shot_difficulties, VISUALISATIONS_DIR)

    chosen_distribution = xai_agent.record['function_chooser']['model_distributions'][xai_agent.record['best_shot_index']]

    print_distribution(chosen_distribution)

    start_state = state.copy()
    env.from_state(state)
    env.strike(**shot)
    state = env.get_state()
    events = env.get_events()

    print(f"Chosen shot: {shot}")
    for e in events:
        print(f"- {e}")

    explanation = explain_shot_func(
        llm_config, 
        start_state,
        shot,
        events, 
        shot_values[xai_agent.record['best_shot_index']], 
        shot_difficulties[xai_agent.record['best_shot_index']]
    )

    print("="*25 + "EXPLANATION" + "="*25)
    print(explanation)
    print("="*25 + "EXPLANATION" + "="*25)



    decision = input("Press Enter to continue, type restart to restart, or exit to exit: ")

    if decision == "exit":
        break
    elif decision == "restart":
        state = State().randomize()
        continue
