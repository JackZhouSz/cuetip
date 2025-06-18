import datetime, dspy, dsp, json

from typing import List

from poolagent.dspy_definitions import ExplainSingleShotSignature, ExplainSingleShotSignatureNoFunctions
from poolagent.path import DATA_DIR, VISUALISATIONS_DIR, RULES_DIR
from poolagent.utils import dspy_setup, State, Event

def explain_shot_func_no_functions(llm_config : dict, state: State, shot_params : dict, events : List[Event], seed : int = 0) -> str:

    ### Load DSPy
    dspy_setup(llm_config)
    dspy_explain_shot = dspy.ChainOfThought(ExplainSingleShotSignatureNoFunctions)

    shot_params_str = ""
    for key, value in shot_params.items():
        shot_params_str += f"{key}: {value:.2f}\n"

    ### Board coordinates
    board_coordinates_str = ""
    board_coordinates_str += f"Balls:\n"
    for k, v in state.ball_positions.items():
        if isinstance(v[0], str):
            continue
        board_coordinates_str += f"'{k}': ({v[0]:.2f},{v[1]:.2f})\n"
    board_coordinates_str += f"Pockets:\n"
    for k, v in state.pocket_positions.items():
        board_coordinates_str += f"'{k}': ({v[0]:.2f},{v[1]:.2f})\n"

    ### Events
    events_str = ""
    events_str += f"Events:\n"
    for e in events:
        events_str += f"'{e.encoding}' at {e.pos}\n"

    ### Explain shot
    response = dspy_explain_shot(
        seed=f"{seed}",
        shot_params=shot_params_str,
        board_coordinates=board_coordinates_str,
        events=events_str,
    )
    return response.explanation

def explain_shot_func(llm_config : dict, state: State, shot_params : dict, events : List[Event], value_rules_weights : List[float], difficulty_rules_weights: List[float], seed : int = 0) -> str:
    """Use a LLM to explain a shot using the provided value and difficulty function values.

    Args:
        llm_config (dict): LLM configuration
        shot_params (dict): Shot parameters
        value_rules_weights (List[float]): List of value function weights
        difficulty_rules_weights (List[float]): List of difficulty function weights
    """

    K_POSITIVE_RULES = 2
    K_NEGATIVE_RULES = 1

    ### Load DSPy
    dspy_setup(llm_config)
    dspy_explain_shot = dspy.ChainOfThought(ExplainSingleShotSignature)

    shot_params_str = ""
    for key, value in shot_params.items():
        shot_params_str += f"{key}: {value:.2f}\n"

    ### Load Rules and collect the top 3 and bottom 3 rules
    value_function_rules = {}
    with open(f"{RULES_DIR}/value_rules.json") as f:
        value_function_rules = json.load(f)
    difficulty_function_rules = {}
    with open(f"{RULES_DIR}/difficulty_rules.json") as f:
        difficulty_function_rules = json.load(f)

    value_function_values = []
    for i, w in enumerate(value_rules_weights):
        value_function_values.append([float(w), f"weight: {float(w):.2f}, rule: {value_function_rules[str(i+1)]}"])

    difficulty_function_values = []
    for i, w in enumerate(difficulty_rules_weights):
        difficulty_function_values.append([float(w), f"weight: {float(w):.2f}, rule: {difficulty_function_rules[str(i+1)]}"])

    value_function_values = sorted(value_function_values, key=lambda x: x[0], reverse=True)[:K_POSITIVE_RULES] + sorted(value_function_values, key=lambda x: x[0])[:K_NEGATIVE_RULES][::-1]

    difficulty_function_values = sorted(difficulty_function_values, key=lambda x: x[0], reverse=True)[:K_POSITIVE_RULES] + sorted(difficulty_function_values, key=lambda x: x[0])[:K_NEGATIVE_RULES][::-1]

    value_function_values = [v[1] for v in value_function_values]
    difficulty_function_values = [v[1] for v in difficulty_function_values]

    value_function_str = ""
    for v in value_function_values:
        value_function_str += f"{v}\n"

    difficulty_function_str = ""
    for v in difficulty_function_values:
        difficulty_function_str += f"{v}\n"

    ### Board coordinates
    board_coordinates_str = ""
    board_coordinates_str += f"Balls:\n"
    for k, v in state.ball_positions.items():
        if isinstance(v[0], str):
            continue
        board_coordinates_str += f"'{k}': ({v[0]:.2f},{v[1]:.2f})\n"
    board_coordinates_str += f"Pockets:\n"
    for k, v in state.pocket_positions.items():
        board_coordinates_str += f"'{k}': ({v[0]:.2f},{v[1]:.2f})\n"

    ### Events
    events_str = ""
    events_str += f"Events:\n"
    for e in events:
        events_str += f"'{e.encoding}' at {e.pos}\n"

    ### Explain shot
    response = dspy_explain_shot(
        seed=f"{seed}",
        shot_params=shot_params_str,
        board_coordinates=board_coordinates_str,
        events=events_str,
        value_function_rules_vals=value_function_str,
        difficulty_function_rules_vals=difficulty_function_str
    )
    return response.explanation

if __name__ == "__main__":

    from poolagent.utils import start_pheonix

    start_pheonix()

    llm_config = {
        "temperature": 0.2,
        "top_k": 40,
        "top_p": 1.0,
        "max_tokens": 2048,
        "model": "gpt-4o-mini",
        "backend": "azure",
    }

    from poolagent.agents import ValueDifficultyAgent
    from poolagent.utils import TwoPlayerState, plt_best_shot, plot_heatmaps
    from poolagent.pool import Pool
    import numpy as np

    env = Pool()
    agent_config = ValueDifficultyAgent.default_dict()
    agent = ValueDifficultyAgent(
        target_balls=['red', 'blue', 'yellow'],
        config=agent_config
    )

    while True:
        starting_state = TwoPlayerState().randomize()
        env.from_state(starting_state)

        shot_index, res = agent.take_100_shot(env)
        events = res["events"][shot_index]

        values = agent.last_shot_results['values'][shot_index]
        difficulties = agent.last_shot_results['difficulties'][shot_index]

        values, difficulties = agent.normalise(values, difficulties)

        best_shot = res["shots"][shot_index]

        explanation1 = explain_shot_func(
            llm_config, 
            starting_state,
            best_shot,
            events, 
            values, 
            difficulties
        )
        explanation2 = explain_shot_func_no_functions(
            llm_config, 
            starting_state,
            best_shot,
            events,
        )

        print("\n\n")
        print(explanation1)
        print("\n\n")
        print(explanation2)


        env.save_shot_gif(starting_state, best_shot, f"{VISUALISATIONS_DIR}/tmp_shot.gif")
        plot_heatmaps(agent, shot_index)

        input("Press Enter to continue...")
