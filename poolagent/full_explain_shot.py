import datetime, dspy, json
import numpy as np

from typing import List

from poolagent.pool import Pool
from poolagent.dspy_definitions import FullExplainSingleShotSignature, FullUnconditionalExplainSingleShotSignature
from poolagent.path import DATA_DIR, VISUALISATIONS_DIR, RULES_DIR
from poolagent.experiment_manager import LLM
from poolagent.agents import FunctionChooser
from poolagent.utils import State, Event

def extract_pros_and_cons(
    values,
    difficulties,
    value_rules,
    difficulty_rules
):
    pros = []
    cons = []

    for i, value in enumerate(values):
        # Create array of all other shots' values for comparison
        other_values = np.concatenate([values[j] for j in range(len(values)) if j != i])
        other_values = other_values.reshape(-1, len(value))

        # Calculate average of other shots for each metric
        other_avg = np.mean(other_values, axis=0)

        # Find relative differences
        relative_diffs = value - other_avg

        # Get indices of max and min differences
        best_idx = np.argmax(relative_diffs)

        # Get best and worst rules
        pro = f"Pro: {value_rules[best_idx]}"

        # Add to pros and cons lists
        pros.append(pro)
    
    for i, difficulty in enumerate(difficulties):
        # Create array of all other shots' difficulties for comparison
        other_difficulties = np.concatenate([difficulties[j] for j in range(len(difficulties)) if j != i])
        other_difficulties = other_difficulties.reshape(-1, len(difficulty))

        # Calculate average of other shots for each metric
        other_avg = np.mean(other_difficulties, axis=0)

        # Find relative differences
        relative_diffs = difficulty - other_avg

        # Get indices of max and min differences
        best_idx = np.argmax(relative_diffs)

        # Get best and worst rules
        con = f"Con: {difficulty_rules[best_idx]}"

        # Add to pros and cons lists
        cons.append(con)
    
    return pros, cons

def trim_events(events: List[Event], target_balls: List[str]) -> str:
    """Removes irrelevant events from the list, like:
        - Events that don't involve the target balls

    Args:
        events (List[Event]): The list of events to trim
        target_balls (List[str]): The target balls

    Returns:
        str: The trimmed list of events as a string
    """

    chosen_events = []
    balls_of_interest = target_balls + ["cue"]

    for event in events:
        if any([ball in event.encoding for ball in balls_of_interest]):
            encoding = event.encoding

            if 'stop' in encoding:
                continue

            if 'cushion' in encoding:
                encoding = f"ball-cushion-{event.arguments[0]}"

            chosen_events.append(encoding)

    return ", ".join(chosen_events)

def llm_inference(model_id, target_balls, events, best_shot_index, pros, cons):
    
    ### Load LLM and configure DSPy

    llm = LLM(model_id, max_tokens=1024, temperature=0.2)
    dspy.configure(lm=llm.llm)

    ### Prepare inputs
    
    # Events
    events_str = ""
    for idx, e in enumerate(events):
        events_str += f"Shot {idx+1}: " + trim_events(e, target_balls) + "\n"

    # Best shot index
    best_shot_index_str = f"The best shot is shot number {best_shot_index+1}\n"

    # Pros Cons
    pros_cons_str = ""
    for idx, (pro, con) in enumerate(zip(pros, cons)):
        pros_cons_str += f"Shot {idx+1}:\n{pro}\n{con}\n"

    # # Certainty
    # shot_certainty_str = ""
    # for idx, c in enumerate(certainty):
    #     if c < 0.2:
    #         classification = "Very Low"
    #     elif c < 0.4:
    #         classification = "Low"
    #     elif c < 0.6:
    #         classification = "Medium"
    #     elif c < 0.8:
    #         classification = "High"
    #     else:
    #         classification = "Very High"
    #     shot_certainty_str += f"The Certainty of {idx+1} is {classification}\n"

    ### Call LLM and return explanation

    if pros_cons_str:
        cot = dspy.ChainOfThought(FullExplainSingleShotSignature)
        response = cot(
            events=events_str,
            best_shot_index=best_shot_index_str,
            pros_cons = pros_cons_str,
            #shot_certainty = shot_certainty_str
        )
    else:
        cot = dspy.ChainOfThought(FullUnconditionalExplainSingleShotSignature)
        response = cot(
            events=events_str,
            best_shot_index=best_shot_index_str,
            #pros_cons = pros_cons_str,
            #shot_certainty = shot_certainty_str
        )
    return response.explanation

def calc_certainty(probs):
    MAX_ENTROPY = 0.367879
    probs_trimmed = np.array([p for p in probs if p > 0])
    bits = -np.sum(probs_trimmed * np.log(probs_trimmed))
    return (MAX_ENTROPY - (bits / len(probs))) / MAX_ENTROPY

def explain_shots(
    env: Pool,
    model_id: str,
    state: State,
    shot_params: List[dict],
    target_balls: List[str], 
    unconditional: bool = False 
):
    """Explain a shot choice, with regards to the other shots in the list.

    Args:
        env (Pool): The pool environment
        model_id (str): LLM model id
        state (State): The current state of the game
        shot_params (List[dict]): The parameters of the shots to explain
        chosen_shot (int): The index of the chosen shot in the list
        target_balls (List[str]): The target balls for the shot
    """

    chooser = FunctionChooser(target_balls)

    shot_variables = {
        "starting_state": state,
        "shot_params": shot_params,
        "shot_events": [],
        "final_states": [],
        "norm_values": [],
        "norm_difficulties": [],
        "best_shot_index": 0,
        "pros": [],
        "cons": [],
        #"shot_certainty": []
    }

    ### Simulate Shots

    for i, shot in enumerate(shot_params):
        env.from_state(state)
        env.strike(**shot)
        shot_variables["shot_events"].append(env.get_events())
        shot_variables["final_states"].append(env.get_state())

    ### Evaluate Shots

    best_shot_index, model_distributions, _, raw_values, raw_difficulties = chooser.evaluate_shots(state, shot_params, shot_variables["shot_events"], shot_variables["final_states"])
    normalised_values = [chooser.normalise(v, d) for v, d in zip(raw_values, raw_difficulties)]
    shot_variables["norm_values"] = [v[0] for v in normalised_values]
    shot_variables["norm_difficulties"] = [d[1] for d in normalised_values]
    shot_variables["best_shot_index"] = best_shot_index
    #shot_variables["shot_certainty"] = [calc_certainty(dist[0]) for dist in model_distributions]

    ### Extract Pros and Cons

    with open(f"{RULES_DIR}/value_rules.json") as f:
        value_function_rules = json.load(f)
    with open(f"{RULES_DIR}/difficulty_rules.json") as f:
        difficulty_function_rules = json.load(f)
    shot_variables["pros"], shot_variables["cons"] = extract_pros_and_cons(
        values=shot_variables["norm_values"],
        difficulties=shot_variables["norm_difficulties"],
        value_rules=list(value_function_rules.values()),
        difficulty_rules=list(difficulty_function_rules.values()),
    )

    ### LLM Inference
    return llm_inference(
        model_id=model_id,
        target_balls=target_balls,
        events=shot_variables["shot_events"],
        best_shot_index=best_shot_index,
        pros=shot_variables["pros"] if not unconditional else [],
        cons=shot_variables["cons"] if not unconditional else [],
        #certainty=shot_variables["shot_certainty"]
    )