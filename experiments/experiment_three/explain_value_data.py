import json
import torch
import numpy as np 
from tqdm import tqdm
from datetime import datetime

from poolagent.domain_expert_info.explain_shot import explain_shot_func, explain_shot_func_no_functions
from poolagent.path import DATA_DIR
from poolagent.agents import FunctionChooser
from poolagent.pool import Pool
from poolagent.utils import State, start_pheonix

LLM_MODELS = [
    ("gpt-4o", "azure"),
    ("gpt-4o-mini", "azure"),
    ("llama3.1:8b-instruct-fp16", "ollama"), 
    ("gemma:7b-instruct-v1.1-fp16", "ollama"),
    ("mistral:7b-instruct-fp16", "ollama")
]  
MAX_COUNT = 50
K_EXPLANATIONS = 3

if __name__ == "__main__":

    start_pheonix()

    with open(f"{DATA_DIR}/mcts_training_data.json", "r") as f:
        training_data = json.load(f)['train']

    shuffle_indices = np.random.permutation(int(MAX_COUNT*1.5))
    training_data = [training_data[i] for i in shuffle_indices]

    with open(f"{DATA_DIR}/fixed_function_averages.json", "r") as f:
        fixed_function_averages = json.load(f)

    env = Pool()
    agent_config = {}
    agent_config['value_weight'] = 0.5
    agent_config['difficulty_weight'] = 0.5
    chooser = FunctionChooser(target_balls=['red', 'blue', 'yellow'], config=agent_config)
    
    try:
        with open(f"{DATA_DIR}/shot_explanations.json", "r") as f:
            explanations = json.load(f)
            explanations["timestamp"] = datetime.now().isoformat()  # Update timestamp
    except FileNotFoundError:
        explanations = {"timestamp": datetime.now().isoformat()}

    for LLM_MODEL, BACKEND in LLM_MODELS:
        if LLM_MODEL not in explanations:
            explanations[LLM_MODEL] = []

        existing_count = len(explanations[LLM_MODEL])

        llm_config = {
            "model": LLM_MODEL,
            "temperature": 0.2,
            "max_tokens": 4096,
            "backend": BACKEND,
        }
        
        if existing_count >= MAX_COUNT:
            print(f"All explanations have been generated for {LLM_MODEL}.")
            continue

        for entry in tqdm(training_data[existing_count:MAX_COUNT], initial=existing_count, total=MAX_COUNT, desc=f"Processing {LLM_MODEL}"):
            starting_state = State.from_json(entry['starting_state'])
            shots = [state['params'] for state in entry['follow_up_states']]
            states = [State.from_json(state) for state in entry['follow_up_states']]
            
            env.from_state(starting_state)
            events = []
            for shot in shots:
                env.strike(**shot)
                events.append(env.get_events())
                env.from_state(starting_state)

            best_shot_index, model_distributions, expected_values, raw_values, raw_difficulties = chooser.evaluate_shots(starting_state, shots, events, states)

            normalized_values, normalized_difficulties = chooser.normalise(raw_values, raw_difficulties)

            mcts_best_shot_index = np.argmax(entry['visit_distribution'])

            explanation_data = {
                "starting_state": starting_state.to_json(),
                "follow_up_states": [state.to_json() for state in states],
                "mcts_visit_distribution": entry['visit_distribution']
            }

            def get_shot_data(index):
                return {
                    "shot_params": shots[index],
                    "end_state": states[index].to_json(),
                    "events": [e.to_json() for e in events[index]],
                    "raw_values": raw_values[index].tolist(),
                    "raw_difficulty": raw_difficulties[index].tolist(),
                    "normalized_values": normalized_values[index].tolist(),
                    "normalized_difficulty": normalized_difficulties[index].tolist(),
                }
            
            def generate_explanations(starting_state, shot_params, events, norm_values, norm_difficulty):
                seeds = np.random.randint(0, 10000, K_EXPLANATIONS)
                return {
                    "with_functions": [
                        explain_shot_func(llm_config, starting_state, shot_params, events, norm_values, norm_difficulty, seed=seeds[i])
                        for i in range(K_EXPLANATIONS)
                    ],
                    "without_functions": [
                        explain_shot_func_no_functions(llm_config, starting_state, shot_params, events, seed=seeds[i])
                        for i in range(K_EXPLANATIONS)
                    ]
                }
            
            explanation_data["mcts_best_shot"] = {
                "index": int(mcts_best_shot_index),
                **get_shot_data(mcts_best_shot_index),
                "explanations": generate_explanations(
                    starting_state,
                    shots[mcts_best_shot_index],
                    events[mcts_best_shot_index],
                    normalized_values[mcts_best_shot_index].tolist(),
                    normalized_difficulties[mcts_best_shot_index].tolist()
                )
            }

            explanations[LLM_MODEL].append(explanation_data)

            # Save after each iteration to preserve progress
            with open(f"{DATA_DIR}/shot_explanations.json", "w") as f:
                json.dump(explanations, f, indent=4)

    print("Processing completed for all LLM models.")