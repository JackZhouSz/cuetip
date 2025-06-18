import json
import random
import dspy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import itertools
import os
import argparse
import weave

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poolagent.path import DATA_DIR, VISUALISATIONS_DIR
from poolagent.dspy_definitions import ChooseBetweenExplanations
from poolagent.utils import dspy_setup

EVALUATOR_LLM = "gpt-4o"
SAVE_DATA_NAME = "exp3-results"
RESULTS_DIR = "results"

class LLM_Evaluator:
    def __init__(self, config):
        dspy_setup(config)
        self.evaluate_explanations = dspy.ChainOfThought(ChooseBetweenExplanations)
    
    @weave.op()
    def evaluate_pair(self, shot_params, shot_events, explanation1, explanation2):
        max_attempts = 3
        attempts = 0
        decision = None

        while attempts < max_attempts:
            try:
                decision = self.evaluate_explanations(
                    shot_params=shot_params,
                    shot_events=shot_events,
                    explanation1=explanation1,
                    explanation2=explanation2
                ).better_explanation
                break
            except Exception as e:
                print(f"Error evaluating pair: {e}")
                attempts += 1
                continue

        if decision is None:
            print(f"Failed to evaluate pair after {max_attempts} attempts")
            return 'with_functions' if random.random() < 0.5 else 'without_functions'

        if 'ONE' in decision:
            return 'with_functions'
        elif 'TWO' in decision:
            return 'without_functions'
        else:
            print("Invalid decision - neither ONE nor TWO in the response")
            print(decision)

            return 'with_functions' if 'Explanation 1' in decision else 'without_functions'

    def evaluate(self, entry):
        shot_params = ", ".join([f"{key}: {value}" for key, value in entry['shot_params'].items()])
        shot_events = ", ".join([e["encoding"] for e in entry['events']])

        with_functions_explanations = entry['explanations']['with_functions']
        without_functions_explanations = entry['explanations']['without_functions']

        total_comparisons = 0
        with_functions_count = 0
        
        # Add control comparisons
        control_with_count = 0
        control_without_count = 0
        control_total = 0

        # Control comparisons within same type
        for exp1, exp2 in tqdm(itertools.combinations(with_functions_explanations, 2), desc='Control With Functions'):
            result = self.evaluate_pair(shot_params, shot_events, exp1, exp2)
            if result == 'with_functions':
                control_with_count += 1
            control_total += 1

        for exp1, exp2 in tqdm(itertools.combinations(without_functions_explanations, 2), desc='Control Without Functions'):
            result = self.evaluate_pair(shot_params, shot_events, exp1, exp2)
            if result == 'with_functions':
                control_without_count += 1
            control_total += 1

        # Main comparisons between types
        for exp1, exp2 in tqdm(itertools.product(with_functions_explanations, without_functions_explanations), desc='Evaluating Explanations'):
            result = self.evaluate_pair(shot_params, shot_events, exp1, exp2)
            if result == 'with_functions':
                with_functions_count += 1
            total_comparisons += 1

        if total_comparisons == 0:
            return 0.5, 0.5, 0.5  # Return defaults if no comparisons

        average_score = with_functions_count / total_comparisons
        control_with_score = control_with_count / (control_total/2) if control_total > 0 else 0.5
        control_without_score = control_without_count / (control_total/2) if control_total > 0 else 0.5
        
        return average_score, control_with_score, control_without_score

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def compare_explanation_types(output_file, evaluator, data, model_filter=None):
    results = defaultdict(lambda: {
        'with_functions': [], 
        'without_functions': [],
        'control_with': [],
        'control_without': []
    })
    existing_results = load_existing_results(output_file)

    data.pop('timestamp', None)

    if model_filter:
        data = {k: v for k, v in data.items() if k == model_filter}
        if not data:
            raise ValueError(f"Model '{model_filter}' not found in the data")

    for model, entries in tqdm(data.items(), desc='Evaluating Models'):
        for entry in entries:
            entry_id = f"{model}_{hash(json.dumps(entry, sort_keys=True))}"

            if entry_id in existing_results:
                score = existing_results[entry_id]['with_functions']
                control_with = existing_results[entry_id]['control_with']
                control_without = existing_results[entry_id]['control_without']
            else:
                score, control_with, control_without = evaluator.evaluate(entry)
                existing_results[entry_id] = {
                    'with_functions': score,
                    'without_functions': 1 - score,
                    'control_with': control_with,
                    'control_without': control_without
                }
            
            results[model]['with_functions'].append(score)
            results[model]['without_functions'].append(1 - score)
            results[model]['control_with'].append(control_with)
            results[model]['control_without'].append(control_without)

            averaged_results = average_results(results)
            save_results(averaged_results, output_file)

            if os.path.exists(f"{DATA_DIR}/{SAVE_DATA_NAME}-working.json"):
                with open(f"{DATA_DIR}/{SAVE_DATA_NAME}-working.json", 'r') as file:
                    existing_all_results = json.load(file)

                    if model not in existing_all_results:
                        existing_all_results[model] = {
                            'with_functions': [],
                            'without_functions': [],
                            'control_with': [],
                            'control_without': []
                        }

                    for key in ['with_functions', 'without_functions', 'control_with', 'control_without']:
                        if key not in existing_all_results[model]:
                            existing_all_results[model][key] = []
                        existing_all_results[model][key].append(results[model][key][-1])
            else:
                existing_all_results = results
            with open(f"{DATA_DIR}/{SAVE_DATA_NAME}-working.json", 'w') as file:
                json.dump(existing_all_results, file, indent=4)
    
    return averaged_results

def load_existing_results(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return {}

def average_results(results):
    averaged_results = {}
    for model, scores in results.items():
        averaged_results[model] = {
            key: sum(values) / len(values) if values else 0
            for key, values in scores.items()
        }
    return averaged_results

def save_results(results, file_path):
    current_results = load_existing_results(file_path)  
    current_results.update(results)
    with open(file_path, 'w') as file:
        json.dump(current_results, file, indent=2)

def plot_comparison(results):
    models = list(results.keys())
    metrics = ['with_functions', 'without_functions', 'control_with', 'control_without']
    values = {metric: [results[model][metric] for model in models] for metric in metrics}

    x = range(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, vals) in enumerate(values.items()):
        ax.bar([pos + (i-1.5)*width for pos in x], vals, width, label=metric.replace('_', ' ').title())

    ax.set_ylabel('Average Score')
    ax.set_title('Comparison of Explanation Types and Controls by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{VISUALISATIONS_DIR}/explanation_type_comparison.png')
    plt.close()

def get_most_recent_file():
    files = os.listdir(f"./{RESULTS_DIR}/")
    files = [f for f in files if '.json' in f]
    files.sort()
    return f"./{RESULTS_DIR}/{files[-1]}"

def main(llm_config, input_file, output_file, model_filter=None):
    evaluator = LLM_Evaluator(llm_config)
    data = load_json(input_file)
    results = compare_explanation_types(output_file, evaluator, data, model_filter)
    plot_comparison(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate explanations with optional model filter')
    parser.add_argument('--model', type=str, help='Filter evaluation to specific model name', default=None)
    args = parser.parse_args()

    weave.init(f"ExperimentThree-gpt-4o-eval") 

    input_json = get_most_recent_file()
    output_json = f"{DATA_DIR}/{SAVE_DATA_NAME}.json"

    llm_config = {
        "temperature": 0.1,
        "top_k": 40,
        "top_p": 1.0,
        "max_tokens": 2048,
        "model": EVALUATOR_LLM,
        "backend": "openai",
    }
    main(llm_config, input_json, output_json, args.model)