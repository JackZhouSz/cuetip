import json, random
import numpy as np
from tqdm import tqdm

from poolagent.utils import State, Event
from poolagent.path import DATA_DIR, ROOT_DIR
from poolagent.domain_expert_info import VALUE_INPUT_SIZE, DIFFICULTY_INPUT_SIZE, FixedRulesEvaluator

from poolagent.pool import Pool, LIMITS
from poolagent.agents import SKILL_LEVELS

BIN_COUNT = 11

def get_distribution(entry):
    bins = [0] * BIN_COUNT

    for estimate in entry['estimates']:
        bins[int(estimate * 10)] += 1

    total_sum = sum(bins)
    bins = [count / total_sum for count in bins]

    return bins

def get_expected_value(distribution):
    expected_value = 0

    for i, prob in enumerate(distribution):

        x = i / 10
        p_x = prob

        expected_value += x * p_x

    return expected_value

def process_dataset(env, dataset_path, evaluator, key):
    with open(dataset_path, 'r') as f:
        data = json.load(f)[key]

    results = []
    target_balls = ['red', 'blue', 'yellow']

    for entry in tqdm(data, desc="Processing Stochastic data"):

        start_state = State.from_json(entry['state'])
        shot = entry['action']

        env.from_state(start_state)
        env.strike(**shot)
        end_state = env.get_state()
        events = env.get_events()

        entry['values'] = evaluator.evaluate_state(start_state, shot, end_state, target_balls)
        entry['difficulties'] = evaluator.evaluate_difficulty(start_state, shot, events, target_balls)

        entry['distribution'] = get_distribution(entry)
        entry['expected_value'] = get_expected_value(entry['distribution'])
        entry['entropy'] = -sum([p * np.log(p) for p in entry['distribution'] if p > 0])

        entry['end_state'] = end_state.to_json()
        entry['events'] = [[event.encoding, event.pos] for event in events]

        results.append(entry)

    return results

def save_averages(data):
    values = []
    difficulties = []
    
    for entry in data:
        values.append(entry['values'])
        difficulties.append(entry['difficulties'])

    values = np.array(values)
    difficulties = np.array(difficulties)

    with open(f"{DATA_DIR}/fixed_function_averages.json", 'w') as f:
        json.dump({
            'value': [(np.mean(values[:, i]), np.std(values[:, i])) for i in range(VALUE_INPUT_SIZE)],
            'difficulty': [(np.mean(difficulties[:, i]), np.std(difficulties[:, i])) for i in range(DIFFICULTY_INPUT_SIZE)]
        }, f, indent=4)

if __name__ == "__main__":
    evaluator = FixedRulesEvaluator()
    env = Pool()
    train_results = process_dataset(env, f"{DATA_DIR}/poolmaster_training_data.json", evaluator, 'train')
    test_results = process_dataset(env, f"{DATA_DIR}/poolmaster_training_data.json", evaluator, 'test')

    with open(f"{DATA_DIR}/poolmaster_training_data.json", 'w') as f:
        json.dump({"train": train_results, "test": test_results}, f, indent=4)

    save_averages(train_results)

    print("Processing complete. Results and averages saved.")
