import json, random
import numpy as np
from tqdm import tqdm

from poolagent.utils import State, Event
from poolagent.path import DATA_DIR, ROOT_DIR
from poolagent.domain_expert_info import VALUE_INPUT_SIZE, DIFFICULTY_INPUT_SIZE, FixedRulesEvaluator

from poolagent.pool import Pool, LIMITS
from poolagent.agents import SKILL_LEVELS

BRANCHING_FACTOR = 10

def calculate_difficulty(env, state, shot, events):
    """Returns the ground truth difficulty of a shot. This is calculated by repeatedly sampling the shot with a small amount of noise and finding the average number of events that co-occur. Importantly, we want to remove the unimportant events from the difficulty calculation, so only include:
    1. Ball pocket events
    2. Cue ball events
    3. No ball stopping events 

    Args:
        state (_type_): The starting state of the pool table
        shot (_type_): The shot to evaluate
        events (_type_): The events that should occur
    """

    env.from_state(state)

    def strip_events(events):
        stripped_events = []

        potted_balls = [
            e.arguments[0] for e in events if 'ball_pocket' in e.encoding
        ]

        for e in events:
            
            if any([potted_ball in e.arguments for potted_ball in potted_balls]):
                stripped_events.append(e)
                continue

            if 'cue' in e.arguments:
                stripped_events.append(e)
                continue

        return stripped_events
    
    def sample_shot(shot, skill_level):
        new_shot = {}
        for k in shot.keys():
            new_shot[k] = shot[k] + random.uniform(-skill_level[k], skill_level[k])
            new_shot[k] = np.clip(new_shot[k], LIMITS[k][0], LIMITS[k][1])
        return new_shot
    
    events = strip_events(events)
    N = 100
    skill_level = SKILL_LEVELS.AMATEUR
    score = 0

    increment = 1 / (len(events) * N)

    for _ in range(N):

        env.from_state(state)
        new_shot = sample_shot(shot, skill_level)
        env.strike(**new_shot)
        new_events = env.get_events()

        for e in events:
            if e in new_events:
                score += increment
                new_events = new_events[new_events.index(e) + 1:]

    return 1.0 - score


def process_dataset(env, dataset_path, evaluator, key):
    with open(dataset_path, 'r') as f:
        data = json.load(f)[key]

    results = []
    target_balls = ['red', 'blue', 'yellow']

    max_values = np.zeros(VALUE_INPUT_SIZE)
    min_values = np.ones(VALUE_INPUT_SIZE)
    avg_values = np.zeros(VALUE_INPUT_SIZE)

    max_difficulties = np.zeros(DIFFICULTY_INPUT_SIZE)
    min_difficulties = np.ones(DIFFICULTY_INPUT_SIZE)
    avg_difficulties = np.zeros(DIFFICULTY_INPUT_SIZE)

    for entry in tqdm(data, desc="Processing MCTS data"):
        if max(entry['visit_distribution']) - min(entry['visit_distribution']) < 0.05:
            continue

        state = State.from_json(entry['starting_state'])
        shots = [s['params'] for s in entry['follow_up_states']]
        states = [State.from_json(s) for s in entry['follow_up_states']]
        all_events = entry['events']
        
        entry['values'] = [
            evaluator.evaluate_state(state, shot, final_state, target_balls) for shot, final_state in zip(shots, states)
        ]

        entry['difficulties'] = [
            evaluator.evaluate_difficulty(state, shot, [Event.from_encoding(e, pos) for e, pos in events], target_balls)
            for shot, events in zip(shots, all_events)
        ]

        # entry['target_difficulty'] = [
        #     calculate_difficulty(env, state, shot, [Event.from_encoding(e, pos) for e, pos in events])
        #     for shot, events in zip(shots, all_events)
        # ]

        for i in range(VALUE_INPUT_SIZE):
            max_values[i] = max(max_values[i], max([v[i] for v in entry['values']]))
            min_values[i] = min(min_values[i], min([v[i] for v in entry['values']]))
            avg_values[i] += sum([v[i] for v in entry['values']])
        for i in range(DIFFICULTY_INPUT_SIZE):
            max_difficulties[i] = max(max_difficulties[i], max([v[i] for v in entry['difficulties']]))
            min_difficulties[i] = min(min_difficulties[i], min([v[i] for v in entry['difficulties']]))
            avg_difficulties[i] += sum([v[i] for v in entry['difficulties']])
    
        results.append(entry)

        if len(results) % 10 == 0:
            d = evaluator.get_average_execution_times()
            for k, v in d.items():
                i = int(k.split('_')[-1]) - 1
                if 'value' in k:
                    avg = avg_values[i] / ( len(results) * BRANCHING_FACTOR )
                    print(f"{k}: {v:.6f}ms")
                    print(f"    Min {min_values[i]:.2f} - Max {max_values[i]:.2f} - Avg {avg:.4f}")
                elif 'difficulty' in k:
                    avg = avg_difficulties[i] / ( len(results) * BRANCHING_FACTOR )
                    print(f"{k}: {v:.6f}ms")
                    print(f"    Min {min_difficulties[i]:.2f} - Max {max_difficulties[i]:.2f} - Avg {avg:.4f}")
                else:
                    raise ValueError(f"Unknown key: {k}")

    return results

def save_averages(data):
    values = []
    difficulties = []
    
    for entry in data:
        for i in range(BRANCHING_FACTOR):
            values.append(entry['values'][i])
            difficulties.append(entry['difficulties'][i])

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
    train_results = process_dataset(env, f"{DATA_DIR}/mcts_training_data.json", evaluator, 'train')
    test_results = process_dataset(env, f"{DATA_DIR}/mcts_training_data.json", evaluator, 'test')

    with open(f"{DATA_DIR}/mcts_training_data.json", 'w') as f:
        json.dump({"train": train_results, "test": test_results}, f, indent=4)

    save_averages(train_results)

    print("Processing complete. Results and averages saved.")
