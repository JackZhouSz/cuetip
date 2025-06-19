import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import dspy
import weave
from tqdm import tqdm

from poolagent.pool import Pool
from poolagent.utils import State, Event
from poolagent.agents import FunctionChooser

from gen_DEF_explanations import EstimateEnum


def map_to_estimation_enum(value: float) -> EstimateEnum:
    """
    Map a numeric value to the corresponding EstimationEnum.
    Args:
        value (float): Raw value between -1 and 1.

    Returns:
        EstimateEnum: The mapped estimation category.
    """
    if value <= -0.75:
        return EstimateEnum.very_low
    elif -0.75 < value <= -0.5:
        return EstimateEnum.low
    elif -0.5 < value <= -0.25:
        return EstimateEnum.moderately_low
    elif -0.25 < value <= 0.25:
        return EstimateEnum.moderate
    elif 0.25 < value <= 0.5:
        return EstimateEnum.moderately_high
    elif 0.5 < value <= 0.75:
        return EstimateEnum.high
    else:
        return EstimateEnum.very_high

def preprocess_entry(
    env: Pool,
    entry: Dict, 
    value_rules: Dict, 
    difficulty_rules: Dict, 
    with_def_signature: bool,
    percentage_values: bool,
    value_rules_entry:str,
    difficulty_rules_entry:str,
) -> Dict:
    """
    Preprocess a single entry to generate DSPy-compatible input fields.

    Args:
        entry (Dict): Raw entry data from the JSON file.
        value_rules (Dict): Value rules loaded from external sources.
        difficulty_rules (Dict): Difficulty rules loaded from external sources.
        with_def_signature (bool): Whether to follow ExplainDEFWithDEFSignature preprocessing.
        percentage_values (bool): Whether to format the DEF weights as percentages or as scalars in [-1,+1].
        value_rules_entry (str): DSPy signature entry string for value rules and optional weights.
        difficulty_rules_entry (str): DSPy signature entry string for diffuclty rules and optional weights.
    Returns:
        Dict: Preprocessed entry with DSPy inputs.
    """
    chooser = FunctionChooser(target_balls=['red', 'blue', 'yellow'])

    start_state = State.from_json(entry['state'])
    action = entry['action']

    env.from_state(start_state)
    env.strike(**action)
    events = env.get_events()
    end_state = env.get_state()

    _, _, _, raw_values, raw_difficulties = chooser.evaluate_shots(
        start_state, 
        [action], 
        [events], 
        [end_state],
    )
    raw_values, raw_difficulties = raw_values[0], raw_difficulties[0]
    normalized_values, normalized_difficulties = chooser.normalise(
        raw_values, 
        raw_difficulties,
        percentages=percentage_values,
    )
    
    scalars_value_weights = normalized_values
    scalars_difficulty_weights = normalized_difficulties
    if percentage_values:
        scalars_value_weights, scalars_difficulty_weights = chooser.normalise(
            raw_values,
            raw_difficulties,
            percentages=False,
        )

    entry["raw_value_rules_weights"] = raw_values
    entry["raw_difficulty_rules_weights"] = raw_difficulties

    entry["value_rules_weights"] = normalized_values
    entry["difficulty_rules_weights"] = normalized_difficulties
    
    # Target answers:
    value_targets = {
        f"value{idx + 1}_est": map_to_estimation_enum(scalars_value_weights[idx]).name
        for idx in range(len(value_rules))
    }
    difficulty_targets = {
        f"difficulty{idx + 1}_est": map_to_estimation_enum(scalars_difficulty_weights[idx]).name
        for idx in range(len(difficulty_rules))
    }

    # Shot garameters
    shot_params = action
    shot_params_str = "\n".join(f"{k}: {v:.2f}" for k, v in shot_params.items())

    # Board coordinates
    board_coordinates_str = "Balls:\n"
    for ball, pos in start_state.ball_positions.items():
        if isinstance(pos[0], str): continue
        board_coordinates_str += f"'{ball}': ({pos[0]:.2f},{pos[1]:.2f})\n"
    board_coordinates_str += "Pockets:\n"
    for pocket, pos in start_state.pocket_positions.items():
        board_coordinates_str += f"'{pocket}': ({pos[0]:.2f},{pos[1]:.2f})\n"

    # Events
    events_str = "Events:\n"
    json_events = [e.to_json() for e in events]
    for event in json_events:
        events_str += f"'{event['encoding']}' at {event['pos']}\n"

    value_function_str = "\n============================================\n"
    if with_def_signature:
        # Preprocessing for ExplainDEFWithDEFSignature
        value_function_str += "--- Value Rules ---\n"
        difficulty_function_str = "--- Difficulty Rules ---\n"
        if percentage_values:
            value_function_str += "\n".join(
                f"Value Rule {idx + 1} --> [{value_rules[str(idx + 1)]}] with weight {int(entry['value_rules_weights'][idx])}%."
                for idx in range(len(value_rules))
            )
            difficulty_function_str += "\n".join(
                f"Difficulty Rule {idx + 1} --> [{difficulty_rules[str(idx + 1)]}] with weight {int(entry['difficulty_rules_weights'][idx])}%."
                for idx in range(len(difficulty_rules))
            )
        else:
            value_function_str += "\n".join(
                f"Value Rule {idx + 1} --> [{value_rules[str(idx + 1)]}] with weight {float(entry['value_rules_weights'][idx]):.2f}."
                for idx in range(len(value_rules))
            )
            difficulty_function_str += "\n".join(
                f"Difficulty Rule {idx + 1} --> [{difficulty_rules[str(idx + 1)]}] with weight {float(entry['difficulty_rules_weights'][idx]):.2f}."
                for idx in range(len(difficulty_rules))
            )
    else:
        # Preprocessing for ExplainDEFWithOutDEFSignature
        value_function_str += "--- Value Rules ---\n"
        value_function_str += "\n".join(
            f"Value Rule {idx + 1} --> [{value_rules[str(idx + 1)]}]"
            for idx in range(len(value_rules))
        )
        difficulty_function_str = "--- Difficulty Rules ---\n"
        difficulty_function_str += "\n".join(
            f"Difficulty Rule {idx + 1} --> [{difficulty_rules[str(idx + 1)]}]"
            for idx in range(len(difficulty_rules))
        )
    difficulty_function_str += "\n============================================\n"

    rdict = {
        "entry": entry,
        "state": start_state.to_json(),
        "end_state": end_state.to_json(),
        "shot_params": shot_params_str,
        "board_coordinates": board_coordinates_str,
        "events": events_str,
        value_rules_entry: value_function_str,
        difficulty_rules_entry: difficulty_function_str,
        'target_answer': {
            'value': value_targets,
            'difficulty': difficulty_targets,
        },
    }

    return rdict


def create_dspy_dataset(
    metadata, 
    json_path: Path, 
    value_rules_path: Path, 
    difficulty_rules_path: Path, 
    with_def_signature: bool,
    percentage_values: bool,
) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """
    Load and preprocess JSON data to create DSPy training/validation datasets.

    Args:
        metadata: Metadata containing configuration like `num_train_examples`.
        json_path (Path): Path to the JSON file containing the data.
        value_rules_path (Path): Path to the JSON file containing value rules.
        difficulty_rules_path (Path): Path to the JSON file containing difficulty rules.
        with_def_signature (bool): Whether to follow ExplainDEFWithDEFSignature preprocessing.
        percentage_values (bool): Whether to format the DEF weights as percentages or as scalars in [-1,+1].

    Returns:
        Tuple of DSPy training and validation examples.
    """
    # Load JSON data
    with open(json_path, "r") as file:
        data = json.load(file)

    # Load value and difficulty rules
    with open(value_rules_path, "r") as file:
        value_rules = json.load(file)

    with open(difficulty_rules_path, "r") as file:
        difficulty_rules = json.load(file)

    value_rules_entry = "value_function_rules"
    difficulty_rules_entry = "difficulty_function_rules"
    if with_def_signature:
        value_rules_entry = "value_function_rules_vals"
        difficulty_rules_entry = "difficulty_function_rules_vals"

    # Preprocess data
    env = Pool()
    trainval_rows = []
    for entry in tqdm(data['train'][:metadata.num_train_examples+metadata.num_val_examples]):
        row = preprocess_entry(
            env,
            entry, 
            value_rules, 
            difficulty_rules, 
            with_def_signature,
            percentage_values,
            value_rules_entry=value_rules_entry,
            difficulty_rules_entry=difficulty_rules_entry,
        )
        trainval_rows.append(row)
    test_rows = []
    for entry in tqdm(data['test'][:metadata.num_test_examples]):
        row = preprocess_entry(
            env,
            entry, 
            value_rules, 
            difficulty_rules, 
            with_def_signature,
            percentage_values,
            value_rules_entry=value_rules_entry,
            difficulty_rules_entry=difficulty_rules_entry,
        )
        test_rows.append(row)

    # Split into training and validation sets
    train_rows = trainval_rows[:metadata.num_train_examples]
    val_rows = trainval_rows[metadata.num_train_examples:metadata.num_train_examples+metadata.num_val_examples]
    test_rows = test_rows[:metadata.num_test_examples]

    # Create DSPy `Example` objects
    dspy_train_examples = [
        dspy.Example(row).with_inputs(
            "shot_params", 
            "board_coordinates", 
            "events",
            value_rules_entry,
            difficulty_rules_entry,
        )
        for row in train_rows
    ]
    dspy_val_examples = [
        dspy.Example(row).with_inputs(
            "shot_params", 
            "board_coordinates", 
            "events",
            value_rules_entry,
            difficulty_rules_entry,
        )
        for row in val_rows
    ]
    dspy_test_examples = [
        dspy.Example(row).with_inputs(
            "shot_params", 
            "board_coordinates", 
            "events",
            value_rules_entry,
            difficulty_rules_entry,
        )
        for row in test_rows
    ]

    # Publish datasets to Weave for versioning and evaluation
    weave.init("SimLM-DEF-explanations")
    weave.publish(weave.Dataset(name=f"{metadata.task_name}_train", rows=train_rows))
    weave.publish(weave.Dataset(name=f"{metadata.task_name}_val", rows=val_rows))
    weave.publish(weave.Dataset(name=f"{metadata.task_name}_test", rows=test_rows))
    weave.finish()

    return dspy_train_examples, dspy_val_examples, dspy_test_examples

# Metadata for dataset creation
class Metadata:
    def __init__(
        self, 
        task_name:str, 
        with_def_signature:bool,
        percentage_values:bool,
        num_train_examples:int,
        num_val_examples:int,
        num_test_examples:int,
    ):
        self.task_name = task_name
        self.with_def_signature = with_def_signature
        self.num_train_examples = num_train_examples
        self.num_val_examples = num_val_examples
        self.num_test_examples = num_test_examples


def str2bool(inp):
    if not isinstance(inp, bool):
        assert isinstance(inp, str)
        inp = 'true' in inp.lower()
    return inp

    
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate DSPy datasets.")
    parser.add_argument(
        "--json_path", 
        type=str, 
        default="../../data/stochastic_training_data.json",
        help="Path to the JSON file containing data.",
    )
    parser.add_argument(
        "--value_rules_path", 
        default="../../data/value_rules.json",
        type=str, 
        help="Path to the JSON file containing value rules.",
    )
    parser.add_argument(
        "--difficulty_rules_path", 
        default="../../data/difficulty_rules.json",
        type=str, 
        help="Path to the JSON file containing difficulty rules.",
    )
    parser.add_argument(
        "--with_def_signature", 
        type=str2bool, 
        default=False, 
        help="Use ExplainDEFWithDEFSignature preprocessing.",
    )
    parser.add_argument(
        "--percentage_values", 
        type=str2bool, 
        default=False, 
        help="Whether to present DEF values as % or as scalars in [-1,+1].",
    )
    parser.add_argument(
        "--reset", 
        type=str2bool, 
        default=False, 
        help="Generate the dataset anew or retrieve it from Weave.",
    )
    parser.add_argument(
        "--num_train_examples",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--num_val_examples",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--num_test_examples",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    task_name=f"explain_def_task_with{'' if args.with_def_signature else 'out'}_def"
    if args.with_def_signature:
        task_name += f"_{'percentages' if args.percentage_values else 'scalars'}" 
    metadata = Metadata(
        task_name=task_name,
        with_def_signature=args.with_def_signature,
        percentage_values=args.percentage_values,
        num_train_examples=args.num_train_examples,
        num_val_examples=args.num_val_examples,
        num_test_examples=args.num_test_examples,
    )

    if args.reset:
        print("Reset flag detected. Generating dataset anew...")
        train_examples, \
        val_examples, \
        test_examples = create_dspy_dataset(
            metadata=metadata,
            json_path=Path(args.json_path),
            value_rules_path=Path(args.value_rules_path),
            difficulty_rules_path=Path(args.difficulty_rules_path),
            with_def_signature=args.with_def_signature,
            percentage_values=args.percentage_values,
        )
        print(f"Dataset created with {len(train_examples)} training examples and {len(val_examples)} validation examples.")
    else:
        print("Attempting to retrieve dataset from Weave...")
        try:
            train_dataset = weave.Dataset.get(f"{metadata.task_name}_train")
            val_dataset = weave.Dataset.get(f"{metadata.task_name}_val")
            test_dataset = weave.Dataset.get(f"{metadata.task_name}_test")
            print(f"Dataset retrieved:")
            print(f"- {len(train_dataset)} training examples;")
            print(f"- {len(val_dataset)} validation examples;")
            print(f"- {len(test_dataset)} testing examples.")
        except Exception as e:
            print(f"Failed to retrieve dataset: {e}")
    
if __name__ == "__main__":
    main()

