import os, time
import numpy as np

from typing import List, Dict

from poolagent.path import ROOT_DIR
from poolagent.utils import State, Event, safe_exec_function

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

class FixedRulesEvaluator:
    def __init__(self):
        self.value_functions = []
        self.difficulty_functions = []
        self.load_functions(f"{ROOT_DIR}/poolagent/domain_expert_info")
        self.value_function_times = {}
        self.difficulty_function_times = {}

    def load_functions(self, base_path: str):
        value_path = os.path.join(base_path, 'value_functions')
        difficulty_path = os.path.join(base_path, 'difficulty_functions')

        self.value_functions = self.load_functions_from_folder(value_path)
        self.difficulty_functions = self.load_functions_from_folder(difficulty_path)

    def load_functions_from_folder(self, folder_path: str) -> List[str]:
        functions = []
        filenames = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
        for filename in filenames:
            with open(os.path.join(folder_path, filename), 'r') as f:
                function_code = f.read()
                functions.append(function_code)
        return functions

    def evaluate_state(self, starting_state: State, shot: dict, final_state: State, target_balls: List[str]) -> float:
        args = [starting_state, shot, final_state, target_balls]
        values = []

        for i, func in enumerate(self.value_functions):

            start_time = time.time()
            result = safe_exec_function(func, args)[0]
            end_time = time.time()

            execution_time = end_time - start_time
            self.value_function_times[i] = self.value_function_times.get(i, {'total_time': 0, 'count': 0})
            self.value_function_times[i]['total_time'] += execution_time
            self.value_function_times[i]['count'] += 1

            if result is None or np.isnan(result):
                print(f"Value function {i+1} produced NaN values")
                print(func)
                print(safe_exec_function(func, args))
                print(args)
                raise ValueError(f"NaN values found in value estimation")
            
            values.append(float(result))

        return values

    def evaluate_difficulty(self, state: State, shot: Dict[str, float], events: List[Event], target_balls: List[str]) -> List[float]:
        args = [state, shot, events, target_balls]
        difficulties = []

        for i, func in enumerate(self.difficulty_functions):
            start_time = time.time()
            result = safe_exec_function(func, args)[0]
            end_time = time.time()

            execution_time = end_time - start_time
            self.difficulty_function_times[i] = self.difficulty_function_times.get(i, {'total_time': 0, 'count': 0})
            self.difficulty_function_times[i]['total_time'] += execution_time
            self.difficulty_function_times[i]['count'] += 1

            if result is None or np.isnan(result):
                print(f"Difficulty function {i+1} produced NaN values")
                print(func)
                print(safe_exec_function(func, args))
                raise ValueError(f"NaN values found in difficulty estimation")
            
            difficulties.append(float(result))

        return difficulties

    def get_average_execution_times(self):
        value_times = {f"value_{i+1}": data['total_time'] / data['count'] if data['count'] > 0 else 0 
                       for i, data in self.value_function_times.items()}
        difficulty_times = {f"difficulty_{i+1}": data['total_time'] / data['count'] if data['count'] > 0 else 0 
                            for i, data in self.difficulty_function_times.items()}
        return {**value_times, **difficulty_times}
