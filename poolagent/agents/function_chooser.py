import torch, json
import numpy as np

from typing import List, Tuple

from poolagent.pool import Fouls
from poolagent.path import DATA_DIR
from poolagent.domain_expert_info import FixedRulesEvaluator, CombinedModel, AttentionModel, VALUE_INPUT_SIZE, DIFFICULTY_INPUT_SIZE, create_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model:
    def __init__(self):

        self.OUTPUT_BINS = 11

        with open(f"{DATA_DIR}/models/best_config.json", 'r') as f:
            self.config = json.load(f)

        self.mapper = create_model(
            model_type=self.config['model_type'],
            input_type=self.config['input_type'],
            value_input_size=VALUE_INPUT_SIZE,
            difficulty_input_size=DIFFICULTY_INPUT_SIZE,
            hidden_size=self.config['hidden_size'],
            output_size=self.OUTPUT_BINS,
            heads=self.config.get('heads', 1),
        )

        self.mapper.load_state_dict(torch.load(f"{DATA_DIR}/models/best_model.pth"))
        self.mapper.to(device)
        self.mapper.eval()

    def fit_scaler(self, v_data, d_data):
        self.mapper.fit_scaler(v_data, d_data)

    def __call__(self, values, difficulties):

        self.mapper.to(device)
        
        y = []

        for v, d in zip(values, difficulties):
            v = torch.from_numpy(v).float().to(device).unsqueeze(0)
            d = torch.from_numpy(d).float().to(device).unsqueeze(0)

            y.append(self.mapper.forward(v, d).cpu().detach().numpy())

        return np.array(y)

class FunctionChooser():

    def __init__(self, target_balls):

        self.target_balls = target_balls

        self.evaluator = FixedRulesEvaluator()
        self.model = Model()
        self.fit_scaler(self.model)

        self.record = {}

        with open(f"{DATA_DIR}/fixed_function_averages.json", 'r') as f:
            self.avg_std = json.load(f)

    def __call__(self, starting_state, final_states, shot_params, shot_events):
        
        best_shot_index, model_distributions, expected_values, raw_values, raw_difficulties = self.evaluate_shots(starting_state, shot_params, shot_events, final_states)

        return best_shot_index 
    
    def fit_scaler(self, model):
        from poolagent.domain_expert_info.training import load_data, prepare_data
        training_data = load_data(f"{DATA_DIR}/poolmaster_training_data.json")
        value_X, difficulty_X, _, _ = prepare_data(training_data)
        model.fit_scaler(value_X.reshape(-1, VALUE_INPUT_SIZE), difficulty_X.reshape(-1, DIFFICULTY_INPUT_SIZE))

    def evaluate_shots(self, state, shots, events, states, fouls=[], weight_distribution=None) -> Tuple[int, List[float], List[float], List[float]]:
        if fouls == []:
            fouls = [Fouls.NONE] * len(shots)

        values = [self.evaluator.evaluate_state(state, shot, final_state, self.target_balls) for shot, final_state in zip(shots, states)]
        difficulties = [self.evaluator.evaluate_difficulty(state, shot, e, self.target_balls) for shot, e in zip(shots, events)]

        raw_values = np.array(values)
        raw_difficulties = np.array(difficulties)

        with torch.no_grad():
            model_distributions = self.model(raw_values, raw_difficulties)

        expected_values = []
        for dist in model_distributions:
            weight_distribution = weight_distribution if weight_distribution is not None else np.ones(self.model.OUTPUT_BINS)
            expected_values.append(np.sum(dist * np.linspace(0, 1, self.model.OUTPUT_BINS) * weight_distribution))
        expected_values = [v*(1 if f == Fouls.NONE else 0) for v, f in zip(expected_values, fouls)]

        best_shot_index = int(np.argmax(expected_values))

        self.record = {
            "values": list(raw_values.tolist()),
            "difficulties": list(raw_difficulties.tolist()),
            "model_distributions": list(model_distributions.tolist()),
            "best_shot_index": best_shot_index
        }

        return best_shot_index, model_distributions, expected_values, raw_values, raw_difficulties
    
    def normalise(self, values, difficulties):
        avg_std_values = np.array(self.avg_std['value'])
        avg_std_difficulties = np.array(self.avg_std['difficulty'])

        # Normalise values and difficulties using averages and standard deviations
        values = (np.array(values) - avg_std_values[:,0]) / ( avg_std_values[:,1] + 1e-6) 
        difficulties = (np.array(difficulties) - avg_std_difficulties[:,0]) / ( avg_std_difficulties[:,1] + 1e-6) 

        values = 2 / (1 + np.exp(-2*values)) - 1
        difficulties = 2 / (1 + np.exp(-2*difficulties)) - 1

        return values, difficulties