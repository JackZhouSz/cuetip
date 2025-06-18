import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from poolagent import (
    load_data, prepare_data, create_model, evaluate_model,
    VALUE_INPUT_SIZE, DIFFICULTY_INPUT_SIZE, OUTPUT_BINS,
    DATA_DIR, device, LOSS_FN
)

DATA_FILE_NAME = "ablation_results"
N_RUNS = 100
N_EPOCHS = 50

def create_dataloaders(value_X, difficulty_X, y, batch_size, test_size=0.1):
    X_train_val, X_test_val, X_train_diff, X_test_diff, y_train, y_test = train_test_split(
        value_X, difficulty_X, y, test_size=test_size, random_state=42)
    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_train_val), 
            torch.FloatTensor(X_train_diff), 
            torch.FloatTensor(y_train), 
        ), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_test_val), 
            torch.FloatTensor(X_test_diff), 
            torch.FloatTensor(y_test),
        ), batch_size=batch_size)
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, config):
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    best_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        num_batches = 0
        
        for inputs_value, inputs_difficulty, targets in train_loader:
            inputs_value, inputs_difficulty, targets = inputs_value.to(device), inputs_difficulty.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs_value, inputs_difficulty)
            loss = LOSS_FN(outputs.log(), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches
        scheduler.step()
        
        test_loss, test_exp_value_diff = evaluate_model(model, test_loader, LOSS_FN)
        
        if test_loss < best_loss:
            best_loss = test_loss

        print(f"Epoch [{epoch+1}/{config['epochs']}]")
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"Test Expected Value Difference: {test_exp_value_diff:.4f}")

    return best_loss

def create_holdout_data(value_X, difficulty_X, feature_index, is_value_feature):
    if is_value_feature:
        avg_value = np.mean(value_X[:, feature_index])
        value_X_holdout = value_X.copy()
        value_X_holdout[:, feature_index] = avg_value
        return value_X_holdout, difficulty_X
    else:
        avg_value = np.mean(difficulty_X[:, feature_index])
        difficulty_X_holdout = difficulty_X.copy()
        difficulty_X_holdout[:, feature_index] = avg_value
        return value_X, difficulty_X_holdout

def load_or_create_results():
    results_file = f"{DATA_DIR}/{DATA_FILE_NAME}.json"
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        # Retroactively calculate averages and std_devs if missing
        for key, value in results.items():
            if isinstance(value, list):
                results[key] = {
                    'runs': value,
                    'average': np.mean(value),
                    'std_dev': np.std(value)
                }
            elif isinstance(value, dict) and 'std_dev' not in value:
                results[key]['std_dev'] = np.std(value['runs'])
        return results
    return {}

def save_results(results):
    with open(f"{DATA_DIR}/{DATA_FILE_NAME}.json", 'w') as f:
        json.dump(results, f, indent=2)

def update_results(results, key, value):
    if key not in results:
        results[key] = {'runs': [], 'average': 0, 'std_dev': 0}
    results[key]['runs'].append(value)
    results[key]['average'] = np.mean(results[key]['runs'])
    results[key]['std_dev'] = np.std(results[key]['runs'])
    save_results(results)

def train_holdout_models():
    # Load and prepare data
    training_data = load_data(f"{DATA_DIR}/poolmaster_training_data.json")
    value_X, difficulty_X, _, y = prepare_data(training_data)

    config = {
        "model_type": ["attention"], 
        "hidden_size": 256, 
        "heads": 2, 
        "layers": 6,
        "lr": 5e-05, 
        "batch_size": 64, 
        "epochs": N_EPOCHS
    }

    # Load existing results or create new dictionary
    results = load_or_create_results()
    save_results(results)

    # Train base model
    if 'base_model' not in results or len(results['base_model']['runs']) < N_RUNS:
        print(f"Training base model (runs completed: {len(results.get('base_model', {}).get('runs', []))}/{N_RUNS})...")
        while len(results.get('base_model', {}).get('runs', [])) < N_RUNS:
            base_model = create_model(
                model_type=config['model_type'], 
                input_type="combined", 
                value_input_size=VALUE_INPUT_SIZE,
                difficulty_input_size=DIFFICULTY_INPUT_SIZE,
                hidden_size=config['hidden_size'], 
                output_size=OUTPUT_BINS,
                heads=config['heads'],
                dropout=0.25, 
            ).to(device)
            base_model.fit_scaler(value_X, difficulty_X)
            train_loader, test_loader = create_dataloaders(value_X, difficulty_X, y, config['batch_size'])
            base_loss = train_model(base_model, train_loader, test_loader, config)
            update_results(results, 'base_model', base_loss)
            print(f"Base model run {len(results['base_model']['runs'])}/{N_RUNS} completed")

    # Train models with value features held out
    for i in range(VALUE_INPUT_SIZE):
        key = f'value_holdout_{i}'
        if key not in results or len(results[key]['runs']) < N_RUNS:
            print(f"Training model with value feature {i} held out (runs completed: {len(results.get(key, {}).get('runs', []))}/{N_RUNS})...")
            while len(results.get(key, {}).get('runs', [])) < N_RUNS:
                value_X_holdout, _ = create_holdout_data(value_X, difficulty_X, i, True)
                model = create_model(
                    model_type=config['model_type'], 
                    input_type="combined", 
                    value_input_size=VALUE_INPUT_SIZE,
                    difficulty_input_size=DIFFICULTY_INPUT_SIZE,
                    hidden_size=config['hidden_size'], 
                    output_size=OUTPUT_BINS,
                    heads=config['heads'],
                    dropout=0.25, 
                ).to(device)
                model.fit_scaler(value_X, difficulty_X)  # Use original data for scaling
                train_loader, test_loader = create_dataloaders(value_X_holdout, difficulty_X, y, config['batch_size'])
                loss = train_model(model, train_loader, test_loader, config)
                update_results(results, key, loss)
                print(f"Value holdout {i} run {len(results[key]['runs'])}/{N_RUNS} completed")

    # Train models with difficulty features held out
    for i in range(DIFFICULTY_INPUT_SIZE):
        key = f'difficulty_holdout_{i}'
        if key not in results or len(results[key]['runs']) < N_RUNS:
            print(f"Training model with difficulty feature {i} held out (runs completed: {len(results.get(key, {}).get('runs', []))}/{N_RUNS})...")
            while len(results.get(key, {}).get('runs', [])) < N_RUNS:
                _, difficulty_X_holdout = create_holdout_data(value_X, difficulty_X, i, False)
                model = create_model(
                    model_type=config['model_type'], 
                    input_type="combined", 
                    value_input_size=VALUE_INPUT_SIZE,
                    difficulty_input_size=DIFFICULTY_INPUT_SIZE,
                    hidden_size=config['hidden_size'], 
                    output_size=OUTPUT_BINS,
                    heads=config['heads'],
                    dropout=0.25, 
                ).to(device)
                model.fit_scaler(value_X, difficulty_X)  # Use original data for scaling
                train_loader, test_loader = create_dataloaders(value_X, difficulty_X_holdout, y, config['batch_size'])
                loss = train_model(model, train_loader, test_loader, config)
                update_results(results, key, loss)
                print(f"Difficulty holdout {i} run {len(results[key]['runs'])}/{N_RUNS} completed")

    print(f"All models trained. Results saved to {DATA_FILE_NAME}.json")

if __name__ == "__main__":
    train_holdout_models()