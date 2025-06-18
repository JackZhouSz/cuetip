import wandb
from functools import partial
import itertools
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from poolagent.domain_expert_info.training import (
    load_data, 
    train_model, 
    device, 
    N_TRIALS,
    VALUE_INPUT_SIZE,
    DIFFICULTY_INPUT_SIZE
)
from poolagent.path import DATA_DIR

# Define parameter spaces for each model type
base_params = {
    'batch_size': {'values': [64, 128, 256]},
    'lr': {'values': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]},
    'hidden_size': {'values': [128, 256, 512]},
    'epochs': {'value': 25},
    'dropout': {'values': [0.1, 0.2, 0.3]},
}

model_specific_params = {
    'attention': {
        'heads': {'values': [2, 4, 8, 12]},
    },
    'mlp': {
        'layers': {'values': [3, 4, 5, 6]},
    }
}

input_types = ['separate', 'combined']

def create_sweep_configs() -> Dict[str, Dict[str, Any]]:
    """Create separate sweep configurations for each model type and input type combination."""
    sweep_configs = {}
    
    for model_type in ['attention', 'mlp']:
        for input_type in input_types:
            # Create a unique name for this configuration
            config_name = f"{model_type}_{input_type}"
            
            # Start with base parameters
            params = base_params.copy()
            
            # Add model-specific parameters
            params.update(model_specific_params[model_type])
            
            # Add fixed parameters for this configuration
            params['model_type'] = {'value': model_type}
            params['input_type'] = {'value': input_type}
            
            # Create the sweep configuration
            sweep_configs[config_name] = {
                'method': 'bayes',
                'metric': {
                    'name': 'test_exp_value_diff',
                    'goal': 'minimize'
                },
                'parameters': params
            }
    
    return sweep_configs

def run_all_sweeps(training_data):
    """Run sweeps for all model configurations and track results."""
    sweep_configs = create_sweep_configs()
    sweep_results = {}
    
    for config_name, sweep_config in sweep_configs.items():
        print(f"\nStarting sweep for configuration: {config_name}")
        
        # Initialize project with specific model configuration
        project_name = f"distribution_model_{config_name}"
        
        # Create the sweep
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        
        # Create a partial function with the training data
        train_model_with_args = partial(train_model, training_data)
        
        # Calculate number of trials for this configuration
        total_combinations = 1
        for param in sweep_config['parameters'].values():
            if 'values' in param:
                total_combinations *= len(param['values'])
        
        # Ensure minimum number of trials
        trials = max(N_TRIALS, total_combinations)
        
        # Run the sweep
        wandb.agent(sweep_id, function=train_model_with_args, count=trials)
        
        # Get the best run for this configuration
        api = wandb.Api()
        sweep = api.sweep(f"{project_name}/{sweep_id}")
        best_run = sweep.best_run()
        
        # Store results
        sweep_results[config_name] = {
            'sweep_id': sweep_id,
            'best_run_id': best_run.id,
            'best_score': best_run.summary.get('best_loss', float('inf')),
            'config': best_run.config
        }
        
        # Clean up
        wandb.finish()
    
    return sweep_results

def analyze_results(sweep_results):
    """Analyze and print results from all sweeps."""
    print("\nSweep Results Summary:")
    print("=" * 80)
    
    # Sort configurations by best score
    sorted_results = sorted(
        sweep_results.items(),
        key=lambda x: x[1]['best_score']
    )
    
    for config_name, result in sorted_results:
        print(f"\nConfiguration: {config_name}")
        print(f"Best Score: {result['best_score']:.4f}")
        print("Best Configuration:")
        for param, value in result['config'].items():
            print(f"  {param}: {value}")
        print("-" * 40)
    
    # Identify overall best configuration
    best_config = sorted_results[0]
    print("\nOverall Best Configuration:")
    print(f"Model: {best_config[0]}")
    print(f"Score: {best_config[1]['best_score']:.4f}")
    
    return best_config

def save_best_config(best_config):
    """Save the best configuration to a file."""
    import json
    
    config_data = {
        'model_type': best_config[0],
        'score': best_config[1]['best_score'],
        'parameters': best_config[1]['config']
    }
    
    with open(f"{DATA_DIR}/models/best_sweep_config.json", 'w') as f:
        json.dump(config_data, f, indent=2)

if __name__ == '__main__':
    # Initialize wandb
    wandb.login()
    print(f"Using device: {device}")

    # Load data
    print("Loading training data...")
    training_data = load_data(f"{DATA_DIR}/poolmaster_training_data.json")

    try:
        # Run all sweeps
        print("\nStarting sweeps for all configurations...")
        sweep_results = run_all_sweeps(training_data)
        
        # Analyze results
        print("\nAnalyzing results...")
        best_config = analyze_results(sweep_results)
        
        # Save best configuration
        print("\nSaving best configuration...")
        save_best_config(best_config)
        
        print("\nSweep completed successfully!")
        
    except Exception as e:
        print(f"\nError during sweep: {str(e)}")
        raise
    
    finally:
        # Ensure proper cleanup
        wandb.finish()