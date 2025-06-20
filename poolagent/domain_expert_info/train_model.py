import json
import wandb
from typing import Dict, Any
import torch
import argparse


from poolagent.domain_expert_info.training import (
    load_data,
    train_model,
    device,
    create_model,
    prepare_data,
    create_dataloaders,
    evaluate_model,
    LOSS_FN,
    VALUE_INPUT_SIZE,
    DIFFICULTY_INPUT_SIZE,
    OUTPUT_BINS
)
from poolagent.path import DATA_DIR

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate the configuration parameters."""
    required_params = {
        'model_type': str,
        'input_type': str,
        'hidden_size': int,
        'lr': float,
        'batch_size': int,
        'epochs': int,
        'dropout': float,
    }
    
    # Model-specific required parameters
    model_specific_params = {
        'attention': {'heads': int},
        'mlp': {'layers': int}
    }
    
    # Check base parameters
    for param, param_type in required_params.items():
        if param not in config:
            print(f"Missing required parameter: {param}")
            return False
        if not isinstance(config[param], param_type):
            print(f"Parameter {param} should be of type {param_type}")
            return False
    
    # Check model-specific parameters
    model_type = config['model_type']
    if model_type in model_specific_params:
        for param, param_type in model_specific_params[model_type].items():
            if param not in config:
                print(f"Missing required parameter for {model_type} model: {param}")
                return False
            if not isinstance(config[param], param_type):
                print(f"Parameter {param} should be of type {param_type}")
                return False
    
    # Validate ranges
    if config['batch_size'] <= 0:
        print("batch_size must be positive")
        return False
    if config['hidden_size'] <= 0:
        print("hidden_size must be positive")
        return False
    if config['lr'] <= 0:
        print("lr must be positive")
        return False
    if config['epochs'] <= 0:
        print("epochs must be positive")
        return False
    if config['dropout'] < 0 or config['dropout'] > 1:
        print("dropout must be between 0 and 1")
        return False
    
    return True

def train_best_model(training_data: list, config: Dict[str, Any], save_path: str, config_save_path: str, args: Dict[str, Any] = {}) -> None:
    """
    Train a model with the specified configuration.
    
    Args:
        training_data: List of training examples
        config: Model configuration dictionary
        save_path: Path to save the trained model
        args: Dictionnary of parsed argument.
    """
    print("Initializing training with configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # Initialize wandb
    if args.get('use_wandb', False):
        wandb.init(
            project="distribution_model_final",
            config=config,
            mode="online",  # Set to "disabled" to turn off wandb logging
        )
    
    # Create model
    model = create_model(
        model_type=config['model_type'],
        input_type=config['input_type'],
        value_input_size=VALUE_INPUT_SIZE,
        difficulty_input_size=DIFFICULTY_INPUT_SIZE,
        hidden_size=config['hidden_size'],
        output_size=OUTPUT_BINS,
        heads=config.get('heads'),  # Only for attention model
        dropout=config['dropout']
    ).to(device)
    
    # Prepare data
    value_X, difficulty_X, combined_X, y = prepare_data(training_data)
    
    # Initialize model scaler
    if config['input_type'] == 'separate':
        model.fit_scaler(value_X, difficulty_X)
    else:
        model.fit_scaler(combined_X)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'])
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        value_X, difficulty_X, y, 
        config['batch_size'], 
        config['input_type']
    )
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            if config['input_type'] == 'separate':
                inputs_value, inputs_difficulty, targets = batch
                inputs_value = inputs_value.to(device)
                inputs_difficulty = inputs_difficulty.to(device)
                outputs = model(inputs_value, inputs_difficulty)
            else:
                inputs, targets = batch
                inputs = inputs.to(device)
                outputs = model(inputs)
            
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = LOSS_FN(outputs.log(), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        train_loss /= num_batches
        scheduler.step()
        
        # Evaluation phase
        test_loss, test_exp_value_diff = evaluate_model(
            model, test_loader, LOSS_FN, config['input_type'])
        
        # Logging
        if args.get('use_wandb', False):
            wandb.log({
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_exp_value_diff": test_exp_value_diff
            })
        
        print(f"Epoch [{epoch+1}/{config['epochs']}]")
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"Test Expected Value Difference: {test_exp_value_diff:.4f}")
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), save_path)
            with open(config_save_path, 'w') as f:
                json.dump(config, f)
            print(f"Saved new best model with loss: {best_loss:.4f}")
    
    if args.get('use_wandb', False):
        wandb.finish()
    print("Training completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Neural Surrogate Model Training")
    parser.add_argument("--use_wandb", action='store_true', help="Record training in W&B.")
    args = vars(parser.parse_args())

    # Initialize wandb
    if args.get('use_wandb', False):
        # Initialize wandb
        wandb.login()
    
    # Load data
    print(f"Using device: {device}")
    print("Loading training data...")
    training_data = load_data(f"{DATA_DIR}/poolmaster_training_data.json")
    
    # Example configuration - replace with your best configuration
    config = {
        # Required parameters
        'model_type': 'attention',    # or 'mlp'
        'input_type': 'combined',     # or 'combined'
        'hidden_size': 256,           # e.g., 128, 256, 512
        'lr': 0.00005,                   # e.g., 0.0001, 0.0003
        'batch_size': 64,            # e.g., 64, 128, 256
        'epochs': 50,                 # e.g., 25, 50, 100
        'dropout': 0.25,               # e.g., 0.1, 0.2, 0.3
        
        # Model-specific parameters
        'heads': 2,                   # For attention model: 2, 4, 8, 16
        'layers': 6,                  # For MLP model: 3, 4, 5
    }
    
    # Paths for saving model and information
    save_dir = f"{DATA_DIR}/models"
    model_path = f"{save_dir}/best_model.pth"
    config_path = f"{save_dir}/best_config.json"
    
    try:
        # Validate configuration
        if not validate_config(config):
            print("Invalid configuration. Please check the parameters and try again.")
            exit(1)
        
        # Train model
        print("\nStarting model training...")
        train_best_model(training_data, config, model_path, config_path, args=args)

        print("\nModel training completed successfully!")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    
    finally:
        # Ensure proper cleanup
        if args.get('use_wandb', False):
            wandb.finish()
