import wandb, torch, os

from typing import Dict, Any

from poolagent import load_data, train_best_model, validate_config, device
from poolagent.path import DATA_DIR

if __name__ == '__main__':
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
    os.makedirs(save_dir, exist_ok=True)
    model_path = f"{save_dir}/best_model.pth"
    config_path = f"{save_dir}/best_config.json"
    
    try:
        # Validate configuration
        if not validate_config(config):
            print("Invalid configuration. Please check the parameters and try again.")
            exit(1)
        
        # Train model
        print("\nStarting model training...")
        train_best_model(training_data, config, model_path, config_path)

        print("\nModel training completed successfully!")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    
    finally:
        # Ensure proper cleanup
        wandb.finish()
