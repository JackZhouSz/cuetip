import json, torch, wandb
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from poolagent.path import DATA_DIR

N_TRIALS = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_SWEEP_LOSS = float('inf')
VALUE_INPUT_SIZE = 13
DIFFICULTY_INPUT_SIZE = 16
COMBINED_INPUT_SIZE = VALUE_INPUT_SIZE + DIFFICULTY_INPUT_SIZE
OUTPUT_BINS = 11

LOSS_FN = nn.KLDivLoss(reduction="batchmean")

# Updated sweep configuration to include MLP model
sweep_configuration = {
    'method': 'bayes',
    'metric': {'name': 'test_exp_value_diff', 'goal': 'minimize'},
    'parameters': {
        'model_type': {'values': ['attention', 'mlp']},
        'input_type': {'values': ['separate', 'combined']},
        'hidden_size': {'values': [128, 256, 512]},
        'heads': {'values': [2, 4, 8, 16]},
        'lr': {'min': 1e-5, 'max': 1e-3},
        'batch_size': {'values': [64, 128, 256]},
        'epochs': 25
    }
}

class SelfAttention(nn.Module):
    def __init__(self, input_size, heads):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.heads = heads
        self.head_dim = input_size // heads

        assert (self.head_dim * heads == input_size), "Input size needs to be divisible by heads"

        self.values = nn.Linear(input_size, input_size, bias=False)
        self.keys = nn.Linear(input_size, input_size, bias=False)
        self.queries = nn.Linear(input_size, input_size, bias=False)
        self.fc_out = nn.Linear(input_size, input_size)

    def forward(self, x):
        N = x.shape[0]
        
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)
        
        values = values.reshape(N, self.heads, self.head_dim)
        keys = keys.reshape(N, self.heads, self.head_dim)
        queries = queries.reshape(N, self.heads, self.head_dim)

        energy = torch.bmm(queries, keys.transpose(1, 2))
        attention = torch.softmax(energy / (self.input_size ** (1/2)), dim=2)
        
        out = torch.bmm(attention, values).reshape(N, self.input_size)
        out = self.fc_out(out)
        return out

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=4, dropout=0.1):
        super(MLPModel, self).__init__()
        self.scaler = RobustScaler()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        for _ in range(int(n_layers) - 2):
            self.mlp.add_module(f"hidden_{_}", nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        self.mlp.add_module("output", nn.Linear(hidden_size, output_size))

    def forward(self, x):
        x = self.scaler.transform(x.cpu().numpy())
        x = torch.from_numpy(x).float().to(device)
        return self.mlp(x)

    def fit_scaler(self, data):
        self.scaler.fit(data)

class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, heads, dropout):
        super(AttentionModel, self).__init__()
        self.scaler = RobustScaler()

        self.input_fc = nn.Linear(input_size, hidden_size)
        self.attention1 = SelfAttention(hidden_size, heads)
        self.attention2 = SelfAttention(hidden_size, heads)
        self.attention3 = SelfAttention(hidden_size, heads)
        self.norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.scaler.transform(x.cpu().numpy())
        x = torch.from_numpy(x).float().to(device)

        x = self.input_fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.norm(self.attention1(x) + x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.norm(self.attention2(x) + x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.norm(self.attention3(x) + x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc_out(x)
        
        return x

    def fit_scaler(self, data):
        self.scaler.fit(data)

class CombinedModel(nn.Module):
    def __init__(self, value_model, difficulty_model, hidden_size, output_size, input_type='separate'):
        super(CombinedModel, self).__init__()
        self.input_type = input_type
        
        if input_type == 'separate':
            self.value_model = value_model
            self.difficulty_model = difficulty_model
            input_size = hidden_size * 2
        else:
            self.combined_model = value_model  # Only use one model for combined input
            input_size = hidden_size
            
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, value_input, difficulty_input=None):
        if self.input_type == 'separate':
            value_output = self.value_model(value_input)
            difficulty_output = self.difficulty_model(difficulty_input)
            combined = torch.cat((value_output, difficulty_output), dim=1)
        else:
            # For combined input, concatenate inputs before processing
            combined_input = torch.cat((value_input, difficulty_input), dim=1) if difficulty_input is not None else value_input
            combined = self.combined_model(combined_input)
            
        output = self.mlp(combined)
        return nn.functional.softmax(output, dim=1)
    
    def fit_scaler(self, value_data, difficulty_data=None):
        if self.input_type == 'separate':
            self.value_model.fit_scaler(value_data)
            self.difficulty_model.fit_scaler(difficulty_data)
        else:
            # For combined input, concatenate data before fitting
            combined_data = np.concatenate([value_data, difficulty_data], axis=1) if difficulty_data is not None else value_data
            self.combined_model.fit_scaler(combined_data)

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)['train']

def prepare_data(training_data):
    value_x = np.array([entry['values'] for entry in training_data], dtype=np.float32)
    difficulty_x = np.array([entry['difficulties'] for entry in training_data], dtype=np.float32)
    combined_x = np.concatenate([value_x, difficulty_x], axis=1)
    y = np.array([entry['distribution'] for entry in training_data], dtype=np.float32)
    return value_x, difficulty_x, combined_x, y

def create_dataloaders(value_X, difficulty_X, y, batch_size, input_type='separate', test_size=0.1):
    if input_type == 'separate':
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
    else:
        combined_X = np.concatenate([value_X, difficulty_X], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            combined_X, y, test_size=test_size, random_state=42)
        
        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train),
            ), batch_size=batch_size, shuffle=True)
        
        test_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_test),
                torch.FloatTensor(y_test),
            ), batch_size=batch_size)
    
    return train_loader, test_loader

def evaluate_model(model, data_loader, criterion, input_type='separate'):
    model.eval()
    total_loss = 0
    num_batches = 0
    total_exp_value_diff = 0

    average_output = np.zeros(OUTPUT_BINS)
    average_target = np.zeros(OUTPUT_BINS)
    
    with torch.no_grad():
        for batch in data_loader:
            if input_type == 'separate':
                inputs_value, inputs_difficulty, targets = batch
                inputs_value, inputs_difficulty = inputs_value.to(device), inputs_difficulty.to(device)
                outputs = model(inputs_value, inputs_difficulty)
                average_output += outputs.mean(dim=0).cpu().numpy()
            else:
                inputs, targets = batch
                inputs = inputs.to(device)
                outputs = model(inputs)
                average_output += outputs.mean(dim=0).cpu().numpy()
                
            targets = targets.to(device)
            average_target += targets.mean(dim=0).cpu().numpy()
            loss = criterion(outputs.log(), targets)
            total_loss += loss.item() #/ OUTPUT_BINS
            
            # Calculate expected values
            bin_midpoints = torch.linspace(0, 1, OUTPUT_BINS).to(device)
            pred_exp_value = torch.sum(outputs * bin_midpoints, dim=1)
            target_exp_value = torch.sum(targets * bin_midpoints, dim=1)
            
            exp_value_diff = torch.abs(pred_exp_value - target_exp_value).mean()
            total_exp_value_diff += exp_value_diff.item()
            
            num_batches += 1
    
    average_output /= num_batches
    average_target /= num_batches
    print(f"Average output - Average target")
    for i in range(OUTPUT_BINS):
        print(f"{average_output[i]:.4f} - {average_target[i]:.4f}")

    return total_loss / num_batches, total_exp_value_diff / num_batches

def create_model(model_type, input_type, value_input_size, difficulty_input_size, hidden_size, output_size, heads=None, dropout=0.1):
    if model_type == 'mlp':
        if input_type == 'separate':
            value_model = MLPModel(value_input_size, hidden_size, hidden_size, dropout)
            difficulty_model = MLPModel(difficulty_input_size, hidden_size, hidden_size, dropout)
        else:
            combined_input_size = value_input_size + difficulty_input_size
            value_model = MLPModel(combined_input_size, hidden_size, hidden_size, dropout)
            difficulty_model = None
    else:  # attention
        if input_type == 'separate':
            value_model = AttentionModel(value_input_size, hidden_size, hidden_size, heads, dropout)
            difficulty_model = AttentionModel(difficulty_input_size, hidden_size, hidden_size, heads, dropout)
        else:
            combined_input_size = value_input_size + difficulty_input_size
            value_model = AttentionModel(combined_input_size, hidden_size, hidden_size, heads, dropout)
            difficulty_model = None
            
    return CombinedModel(value_model, difficulty_model, hidden_size, output_size, input_type)

def train_model(training_data, config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        value_X, difficulty_X, combined_X, y = prepare_data(training_data)
        
        model = create_model(
            config.model_type,
            config.input_type,
            VALUE_INPUT_SIZE,
            DIFFICULTY_INPUT_SIZE,
            config.hidden_size,
            OUTPUT_BINS,
            config.heads if config.model_type == 'attention' else None
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")

        if config.input_type == 'separate':
            model.fit_scaler(value_X, difficulty_X)
        else:
            model.fit_scaler(combined_X)

        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        
        train_loader, test_loader = create_dataloaders(
            value_X, difficulty_X, y, config.batch_size, config.input_type)

        best_loss = float('inf')
        for epoch in range(config.epochs):
            model.train()
            train_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                if config.input_type == 'separate':
                    inputs_value, inputs_difficulty, targets = batch
                    inputs_value, inputs_difficulty = inputs_value.to(device), inputs_difficulty.to(device)
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
            
            test_loss, test_exp_value_diff = evaluate_model(
                model, test_loader, LOSS_FN, config.input_type)
            
            wandb.log({
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_exp_value_diff": test_exp_value_diff
            })

            print(f"Epoch [{epoch+1}/{config.epochs}]")
            print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            print(f"Test Expected Value Difference: {test_exp_value_diff:.4f}")

            if test_loss < best_loss:
                best_loss = test_loss
                global BEST_SWEEP_LOSS
                if best_loss < BEST_SWEEP_LOSS:
                    BEST_SWEEP_LOSS = best_loss
                    
                    with open(f"{DATA_DIR}/models/config.json", 'w') as f:
                        json.dump(config.as_dict(), f)
                    torch.save(model.state_dict(), f"{DATA_DIR}/models/combined_model.pth")
        
        wandb.log({"best_loss": best_loss})
