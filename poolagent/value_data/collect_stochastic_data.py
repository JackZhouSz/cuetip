import os, json, shutil, random
import numpy as np
from datetime import datetime, timedelta
from poolagent.path import ROOT_DIR, DATA_DIR

TEST_SPLIT = 0.1

def is_valid_entry(entry):

    if all([v==1.0 for v in entry['estimates']]):
        return False

    return True

def is_folder_empty(folder_path):
    return len(os.listdir(folder_path)) == 0

def remove_empty_folders(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            folder_path = os.path.join(root, name)
            if is_folder_empty(folder_path):
                last_modified = os.path.getmtime(folder_path)
                if datetime.now() - datetime.fromtimestamp(last_modified) > timedelta(minutes=10):
                    print(f"Removing empty folder: {folder_path}")
                    shutil.rmtree(folder_path)

def calculate_relative_value(entry):
    W_0 = np.array(entry['initial_estimate'])
    sampled_win_probs = np.array(entry['estimates'])
    
    relative_improvement = (sampled_win_probs - W_0) 

    relative_improvement_mean = np.mean(relative_improvement)

    return relative_improvement_mean

def collect_json_files(base_path):
    all_data = []
        
    for json_file in os.listdir(base_path):
        if not json_file.endswith('.json'):
            continue
        
        json_path = os.path.join(base_path, json_file)

        with open(json_path, 'r') as f:
            file_data = json.load(f)
            
            if isinstance(file_data, list):
                for entry in file_data:
                    if is_valid_entry(entry):
                        entry['relative_improvement'] = calculate_relative_value(entry)
                        all_data.append(entry)
                    else:
                        print(f"Skipped invalid entry in {json_path}")
            elif isinstance(file_data, dict):
                if is_valid_entry(file_data):
                    file_data['relative_improvement'] = calculate_relative_value(file_data)
                    all_data.append(file_data)
                else:
                    print(f"Skipping invalid entry in {json_path}")
            else:
                print(f"Unexpected data format in {json_path}")
                
    return all_data

def create_train_test_split(data):
    random.shuffle(data)
    split_index = int(len(data) * (1 - TEST_SPLIT))
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data

def main():
    base_path = ROOT_DIR + "/poolagent/value_data/logs/stochastic/"
    
    all_data = collect_json_files(base_path)
    
    train_data, test_data = create_train_test_split(all_data)
    
    output_path = DATA_DIR + "/poolmaster_training_data.json"
    with open(output_path, 'w') as f:
        json.dump({
            "train": train_data,
            "test": test_data
        }, f, indent=4)
    
    print(f"Data collection complete. Train and test data written to {output_path}")
    print(f"Total number of valid entries:")
    print(f"  Train: {len(train_data)}")
    print(f"  Test: {len(test_data)}")

    print("Cleaning up empty folders...")
    remove_empty_folders(base_path)
    print("Cleanup complete.")

if __name__ == "__main__":
    main()
