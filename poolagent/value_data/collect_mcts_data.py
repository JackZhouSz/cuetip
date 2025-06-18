import os, json, shutil, random

from datetime import datetime, timedelta

from poolagent.path import ROOT_DIR, DATA_DIR

TEST_SPLIT = 0.1
PLAYERS = ['one'] #, 'two'

def is_valid_entry(entry):
    """
    Check if a single entry has a valid structure.
    """
    if not isinstance(entry, dict):
        return False
    
    if 'player' not in entry or entry['player'] not in PLAYERS:
        return False
    
    if 'follow_up_states' not in entry or 'visit_distribution' not in entry:
        return False
    
    if len(entry['follow_up_states']) != len(entry['visit_distribution']):
        return False
        
    if not entry['follow_up_states'][0]['params']:
        return False
    
    # if more than 3 of the top visit values are the same, skip
    top_visits = sorted(entry['visit_distribution'], reverse=True)[:3]
    if top_visits[0] == top_visits[1] and top_visits[1] == top_visits[2]:
        print(f"Skipping entry with top visits: {top_visits}")
        return False
    
    return True

def is_folder_empty(folder_path):
    """
    Check if a folder is empty.
    """
    return len(os.listdir(folder_path)) == 0

def remove_empty_folders(path):
    """
    Remove empty folders in the given path if the time it was last modified is more than 10 minutes ago.
    """
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            folder_path = os.path.join(root, name)
            if is_folder_empty(folder_path):
                last_modified = os.path.getmtime(folder_path)
                if datetime.now() - datetime.fromtimestamp(last_modified) > timedelta(minutes=10):
                    print(f"Removing empty folder: {folder_path}")
                    shutil.rmtree(folder_path)

def collect_json_files(base_path):
    all_data = []
        
    for json_file in os.listdir(base_path):
        if not json_file.endswith('.json'):
            continue
        
        json_path = os.path.join(base_path, json_file)

        with open(json_path, 'r') as f:
            file_data = json.load(f)
            
            # Check if file_data is a list or a single entry
            if isinstance(file_data, list):
                for entry in file_data:
                    if is_valid_entry(entry):
                        all_data.append(entry)
                    else:
                        print(f"Skipped invalid entry in {json_path}")
            elif isinstance(file_data, dict):
                if is_valid_entry(file_data):
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
    # Set your base path here
    base_path = ROOT_DIR + "/poolagent/value_data/logs/mcts/"

    
    all_data = collect_json_files(base_path)
    
    # Create train/test split
    train_data, test_data = create_train_test_split(all_data)
    
    # Write the train data to mcts_train_data.json
    output_path = DATA_DIR + "/mcts_training_data.json"
    with open(output_path, 'w') as f:
        json.dump({
            "train": train_data,
            "test": test_data
        }, f, indent=4)
    
    print(f"Data collection complete. Train and test data written to {output_path}")
    print(f"Total number of valid entries:")
    print(f"  Train: {len(train_data)}")
    print(f"  Test: {len(test_data)}")

    # Remove empty folders
    print("Cleaning up empty folders...")
    remove_empty_folders(base_path)
    print("Cleanup complete.")

if __name__ == "__main__":
    main()
