import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

def extract_reasoning_data(data_dir: str = "./reasoning_data") -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """
    Extract LLM reasoning data from JSON files in the specified directory.
    
    Args:
        data_dir: Directory containing JSON files
        
    Returns:
        Dictionary organized by llm_name and agent_name with lists of reasoning
    """
    # Structure: {llm_name: {agent_name: {'suggest_reasoning': [], 'choose_reasoning': []}}}
    extracted_data = defaultdict(lambda: defaultdict(lambda: {'suggest_reasoning': [], 'choose_reasoning': []}))
    
    # Process each JSON file in the directory
    for json_file in Path(data_dir).glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                agent_data = json.load(f)

            print(f"Processing {json_file}...")
                
            # Process each entry in the data
            for entry in agent_data.get('entries', []):
                llm_name = entry.get('llm')
                agent_name = entry.get('agent')
                
                if not (llm_name and agent_name):
                    print(f"Skipping entry in {json_file} with missing llm or agent name")
                    continue
                
                # Extract reasoning from lm_chooser
                if 'lm_chooser' in entry:
                    reasoning = entry['lm_chooser']['response'].get('reasoning')
                    if reasoning:
                        
                        if '\n\nReasoning: ' in reasoning:
                            reasoning = reasoning.split('\n\nReasoning: ')[1]

                        extracted_data[llm_name][agent_name]['choose_reasoning'].append(reasoning)
                
                # Extract reasoning from lm_suggester
                if 'lm_suggester' in entry:
                    reasoning = entry['lm_suggester']['response'].get('reasoning')
                    if reasoning:

                        if '\n\nReasoning: ' in reasoning:
                            reasoning = reasoning.split('\n\nReasoning: ')[1]

                        extracted_data[llm_name][agent_name]['suggest_reasoning'].append(reasoning)
                        
        except json.JSONDecodeError as e:
            print(f"Error reading {json_file}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {json_file}: {e}")
    
    # Convert defaultdict to regular dict
    return {
        llm: {
            agent: dict(data)
            for agent, data in agents.items()
        }
        for llm, agents in extracted_data.items()
    }

def save_json_output(data: Dict, output_file: str = "clean_reasoning_data.json"):
    """
    Save the extracted data as a JSON file.
    
    Args:
        data: Dictionary of extracted reasoning data
        output_file: JSON file to save the data
    """
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    print("Extracting reasoning data...")
    data = extract_reasoning_data()
    
    # Print summary statistics
    total_suggest = sum(
        len(agent_data['suggest_reasoning'])
        for llm_data in data.values()
        for agent_data in llm_data.values()
    )
    total_choose = sum(
        len(agent_data['choose_reasoning'])
        for llm_data in data.values()
        for agent_data in llm_data.values()
    )
    
    print(f"\nFound {total_suggest} suggester reasoning entries and {total_choose} chooser reasoning entries")
    
    print("\nSaving JSON output...")
    save_json_output(data)
    print("Done! Output saved to clean_reasoning_data.json")

if __name__ == "__main__":
    main()