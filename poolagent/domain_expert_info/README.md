# Domain Expert Information

This folder contains the scripts that make use of the domain expert functions and difficulty + value data. Importantly, this folder contains all of the fixed function definitions and the scripts for annotating the saved data with the function outputs.

## annotate_data.py

This script performs two functions:
    1. Run all of the 16 value functions and 17 difficulty functions on all of the current value and difficulty data in the data directory.
    2. Collect the averages and standard deviations of the function outputs and save them in DATA_DIR/fixed_function_averages.json. This is used in the heatmap visualisations of run_xai_agent.py.

## fixed_rules_evaluator.py

This file contains the Evaluator class that loads the fixed functions and executes them on provided states. It is used by agents/fixed_rule_agent.py and annotate_data.py.

## train_model.py

This script contains starts the model training (and hyperparamater search) for the mapping network between function output and estimated value/difficulty. There is a simple MLP network and an attempt at a self-attention network, neither do amazingly well currently but they are good enough for now.

## training.py 

This file contains the model training code and sweep configurations.