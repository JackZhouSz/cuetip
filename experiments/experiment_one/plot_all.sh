#!/bin/bash

python plot_grid.py 
python plot_grid.py --model Meta-Llama-3.1-70B-Instruct-Turbo
python plot_scatter.py 
python plot_single_model.py --model_name Meta-Llama-3.1-70B-Instruct-Turbo
python plot_ball_diff.py --model_name Meta-Llama-3.1-70B-Instruct-Turbo
python plot_game_lengths.py --model_name Meta-Llama-3.1-70B-Instruct-Turbo