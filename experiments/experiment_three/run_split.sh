#!/bin/bash

# Define array of model names
MODELS=(
    "together/meta-llama/Llama-3.2-3B-Instruct-Turbo"
    "together/meta-llama/Llama-3-8b-chat-hf"
    #"together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    "together/meta-llama/Llama-3.3-70B-Instruct-Turbo"
)

# Check if argument is provided
if [ -z "$1" ]; then
    echo "Error: Please provide a model index (0-$((${#MODELS[@]}-1)))"
    echo "Available models:"
    for i in "${!MODELS[@]}"; do
        echo "$i: ${MODELS[$i]}"
    done
    exit 1
fi

# Validate input is a number
if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Error: Argument must be a number"
    exit 1
fi

# Check if index is valid
if [ "$1" -ge "${#MODELS[@]}" ]; then
    echo "Error: Index out of range. Maximum index is $((${#MODELS[@]}-1))"
    exit 1
fi

# Get selected model
MODEL="${MODELS[$1]}"

# The command to run and monitor
CMD="python gen_seq_dataset.py --k_explanations 3 --temperature 0.05 --model \"$MODEL\" && python collect_data.py"

echo "Selected model: $MODEL"
echo "Running command: $CMD"

eval "$CMD"