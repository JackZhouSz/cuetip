#!/bin/bash

# Now you can use $HUGGINGFACE_TOKEN
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Error: HUGGINGFACE_TOKEN is not set"
    exit 1
fi

if [ -z "$API_KEY" ]; then
    echo "Error: OpenAI's API_KEY is not set"
    exit 1
fi

python -m ipdb -c c gen_dataset.py api --n_threads=1 --n_gpus=0 --temperature=0.0 --max_tokens=4096

