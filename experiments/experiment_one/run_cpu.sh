#!/bin/bash

# Now you can use $HUGGINGFACE_TOKEN
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Error: HUGGINGFACE_TOKEN is not set"
    exit 1
fi

python -m ipdb -c c run_seq_exp_cpu.py api --temperature=0.0 --max_tokens=4096

