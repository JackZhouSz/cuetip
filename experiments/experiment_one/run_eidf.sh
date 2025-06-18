#!/bin/bash

cd /PoolAgent/experiments/experiment_one/

# Now you can use $HUGGINGFACE_TOKEN
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Error: HUGGINGFACE_TOKEN is not set"
    exit 1
fi

# If environment variable SIZE equals SMALL
if [ "${SIZE}" = "SMALL" ]; then
    N_GPUS=1
    GPU_SIZE=40
    N_THREADS=1
# If environment variable SIZE equals MEDIUM
elif [ "${SIZE}" = "MEDIUM" ]; then
    N_GPUS=1
    GPU_SIZE=40
    N_THREADS=1
# If environment variable SIZE equals LARGE
elif [ "${SIZE}" = "LARGE" ]; then
    N_GPUS=6
    GPU_SIZE=40
    N_THREADS=1
else
    echo "Error: SIZE is not set"
    exit 1
fi

mkdir results/
mkdir logs/

python run_seq_exp.py --gpu_size $GPU_SIZE --n_gpus $N_GPUS --eidf local 

