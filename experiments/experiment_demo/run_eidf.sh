#!/bin/bash

cd /mnt/ceph_rbd/PoolAgent/src/

# Now you can use $HUGGINGFACE_TOKEN
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Error: HUGGINGFACE_TOKEN is not set"
    exit 1
fi


# If environment variable SIZE equals SMALL
if [ "${SIZE}" = "SMALL" ]; then
    N_GPUS=2
    GPU_SIZE=40
# If environment variable SIZE equals MEDIUM
elif [ "${SIZE}" = "MEDIUM" ]; then
    N_GPUS=3
    GPU_SIZE=40
# If environment variable SIZE equals LARGE
elif [ "${SIZE}" = "LARGE" ]; then
    N_GPUS=3
    GPU_SIZE=80
else
    echo "Error: SIZE is not set"
    exit 1
fi

python3 -m experiment_demo.gen_skill_estimate --gpu_size $GPU_SIZE --n_gpus $N_GPUS --n_threads 3 local 