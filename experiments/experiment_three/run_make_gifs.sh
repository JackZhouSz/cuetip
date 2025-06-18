#!/bin/bash

# Loop from 0 to 24
for i in {0..24}; do
    echo "Processing shot $i"
    python make_gifs.py $i
done
