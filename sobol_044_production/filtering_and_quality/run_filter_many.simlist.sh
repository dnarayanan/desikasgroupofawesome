#!/bin/bash
ml conda
mamba activate /blue/narayanan/desika.narayanan/conda/envs/py38

# Define the number of parallel jobs
# This can be the number of cores on your node
NUM_JOBS=32

# Read sim_list.txt and pipe it to xargs to run in parallel
cat sim_list.txt | xargs -P $NUM_JOBS -I {} python filter_illustris_many.py "{}"

echo "All jobs launched."
