#!/bin/bash

# Correct the path to train.py and the arguments
python3 ../train.py \
    --model_name "DecisionTreeClassifier" \
    --random_state 101 \
    --n_trials 100 \
    --scale_data False \
    --top_n 8 \
    --feature_importance True
