#!/bin/bash

TASK_NAME="pick_up_the_tomato_sauce_and_place_it_in_the_basket"
STEERING_ARGS=(
    "45366:0.0,8447:100.0,2986:0.0,26272:0.0,26272:0.0"
    "45356:0.0,8447:0.0,2986:100.0,26272:0.0,26272:0.0"
    "45356:0.0,8447:0.0,2986:0.0,26272:100.0,26272:0.0"
    "45356:0.0,8447:0.0,2986:0.0,26272:0.0,26272:100.0"
)

for STEERING in "${STEERING_ARGS[@]}"; do
    echo "Running evaluation with steering: $STEERING"
    python steer_eval.py \
        --task_name "$TASK_NAME" \
        --steering "$STEERING" 
done