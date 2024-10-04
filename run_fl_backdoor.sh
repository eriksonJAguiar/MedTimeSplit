#!/bin/bash

for percent in 0.7 0.5 0.3 0.2 0.1; do
    echo "Running simulation for FedNova with split percentage: $percent"
    python run_simulation_backdoor.py --model_name "resnet152" --strategy "FedNova" --percentage $percent
    #done
done