#!/bin/bash

# Loop through each combination of dataset and split method
for split in "SplitMnist" "PermutedMnist" "MedTimeSplit"; do
    for strategy in "FedAvg" "FedProx" "FedNova" "FedScaffold"; do
        echo "Running simulation for Strategy: $strategy with split method: $split"
        python run_simulation_cl_fl.py --model_name "resnet152" --strategy $strategy --split $split
    done
done