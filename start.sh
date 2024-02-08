#!/bin/bash

echo "staring new env..."
python3 -m venv fl-env
source fl-env/bin/activate

echo "create folder to inser your datasets"
mkdir datasets
