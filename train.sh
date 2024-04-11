#!/bin/bash

# Command 1

echo "Training with sigma=0.04"
python3 ema_train_deep_ppca.py --sigma 0.04 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.01

# Command 2
echo "Training with sigma=0.06"
python3 ema_train_deep_ppca.py --sigma 0.06 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.01


