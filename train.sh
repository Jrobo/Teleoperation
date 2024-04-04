#!/bin/bash

# Command 1

echo "Training with sigma=0.01"
python3 ema_train_deep_ppca.py --sigma 0.01 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.01

# Command 2
echo "Training with sigma=0.05"
python3 ema_train_deep_ppca.py --sigma 0.05 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.01

# Command 3
echo "Training with sigma=0.1"
python3 ema_train_deep_ppca.py --sigma 0.1 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.01


# Command 4
echo "Training with sigma=0.5"
python3 ema_train_deep_ppca.py --sigma 0.5 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.01

# Command 5
echo "Training with sigma=1"
python3 ema_train_deep_ppca.py --sigma 1 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.01

# Command 6
echo "Training with sigma=1.5"
python3 ema_train_deep_ppca.py --sigma 1.5 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.01

# Command 7
echo "Training with sigma=2"
python3 ema_train_deep_ppca.py --sigma 2 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.01

# Command 8
echo "Training with sigma=2.5"
python3 ema_train_deep_ppca.py --sigma 2.5 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.01
