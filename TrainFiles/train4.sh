#!/bin/bash

# Command 6
echo "Training with sigma= 0.3"
python3 ema_train_deep_ppca.py --sigma 0.3 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02

# Command 7
echo "Training with sigma= 0.8"
python3 ema_train_deep_ppca.py --sigma 0.8 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02

# Command 8
echo "Training with sigma= 2"
python3 ema_train_deep_ppca.py --sigma 2 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02

# Command 9
echo "Training with sigma= 3"
python3 ema_train_deep_ppca.py --sigma 3 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02

# Command 10
echo "Training with sigma= 4"
python3 ema_train_deep_ppca.py --sigma 4 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02

# Command 11
echo "Training with sigma= 5"
python3 ema_train_deep_ppca.py --sigma 5 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02
