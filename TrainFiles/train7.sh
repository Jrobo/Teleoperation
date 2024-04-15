#!/bin/bash

#Doing the same values with different seed (42) part 2

# Command 10
echo "Training with sigma= 0.4"
python3 ema_train_deep_ppca.py --sigma 0.4 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 11
echo "Training with sigma= 0.3"
python3 ema_train_deep_ppca.py --sigma 0.3 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 12
echo "Training with sigma= 0.1"
python3 ema_train_deep_ppca.py --sigma 0.1 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 13
echo "Training with sigma= 0.06"
python3 ema_train_deep_ppca.py --sigma 0.06 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 14
echo "Training with sigma= 0.03"
python3 ema_train_deep_ppca.py --sigma 0.03 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 15
echo "Training with sigma= 0.02"
python3 ema_train_deep_ppca.py --sigma 0.02 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 16
echo "Training with sigma= 0.01"
python3 ema_train_deep_ppca.py --sigma 0.01 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 17
echo "Training with sigma= 0.006"
python3 ema_train_deep_ppca.py --sigma 0.006 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 18
echo "Training with sigma= 0.004"
python3 ema_train_deep_ppca.py --sigma 0.004 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 19
echo "Training with sigma= 0.003"
python3 ema_train_deep_ppca.py --sigma 0.003 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02
