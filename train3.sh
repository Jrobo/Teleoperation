#!/bin/bash


# Command 1
echo "Training with sigma= 0.0001"
python3 ema_train_deep_ppca.py --sigma 0.0001 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02

# Command 2
echo "Training with sigma= 0.0006"
python3 ema_train_deep_ppca.py --sigma 0.0006 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02

# Command 3
echo "Training with sigma= 0.004"
python3 ema_train_deep_ppca.py --sigma 0.004 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02

# Command 4
echo "Training with sigma= 0.006"
python3 ema_train_deep_ppca.py --sigma 0.006 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02

# Command 5
echo "Training with sigma= 0.03"
python3 ema_train_deep_ppca.py --sigma 0.03 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02



