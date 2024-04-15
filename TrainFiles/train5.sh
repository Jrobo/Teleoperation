#!/bin/bash

# Command 12
echo "Training with sigma= 0.00001"
python3 ema_train_deep_ppca.py --sigma 0.00001 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02

# Command 13
echo "Training with sigma= 0.02"
python3 ema_train_deep_ppca.py --sigma 0.02 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02

# Command 14
echo "Training with sigma= 0.003"
python3 ema_train_deep_ppca.py --sigma 0.003 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02

# # Command 15
# echo "Training with sigma= 3"
# python3 ema_train_deep_ppca.py --sigma 3 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02

# # Command 16
# echo "Training with sigma= 4"
# python3 ema_train_deep_ppca.py --sigma 4 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02

# # Command 17
# echo "Training with sigma= 5"
# python3 ema_train_deep_ppca.py --sigma 5 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02

