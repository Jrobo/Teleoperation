#!/bin/bash
#Doing the same values with different seed (42) part 1
#export PYTHONPATH=$PYTHONPATH:/home/kazi.jamil/Teleoperation
# Command 1
echo "Training with sigma= 1"
python3 ema_train_deep_ppca.py --sigma 1 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 2
echo "Training with sigma= 2"
python3 ema_train_deep_ppca.py --sigma 2 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 3
echo "Training with sigma= 3"
python3 ema_train_deep_ppca.py --sigma 3 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 4
echo "Training with sigma= 4"
python3 ema_train_deep_ppca.py --sigma 4 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 5
echo "Training with sigma= 5"
python3 ema_train_deep_ppca.py --sigma 5 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 6
echo "Training with sigma= 0.8"
python3 ema_train_deep_ppca.py --sigma 0.8 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 7
echo "Training with sigma= 0.6"
python3 ema_train_deep_ppca.py --sigma 0.6 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 8
echo "Training with sigma= 0.0006"
python3 ema_train_deep_ppca.py --sigma 0.0006 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 9
echo "Training with sigma= 0.001"
python3 ema_train_deep_ppca.py --sigma 0.001 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

