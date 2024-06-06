  GNU nano 6.2                                                                                     train7.sh                                                                                               
#!/bin/bash
#Doing the training with small values using both seed values
#export PYTHONPATH=$PYTHONPATH:/home/kazi.jamil/Teleoperation

# Command 23
echo "Training with sigma= 0.00006"
python3 ema_train_deep_ppca.py --sigma 0.00006 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 24
echo "Training with sigma= 0.00001"
python3 ema_train_deep_ppca.py --sigma 0.00001 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 25
echo "Training with sigma= 0.000006"
python3 ema_train_deep_ppca.py --sigma 0.000006 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 26
echo "Training with sigma= 0.00006"
python3 ema_train_deep_ppca.py --sigma 0.00006 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02

# Command 27
echo "Training with sigma= 0.00001"
python3 ema_train_deep_ppca.py --sigma 0.00001 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02

# Command 28
echo "Training with sigma= 0.000006"
python3 ema_train_deep_ppca.py --sigma 0.000006 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.02