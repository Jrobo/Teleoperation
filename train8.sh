  GNU nano 6.2                                                                                     train7.sh                                                                                               
#!/bin/bash

#Doing the same values with different seed (42) part 3
#export PYTHONPATH=$PYTHONPATH:/home/kazi.jamil/Teleoperation
# Command 20
echo "Training with sigma= 0.00006"
python3 ema_train_deep_ppca.py --sigma 0.00006 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 21
echo "Training with sigma= 0.3"
python3 ema_train_deep_ppca.py --sigma 0.00001 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

# Command 22
echo "Training with sigma= 0.3"
python3 ema_train_deep_ppca.py --sigma 0.000006 --lr 0.0001 --num_epochs 30000 --seed 42 --ema_alpha 0.02

