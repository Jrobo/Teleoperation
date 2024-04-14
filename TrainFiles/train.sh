  GNU nano 6.2                        train2.sh                                 
# Command 9
echo "Training with sigma=0.0001"
python3 ema_train_deep_ppca.py --sigma 0.1 --lr 0.0001 --num_epochs 30000

# Command 10
echo "Training with sigma=0.0006"
python3 ema_train_deep_ppca.py --sigma 0.4 --lr 0.0001 --num_epochs 30000

# Command 11
echo "Training with sigma=0.004"
python3 ema_train_deep_ppca.py --sigma 0.6 --lr 0.0001 --num_epochs 30000

# Command 12
echo "Training with sigma=0.006"
python3 ema_train_deep_ppca.py --sigma 0.006 --lr 0.0001 --num_epochs 30000

# Command 13
echo "Training with sigma=0.03"
python3 ema_train_deep_ppca.py --sigma 0.03 --lr 0.0001 --num_epochs 30000

# Command 14
echo "Training with sigma=0.2"
python3 ema_train_deep_ppca.py --sigma 0.2 --lr 0.0001 --num_epochs 30000

# Command 15
echo "Training with sigma=0.3"
python3 ema_train_deep_ppca.py --sigma 0.3 --lr 0.0001 --num_epochs 30000

# Command 16
echo "Training with sigma=2"
python3 ema_train_deep_ppca.py --sigma 2 --lr 0.4 --num_epochs 30000





