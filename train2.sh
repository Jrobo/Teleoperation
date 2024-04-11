# Command 5

echo "Training with sigma=0.1"
python3 ema_train_deep_ppca.py --sigma 0.1 --lr 0.0001 --num_epochs 30000 --s>

# Command 6
echo "Training with sigma=0.4"
python3 ema_train_deep_ppca.py --sigma 0.4 --lr 0.0001 --num_epochs 30000 --s>

# Command  7
echo "Training with sigma=0.6"
python3 ema_train_deep_ppca.py --sigma 0.6 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.01

# Command 8
echo "Training with sigma=1"
python3 ema_train_deep_ppca.py --sigma 1 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.01



