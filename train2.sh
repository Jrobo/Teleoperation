# Command  3
echo "Training with sigma=0.4"
python3 ema_train_deep_ppca.py --sigma 0.4 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.01

# Command 4
echo "Training with sigma=0.6"
python3 ema_train_deep_ppca.py --sigma 0.6 --lr 0.0001 --num_epochs 30000 --seed 123 --ema_alpha 0.01

