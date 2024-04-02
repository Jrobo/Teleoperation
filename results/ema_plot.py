import numpy as np
import matplotlib.pyplot as plt

def exponential_moving_average(data, alpha):
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return ema

def plot_ema(original_data, ema_data):
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(original_data, label='Original Data', color='blue')
    plt.title('Original Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(ema_data, label='EMA Smoothed Data', color='red')
    plt.title('Exponential Moving Average (EMA) Filter')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    # Load data from .npy file
    filename = 'results/loss_sigma_0.01_epochs_3000_lr_0.0001.npy'
    try:
        data = np.load(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    # Define smoothing factor (alpha)
    alpha = 0.2  # You can adjust this value based on your requirements

    # Apply EMA smoothing
    ema_smoothed = exponential_moving_average(data, alpha)

    # Plot original data and smoothed data
    plot_ema(data, ema_smoothed)

if __name__ == "__main__":
    main()
