import numpy as np
import matplotlib.pyplot as plt

def exponential_moving_average(data, alpha):
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return ema

def plot_npy_with_ema(filename, alpha):
    try:
        data = np.load(filename)
        print("Data shape:", data.shape)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return
    
    # Separate the data into two curves
    curve1 = data[0]
    curve2 = data[1]
    
    # Calculate EMA for each curve
    ema_curve1 = exponential_moving_average(curve1, alpha)
    ema_curve2 = exponential_moving_average(curve2, alpha)
    
    # Plot only the EMA smoothed data in symlog scale
    plt.semilogy(ema_curve1, label='Curve 1 (EMA Smoothed)')
    plt.semilogy(ema_curve2, label='Curve 2 (EMA Smoothed)')
    plt.title('Exponential Moving Average (EMA) Smoothed Data (symlog scale)')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def main():
    # Load data from .npy file
    filename = 'results/loss_sigma_0.01_epochs_5000_lr_0.0001.npy'  # Replace 'data.npy' with your file name
    
    # Define smoothing factor (alpha) for EMA
    alpha = 0.2  # You can adjust this value based on your requirements
    
    plot_npy_with_ema(filename, alpha)

if __name__ == "__main__":
    main()
