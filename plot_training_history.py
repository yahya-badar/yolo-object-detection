import numpy as np
import matplotlib.pyplot as plt

def plot_from_log(filename):
    # Skip the header row
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    
    # Column indices: 0=epoch, 2=total, 3=box, 4=cls, 5=dfl
    plt.figure(figsize=(10, 6))
    plt.plot(data[:, 2], label='Total Loss', alpha=0.3) 
    plt.plot(data[:, 3], label='Box Loss')
    plt.plot(data[:, 4], label='Class Loss')
    plt.plot(data[:, 5], label='DFL Loss')
    
    # Set the y-axis to logarithmic scale
    plt.yscale('log')
    
    plt.title("Training Loss History (Log Scale)")
    plt.xlabel("Log Steps (Every 10 batches)")
    plt.ylabel("Loss (Log Scale)")
    plt.legend()
    
    # Using 'both' for the grid helps visualize log minor ticks
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()
    
log_file = 'losshistory.txt'
plot_from_log(log_file)
