"""

    Author: Lute Lillo Portero
    
    Definition
    -----------
    
        Analyze the statistics collected by the simulation and plot figures representing a study of the morphology evolution.

"""
import matplotlib.pyplot as plt
import math

def read_statistics_file():
    with open(f"stats/loss.txt", 'r') as file:
        lines = file.readlines()
        
    # Process the lines to extract the lists of numbers
    lists = []
    for line in lines:
        # Remove brackets and split by comma
        values = line.strip()[1:-1].split(',')
        # Convert values to float
        # values = [float(value.strip()) for value in values]
        values = [float(value.strip()) if float(value.strip()) != 100000 else 0 for value in values]

        lists.append(values)

    # Find the length of the longest list
    xmax = max(max(lst) for lst in lists)
    xmin = min(min(lst) for lst in lists)
    
    return lists, xmax, xmin

def plot_loss(lists, xmax, xmin):
    
    # Plot each list as a line
    for lst in lists:
        plt.scatter(range(len(lst)), lst)

    # Set x-axis limit to the length of the longest list
    plt.xlim(-1, xmax)
    plt.ylim(-0.3, -0.10)

    plt.yticks([-0.3, -0.28, -0.26, -0.24, -0.22, -0.20, -0.18, -0.16, -0.14, -0.12])
    plt.xticks([0, 5, 10, 15, 20])
    # Add labels and show plot
    plt.xlabel('Simulation Step')
    plt.ylabel('Loss')
    plt.title('Values from File')
    plt.savefig(f"plots/loss.png")
    # plt.show()
    
def plot_loss_2(lists, xmin, xmax):
    # Create a 5x4 grid of subplots
    # TODO: Create grid of plots based on how many are in the list
    # fig, axs = plt.subplots(5, 6, figsize=(40, 20))
    num_values = len(lists) + 1
    grid_size = math.ceil(math.sqrt(num_values)) + 1
    print(grid_size)
    fig, axs = plt.subplots(grid_size-1, grid_size, figsize=(16, 12))


    x_ticks = []
    for i in range(0, len(lists)):
        x_ticks.append(i) 
   
    y_ticks = []
    current = xmin
    while current <= xmax:
        diff = xmax - xmin
        ticks_step = round(float(diff / 10), 2)
        y_ticks.append(round(current, 2))  # Round to 2 decimal places
        current += ticks_step
        
    # Plot each list in a subplot
    for idx, lst in enumerate(lists):
        row = idx // grid_size
        col = idx % grid_size
        x = range(len(lst))
        print(row, col)
        
        # Plot dots at specific values
        axs[row, col].plot(x, lst, color='red')
        axs[row, col].scatter(x, lst, color='red', zorder=5)

        # Set y-ticks to specific values for each subplot
        axs[row, col].set_yticks(y_ticks)
        axs[row, col].set_xticks(x_ticks)
        axs[row, col].set_xlabel('Simulation Step', fontsize=12)
        axs[row, col].set_ylabel('Loss', fontsize=12)
        
        if idx < len(lists) - 1:
            axs[row, col].set_title(f'Robot eliminated at step {idx}', fontsize=12)
        else:
            axs[row, col].set_title(f'Fittest Robot', fontsize=12)
       
    # Hide non-used arrays 
    for i in range(num_values-1, grid_size**2 -1):
        row = i // grid_size
        col = i % grid_size
        if row < axs.shape[0] and col < axs.shape[1]:
            axs[row, col].axis('off')

    
    # Add labels and show plot
    fig.suptitle('All loss trajectories', fontsize=24)
    plt.tight_layout()
    plt.savefig(f"plots/exp5_loss_10r_10o.png")
    
lists, xmax, xmin= read_statistics_file()


# plot_loss(lists, xmax)
plot_loss_2(lists, xmin, xmax)