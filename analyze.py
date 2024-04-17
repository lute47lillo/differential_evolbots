"""

    Author: Lute Lillo Portero
    
    Definition
    -----------
    
        Analyze the statistics collected by the simulation and plot figures representing a study of the morphology evolution.

"""
import matplotlib.pyplot as plt

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
    xmax = max(len(lst) for lst in lists)
    
    return lists, xmax

def plot_loss(lists, xmax):
    
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
    
def plot_loss_2(lists, xmax):
    # Create a 5x4 grid of subplots
    fig, axs = plt.subplots(5, 4, figsize=(20, 16))

    x_ticks = []
    for i in range(-1, 20):
        x_ticks.append(i)
        
    # Plot each list in a subplot
    for idx, lst in enumerate(lists):
        row = idx // 4
        col = idx % 4
        x = range(len(lst))
        
        # Plot dots at specific values
        axs[row, col].plot(x, lst, color='red')
        axs[row, col].scatter(x, lst, color='red', zorder=5)

        # Set y-ticks to specific values for each subplot
        axs[row, col].set_yticks([-0.3, -0.28, -0.26, -0.24, -0.22, -0.20, -0.18, -0.16, -0.14, -0.12])
        axs[row, col].set_xticks(x_ticks)
        axs[row, col].set_xlabel('Simulation Step', fontsize=12)
        axs[row, col].set_ylabel('Loss', fontsize=12)
        
        if idx < len(lists) - 1:
            axs[row, col].set_title(f'Robot eliminated at step {idx}', fontsize=12)
        else:
            axs[row, col].set_title(f'Fittest Robot', fontsize=12)
        

    # Add labels and show plot
    fig.suptitle('All loss trajectories', fontsize=24)
    plt.tight_layout()
    plt.savefig(f"plots/all_loss.png")
    
lists, xmax = read_statistics_file()

# plot_loss(lists, xmax)
plot_loss_2(lists, xmax)