"""

    Author: Lute Lillo Portero
    
    Definition
    -----------
    
        Analyze the statistics collected by the simulation and plot figures representing a study of the morphology evolution.

"""
import matplotlib.pyplot as plt
import math
import utils
import numpy as np

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
    
    fittest_values = lists[-1]
    
    with open(f"stats/baseline_loss_random.txt", 'r') as file:
        line = file.readline().rstrip('\n')
        random_values = [float(num) for num in line.strip('[]').split(',')]
        file.close()
        
    with open(f"stats/baseline_loss_test.txt", 'r') as file:
        line = file.readline().rstrip('\n')
        test_values = [float(num) for num in line.strip('[]').split(',')]
        file.close()
    
    return lists, xmax, xmin, fittest_values, random_values, test_values
    
def plot_loss_2(lists, xmin, xmax):
    
    # Create grid of plots based on how many are in the list
    num_values = len(lists) + 1
    grid_size = math.ceil(math.sqrt(num_values)) + 1
    # print(grid_size)
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
    # print(lists)
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
    plt.savefig(f"plots/loss/Loss_{name_experiment}_{n_pop}r_{n_opt}o.png")
    
    
def plot_random_vs_fit_vs_test(fit, random, test, xmax, xmin):
    
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
        
    # Plot each list as a line
    plt.plot(range(len(fit)), fit, label="Co-Evolution")
    plt.plot(range(len(random)), random, label="Random")
    plt.plot(range(len(test)), test, label="Test Co-Ev Controller")
    # plt.yticks(y_ticks)
    plt.xticks(x_ticks)

    # Add labels and show plot
    plt.xlabel('Simulation Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Comparison fitness')
    plt.savefig(f"plots/comparisons/Comp_EXP_{name_experiment}_{n_pop}r_{n_opt}o.png")
    
def plot_probs():
    
    # Read the file and extract the data
    data = []
    with open('stats/probs.txt', 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and square brackets
            line = line.strip()[2:-2]
            # Split the line into sublists
            sublists = line.split("], [")
            # Split each sublist into values and strip single quotes
            values = [sublist.strip(" '").split("', '") for sublist in sublists]
            # Convert values to floats
            values = [[float(val) for val in sublist] for sublist in values]
            data.append(values)

    # Plot each line as its own subplot
    num_values = len(data) + 1
    grid_size = math.ceil(math.sqrt(num_values)) + 1
    fig, axs = plt.subplots(grid_size-1, grid_size, figsize=(16,12))

    mutation_action = ['add', 'remove', 'nothing']
    for idx, line_data in enumerate(data):
        row = idx // grid_size
        col = idx % grid_size
        # Transpose the data to group values by index
        line_data = np.array(line_data).T
        for j, index_data in enumerate(line_data):
            axs[row, col].scatter(range(len(index_data)), index_data*100, label=f'{mutation_action[j]}')
        axs[row, col].legend()
        axs[row, col].set_ylabel(f'Percentage (%) probability')
        axs[row, col].set_xlabel('Simulation Step')
        
     # Hide non-used arrays 
    for i in range(num_values-1, grid_size**2 -1):
        row = i // grid_size
        col = i % grid_size
        if row < axs.shape[0] and col < axs.shape[1]:
            axs[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(f"plots/mutations/Probs_{name_experiment}_{n_pop}r_{n_opt}o.png")


lists, xmax, xmin, fittest_values, random_values, test_values = read_statistics_file()
n_pop, n_opt, name_experiment, _ = utils.parse_args_baseline()

plot_random_vs_fit_vs_test(fittest_values, random_values, test_values, xmax, xmin)
plot_loss_2(lists, xmin, xmax)
plot_probs()