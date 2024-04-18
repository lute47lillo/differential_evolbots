"""

    Author: Lute Lillo Portero
    
    Definition
    -----------
        Util file containing helper functions necessary to read/write and
        calculate diverse simulation-related values.

"""
import os
import math
import random

# -----------------------------------------------------------------

"""
    GLOBAL VARIABLES
"""

# -----------------------------------------------------------------
x_offset = 0.1 
ground_height = 0.1
n_sin_waves = 10

# -----------------------------------------------------------------

"""
    Files helper functions.
"""

# -----------------------------------------------------------------

def create_video(experiment_name):
    os.system("rm simulation.mp4")
    os.system(f" ffmpeg -i images/robot_0/image_%d.png recordings/simulation_{experiment_name}.mp4")
    
# -----------------------------------------------------------------

# Helper function to set indices of files back in range of 0...n_robot_population
def re_order_files(directory, attr):
    
    # Get the list of files in the directory
    files = os.listdir(directory)
    
    # Sort the files by their index in ascending order
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # Loop through the files and rename them
    for idx, filename in enumerate(files):
        old_path = os.path.join(directory, filename)
        
        if attr == 'weights':
            new_filename = f"{attr}_{idx}.npz"
        else:
            new_filename = f"{attr}_{idx}.txt"
            
        new_path = os.path.join(directory, new_filename)
        
        # Rename the file only if its name is different from the new name
        if filename != new_filename:
            os.rename(old_path, new_path)
            print(f"Renamed '{directory}' '{filename}' to '{new_filename}'")
            
def rename_dir(directory):
    
    # Get the list of files in the directory
    dirs = os.listdir(directory)
    
    # Sort the files by their index in ascending order
    dirs.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # Loop through the files and rename them
    for idx, dir_name in enumerate(dirs):
        old_path = directory + "/" + dir_name
        new_dir_ename = f"robot_{idx}"
        new_path = directory + "/" + new_dir_ename
        
        # Rename the file only if its name is different from the new name
        if dir_name != new_dir_ename:
            os.rename(old_path, new_path)
            print(f"Renamed dir '{dir_name}' to '{new_dir_ename}'")

# -----------------------------------------------------------------

def remove_files_before_simulation():
    os.system("rm population/*.txt")
    os.system("rm fitness/*.txt")
    os.system("rm trackers_prob/*.txt")
    os.system("rm trackers_loss/*.txt")
    os.system("rm controller/*.npz")
    os.system(f"rm stats/loss.txt")
    os.system("rm -rf images/*")

# -----------------------------------------------------------------

def update_files():
    re_order_files("population", "robot")
    re_order_files("fitness", "loss")
    re_order_files("trackers_prob", "prob")
    re_order_files("trackers_loss", "loss")
    re_order_files("controller", "weights")
    rename_dir("images")
    
# -----------------------------------------------------------------

"""
    VALUES: Loss & Probabilities for mutation actions helper functions
"""

# -----------------------------------------------------------------

def track_values(robot_idx):
    """
        Definition
        -----------
            Tracks loss values of all robots to later be used to plot.
            TODO: Include mutation action probabilities.
        
        Parameters
        -----------
            - robot_idx (int): Number that represents the current robot on which the calculation is being made.
            
        Returns
        -----------
            None
    """
    
    # Read losses over time independently  
    with open(f"trackers_loss/loss_{robot_idx}.txt", 'r') as file:
        lines = file.readlines()
        losses = [float(line.strip()) for line in lines]
        file.close()
        
    # Save the losses for the robot and its intial index.
    with open(f"stats/loss.txt", 'a+') as file:
        save_line = str(losses) + "\n"
        file.writelines(save_line)
        file.close()

def check_last_and_prev_loss(robot_idx):
    """
        Definition
        -----------
            Util file containing helper functions necessary to read/write and
            calculate diverse simulation-related values.
            
        Parameters
        -----------
            - robot_idx (int): Number that represents the current robot on which the calculation is being made.

        Returns
        -----------
            - prev_loss (float): Loss value of the previous simulation step for the given robot.
            - last_loss (float): Loss value of the current simulation step for the given robot.
    
    """
    
    # Read losses over time independently  
    with open(f"trackers_loss/loss_{robot_idx}.txt", 'r') as file:
        lines = file.readlines()
        
        # Get first loss
        prev_line = lines[-2]
        prev_loss = float(prev_line.strip())
        
        # Get last loss
        last_line = lines[-1]
        last_loss = float(last_line.strip())
        file.close()
        
    return prev_loss, last_loss

def update_probabilities(n_robot_population, simulation_step):
    """
        Definition
        -----------
            Updates set of mutation action probabilities based on current and previous loss values.
        
        Parameters
        -----------
            - n_robot_population (int): Number of remaining robots in the simulated population.
            - simulation_step (int): Current step at which the simulation is at.
            
        Returns
        -----------
            - add_prob (float): Probability of selecting the 'Add an object' mutation action. 
            - remove_prob (float): Probability of selecting the 'Remove an object' mutation action. 
            - nothing_prob (float): Probability of selecting the 'Do nothing' mutation action. 
    """
    
    for robot_idx in range(n_robot_population):
        
        # Read from file existing objects
        if simulation_step > 0:
            previous_loss, last_loss = check_last_and_prev_loss(robot_idx)
    
            # Read from file existing objects
            with open(f"trackers_prob/prob_{robot_idx}.txt", 'r') as file:
                all_lines = file.readlines()
                last_line = all_lines[-1]
            
                # Get probabilities for each action
                probabilities = last_line.split()
                probabilities = [float(token) for token in probabilities]
                
                # TODO: Could include a delta that helps updating
                # TODO: Never removing objects prob -> Fix it
                # Update probabilities
                if last_loss > previous_loss: # Worse loss than prev. 
                    probabilities[0] = min(1.0, probabilities[0] + 0.30)
                    probabilities[1] = min(1.0, probabilities[1] + 0.15)
                    probabilities[2] = max(0.0, probabilities[2] - 0.45)
                                    
                elif last_loss > (previous_loss + 0.5): 
                    probabilities[0] = min(1.0, probabilities[0] + 0.40)
                    probabilities[1] = min(1.0, probabilities[1] + 0.35)
                    probabilities[2] = max(0.0, probabilities[2] - 0.75)
                                        
                elif last_loss < (previous_loss - 0.5):  # New loss is way smaller than previous -> Increase doing nothing
                    probabilities[0] = max(0.0, probabilities[0] - 0.40)
                    probabilities[1] = max(0.0, probabilities[1] - 0.35)
                    probabilities[2] = min(1.0, probabilities[2] + 0.75)
                    
                else: # Increase chance of adding or doing nothing.
                    probabilities[0] = max(0.0, probabilities[0] - 0.3)
                    probabilities[1] = max(0.0, probabilities[1] - 0.3)
                    probabilities[2] = min(1.0, probabilities[2] + 0.6)

                # Normalize the values
                total = probabilities[0] + probabilities[1]+ probabilities[2]
                add_prob = probabilities[0] / total
                remove_prob = probabilities[1] / total
                nothing_prob = probabilities[2] / total
                
        else:
            add_prob = 0.25
            remove_prob = 0.25
            nothing_prob = 0.50
            
        # Save to file
        with open(f"trackers_prob/prob_{robot_idx}.txt", 'a+') as file:
            save_actions = str(add_prob) + " " + str(remove_prob) + " " + str(nothing_prob) + "\n"
            file.write(save_actions)
            file.close()
            
    return add_prob, remove_prob, nothing_prob

def get_probabilities(robot_idx):
    """
        Definition
        -----------
            Get current set of mutation action probabilities.
        
        Parameters
        -----------
            - robot_idx (int): Number that represents the current robot on which the calculation is being made.

        Returns
        -----------
            - add_prob (float): Probability of selecting the 'Add an object' mutation action. 
            - remove_prob (float): Probability of selecting the 'Remove an object' mutation action. 
            - nothing_prob (float): Probability of selecting the 'Do nothing' mutation action. 
    """
    
    # Read from file existing objects
    with open(f"trackers_prob/prob_{robot_idx}.txt", 'r') as file:
        all_lines = file.readlines()
        last_line = all_lines[-1]
    
        # Get probabilities for each action
        probabilities = last_line.split()
        probabilities = [float(token) for token in probabilities]
        
        add_prob = probabilities[0]
        remove_prob = probabilities[1]
        nothing_prob = probabilities[2]
        
    return add_prob, remove_prob, nothing_prob
# -----------------------------------------------------------------

"""
    OBJECT and SPRING GENERATION FUNCTIONS
"""

# -----------------------------------------------------------------

def generate_obj_positions(n_objects):
    """
        Definition
        -----------
            Generates an object based on x, y coordinates.
            
        Parameters
        -----------
            - n_objects (int): number of objects to be generated
            
        Returns
        -----------
            - new_obj_pos (list): List of newly created objects for a given robot.

    """
    
    new_obj_pos = []
    for _ in range(n_objects):
        
        # Generate random x_pos and y_pos
        obj_x_pos = random.uniform(0, 0.25)
        obj_y_pos = random.uniform(0, 0.3)
        
        # Check there is no object in same x and y.
        for created_obj in new_obj_pos:
            x, y = created_obj
            
            # Add an arbitrary offset to undraw
            if x == obj_x_pos and y == obj_y_pos:
                obj_x_pos += 0.05
                obj_y_pos += 0.05
        
        # Add object
        new_obj_pos.append([x_offset + obj_x_pos, ground_height + obj_y_pos])
        
    return new_obj_pos

# -----------------------------------------------------------------

def create_spring(springs_robot, i, j, is_motor, startingObjectPositions):
    """
        Definition
        -----------
            Create a spring between objects at index i-th and j-th in the list startingObjectPositions. 
            Can be either motorized or not. Appends to list of springs.
            
        Parameters
        -----------
            - springs_robot (list): list of information of the generated robot.
            - i (int): object at index i-th in startingObjectPositions
            - j (int): object at index j-th in startingObjectPositions
            - is_motor (int): if the spring is motorized or not.
            - startingObjectPositions (list): List of objects of the specific robot.
            
        Returns
        -----------
            None
    """
    
    object_a = startingObjectPositions[i]
    object_b = startingObjectPositions[j]

    # Get x and y coordinates of objects to calculate distance
    x_distanceAB = object_a[0] - object_b[0]
    y_distanceAB = object_a[1] - object_b[1]

    # Pythagorean Distance.
    # Springs need a "at rest"-length that is the length that "likes" to stay at.
    distance_A_to_B = math.sqrt(x_distanceAB**2 + y_distanceAB**2)
    resting_length = distance_A_to_B
    
    springs_robot.append([i, j, resting_length, is_motor])

# -----------------------------------------------------------------

def n_sensors(n_objects):
    return n_sin_waves + 4 * n_objects + 2

# -----------------------------------------------------------------

