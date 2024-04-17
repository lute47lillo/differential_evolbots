"""

    Author: Lute Lillo Portero
    
    Definition
    -----------
        Util file containing helper functions necessary to read/write and
        calculate diverse simulation-related values.

"""

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
                # Update probabilities
                if last_loss > previous_loss:
                    probabilities[0] = min(1.0, probabilities[0] + 0.55)
                    probabilities[1] = max(0.0, probabilities[1] - 0.15)
                    probabilities[2] = max(0.0, probabilities[2] - 0.4)
                                    
                elif last_loss > (previous_loss + 0.3): 
                    probabilities[0] = min(1.0, probabilities[0] + 0.8)
                    probabilities[1] = max(0.0, probabilities[1] - 0.4)
                    probabilities[2] = max(0.0, probabilities[2] - 0.4)
                                        
                elif last_loss < (previous_loss - 0.4):  # New loss is way smaller than previous -> Increase doing nothing
                    probabilities[0] = max(0.0, probabilities[0] - 0.35)
                    probabilities[1] = max(0.0, probabilities[1] - 0.35)
                    probabilities[2] = min(1.0, probabilities[2] + 0.7)
                    
                else: # Increase chance of adding or doing nothing.
                    probabilities[0] = max(0.0, probabilities[0] - 0.2)
                    probabilities[1] = max(0.0, probabilities[1] - 0.3)
                    probabilities[2] = min(1.0, probabilities[2] + 0.5)

                # Normalize the values
                total = probabilities[0] + probabilities[1]+ probabilities[2]
                add_prob = probabilities[0] / total
                remove_prob = probabilities[1] / total
                nothing_prob = probabilities[2] / total
                
        else:
            add_prob = 0.34
            remove_prob = 0.33
            nothing_prob = 0.33
            
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

