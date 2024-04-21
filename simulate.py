
import taichi as tai
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from robot import Robot
import shutil
import utils
import time

# -------------------------------------------------------------

"""
    GLOBAL VARIABLES
    TODO: Create a single file to have all global variables.
"""
max_steps = 200
ground_height = 0.1
stiffness = 1000 # Strength of the spring in the example
dt = 0.01 # Amount of time that elapses between time steps.
gravity = -9.89
learning_rate = 1
piston_force = 0.07 # Force applied to the actuations. Bigger, piston force, less acrobatic
x_offset = 0.1 # How far from left screen robot starts
damping = 0.65 # Is a constant that controls how much you slow the velocity of the object to which is applied. (1-damping) = X% reductions each time-step
n_hidden_neurons = 32
n_sin_waves = 10
n_robot_population = 15
n_optimization_steps = 10
initial_robot_population = n_robot_population

"""
    UTIL FUNCTIONS
"""

# -----------------------------------------------------------------

def simulate_robot(robot_index):
    """
        Definition
        -----------
            Creates an individual random robot within an initialized population.
            Morphology initial search space is max'd at 6 objects. Minimum of 2 objects.
            
        Parameters
        -----------
            - robot_index (int): Indicating the index of the robot in the population.
            
        Returns
        -----------
            - springs_robot (list): list of information of the generated robot.
            - startingObjectPositions (list): List of objects of the specific robot.
    """
    
    springs_robot = []
    startingObjectPositions = []
    
    # First object is always given.
    startingObjectPositions.append([x_offset, ground_height+0.02])
    startingObjectPositions.append([x_offset+0.2, ground_height+0.02])
    
    # TODO: How many more are created? Should I start at least with 3 minimum? Make 4 maximum?
    total_objects = random.randint(1, 4)
    
    # Generate the objects
    startingObjectPositions += utils.generate_obj_positions(total_objects)
        
    # Generate Springs. Randomly select if they are motorized or not.
    for i in range(len(startingObjectPositions)):
        for j in range(i+1, len(startingObjectPositions)):
            is_motor = random.choice([0, 1])
            utils.create_spring(springs_robot, i, j, is_motor, startingObjectPositions)
 
    # Write information of the robot morphology to text
    with open(f"population/robot_{robot_index}.txt", 'w') as file:
        
        # Write Object position
        pos_line = str(startingObjectPositions) + '\n'
        file.write(pos_line)
        
        # Write springs information
        for sublist in springs_robot:
            line = ' '.join(map(str, sublist)) + '\n'
            file.write(line)

    return springs_robot, startingObjectPositions

# -----------------------------------------------------------------

def create_population(n_robots_population):
    """
        Definition
        -----------
            Creates a population of individual robots.
            
        Parameters
        -----------
            - n_robots_population (int): Indicates the number of robots in the population.
            
        Returns
        -----------
            - springs_robot_population (List[list]): List of lists of information of the generated robot.
            - startingObjectPositions (List[list]): List of lists of objects of the specific robot.
    """
    springs_population = []
    startingObjectPositions_population = []
    
    # Simulate individual robot and gather springs
    for idx_robot in range(n_robots_population):
        springs, startingObjectPositions = simulate_robot(idx_robot)
        springs_population.append(springs)   
        startingObjectPositions_population.append(startingObjectPositions) 
    
    return springs_population, startingObjectPositions_population 

# -----------------------------------------------------------------

# Execute by Taichi and not python by using decorater
@tai.kernel
def Compute_loss():
    
    # Focus on position of the objects to determine loss fn. Arbitrary choice
    # Second component of zeroth object. Loss = Height of 0th objects at last time_step [1]
    
    # TODO: Try with negative final position
    loss[None] -= (1.2 * (r.positions[r.max_steps-1, 0][0]) - 0.1) \
                + (1.2 * (r.positions[r.max_steps-1, 1][0]) - 0.3) \
                + (0.7 * (r.goal[None][0] - r.positions[r.max_steps-1, 0][0])) \
                + (0.7 * (r.goal[None][0] - r.positions[r.max_steps-1, 1][0]))
    
# -------------------------------------------------------------

@tai.kernel
def calculate_center_robot(time_step: tai.i32, r: tai.template()): # type: ignore
    
    for _ in range(1): #Taichi sugar code
        c = tai.Vector([0.0, 0.0])
        
        for i in range(r.n_objects):
            c += r.positions[time_step, i] # Position of i-th object at time_step
            
        r.center[time_step] = c / r.n_objects

# -------------------------------------------------------------

# TODO: Needs to use initial r.positions and r.springs in order to draw it properly.
def Draw(r, frame_offset, robot_index):
        
    for time_step in range(0, r.max_steps):
        # Draw the robot using Taichi's built-iGUI. (x,y) size of window
        tai_gui = tai.GUI("Robot", (512, 512),
                          background_color=0xFFFFFF, show_gui=False)

        # Draw the floow
        tai_gui.line(begin=(0, ground_height), end=(1, ground_height),
                     color=0x0, radius=3)

        # Draw the object
        for object_idx in range(r.n_objects):
            
            # Get at time step for each object -> (x,y) coords
            x = r.positions[time_step, object_idx][0]
            y = r.positions[time_step, object_idx][1]
            tai_gui.circle((x,y), color=0x0, radius=7)
            
        for spring_idx in range(r.n_springs):
            object_a_index = r.spring_anchor_a[spring_idx]
            object_b_index = r.spring_anchor_b[spring_idx]
            
            # Get the positions of spring A at every time step
            position_a = r.positions[time_step, object_a_index]
            position_b = r.positions[time_step, object_b_index]
            
            has_motor = r.springs[spring_idx][3]
            
            if has_motor:
                tai_gui.line(begin=position_a, end=position_b, color=0x0, radius=3) # Thicker line
            else:
                tai_gui.line(begin=position_a, end=position_b, color=0x0, radius=1)

        if not os.path.exists(f"images/robot_{robot_index}"):
            os.makedirs(f"images/robot_{robot_index}")  
               
        tai_gui.show(f"images/robot_{robot_index}/image_{frame_offset+time_step}.png")

# -------------------------------------------------------------

def Initialize(r):
    """
        Definition
        -----------
            Initialize the robot. Call at the beginning of each run.
            
    """
    # Initialize the position for each object
    for object_idx in range(r.n_objects):
        # Temp, just propagate positions.
        r.positions[0, object_idx] = r.startingObjectPositions[object_idx]
        
        # Set initial velocites
        r.velocities[0, object_idx] = [0, -0.1]
        
    # spring_anchor_a, spring_anchor_b = spring_anchors
    for spring_idx in range(r.n_springs):
        s = r.springs[spring_idx] # Get spring
        r.spring_anchor_a[spring_idx]         = s[0] # the a object of that spring
        r.spring_anchor_b[spring_idx]         = s[1]
        r.spring_at_rest_length[spring_idx]   = s[2]
        r.spring_actuation[spring_idx]        = s[3]
        
    # Reset Positions (x,y) and velocities
    for i in range(1, r.max_steps):
        for j in range(r.n_objects):
            r.positions[i,j][0] = 0.0
            r.positions[i,j][1] = 0.0
            r.velocities[i,j][0] = 0.0
            r.velocities[i,j][1] = 0.0
    
    for i in range(1, r.max_steps):
        for j in range(r.n_springs):
            r.spring_restoring_forces[i,j][0] = 0.0   
            r.spring_restoring_forces[i,j][1] = 0.0 
            
    # Restore forces apply on the object by the springs
    for i in range(1, r.max_steps):
        for j in range(r.n_objects):
            r.spring_forces_on_objects[i,j][0] = 0.0   
            r.spring_forces_on_objects[i,j][1] = 0.0 
    
    # Reset values
    for i in range(1, r.max_steps):
        for j in range(r.n_hidden_neurons):
            r.hidden[i,j] = 0.0
            
    # Reset values of motor neurons
    for i in range(1, r.max_steps):
        for j in range(r.n_springs):
            r.actuation[i,j] = 0.0
            
    r.goal[None] = [0.9, 0.2]
        
def Initialize_Neural_Network(r):
    """
         Definition
        -----------
            Initialize the paramters of the Neural Network.
    """
    # Initialize sensor to hidden neurons
    for i in range(r.n_hidden_neurons):
        for j in range(utils.n_sensors(r.n_objects)):
            r.weightsSH[i,j] = np.random.randn() * 0.3 - 0.1
    
    # Init bias for hidden neurons
    for i in range(r.n_hidden_neurons):
        r.bias_hidden[i] = np.random.randn() * 2 - 1
        
    # Init weights
    for i in range(r.n_springs):
        for j in range(r.n_hidden_neurons):
            r.weightsHM[i,j] = np.random.randn() * 0.3 - 0.1
            
# -------------------------------------------------------------

def Simulate(r):

    for time_step in range(1, r.max_steps):
        
        step_one(time_step, r)

# -------------------------------------------------------------

# transform senation into action
@tai.kernel
def simulate_neural_network_SH(time_step: tai.i32, r: tai.template()): # type: ignore
    
    # Propagate values
    for i in range(r.n_hidden_neurons):
        activation = 0.0
        
        # for each of the CPPNS
        for j in tai.static(range(n_sin_waves)): 
            # increment act of i-th neuron by the sinuoisoid of time_step. j is a phase offset
            activation += r.weightsSH[i,j] * tai.sin(30 * time_step*dt + \
                                                    2* math.pi / n_sin_waves * j) 
            
        # Simulate the sensors inside the objects
        # First 2 sensors -> 'proprioceptive sensors'. Indicate position of that object wrt robots center of mass.
        for j in tai.static(range(r.n_objects)):
            offset = r.positions[time_step, j] - r.center[time_step]
            
            # Add to i-th neuron, the horizontal dist between j-th object and bot's center
            activation += 0.25 * r.weightsSH[i, j* 4 + n_sin_waves] * offset[0]
            # Add to i-th neuron, the vertical dist between j-th object and bot's center
            activation += 0.25 * r.weightsSH[i, j* 4 + 1 + n_sin_waves] * offset[1]
            
            activation += 0.25 * r.weightsSH[i, j* 4 + 2 + n_sin_waves] * r.positions[time_step, j][1]
            activation += 0.25 * r.weightsSH[i, j* 4 + 3 + n_sin_waves] * r.positions[time_step, j][1]
        
        # goal sensors -> how far the bot got?
        activation += 0.25 * r.weightsSH[i, r.n_objects * 4 + n_sin_waves] * (r.goal[None][0] - r.center[time_step][0])
        activation += 0.25 * r.weightsSH[i, r.n_objects * 4 + n_sin_waves + 1] * (r.goal[None][1] - r.center[time_step][1])
            
        # Apply non-linearity
        activation += r.bias_hidden[i]
        activation = tai.tanh(activation)
        
        # Store in a hidden neuron at every time_step
        r.hidden[time_step, i] = activation
# -------------------------------------------------------------

@tai.kernel
def simulate_neural_network_HM(time_step: tai.i32, r: tai.template()): # type: ignore
    
    # For every spring..
    for i in range(r.n_springs):
        
        activation = 0.0 # Init for each motor neuron
        
        # Visit each hidden neuron. And sum up influence of all hidden neurons for each motor.
        for j in tai.static(range(r.n_hidden_neurons)):
            
            activation += r.weightsHM[i, j] * r.hidden[time_step, j] # pre-synaptic hidden neuron value
            
        activation = tai.tanh(activation)
        r.actuation[time_step, i] = activation
            

# -------------------------------------------------------------

@tai.kernel
def simulate_springs(time_step: tai.i32, r: tai.template()): # type: ignore
    
    # Simulate the physics of each springs at initial step
    for spring_idx in range(r.n_springs):
        object_a_index = r.spring_anchor_a[spring_idx]
        object_b_index = r.spring_anchor_b[spring_idx]
        
        # Get most recent position.
        position_a = r.positions[time_step-1, object_a_index]
        position_b = r.positions[time_step-1, object_b_index]
        
        # Compute distance between objects -> Length of spring at rest
        distance_a_b = position_a - position_b
        curr_rest_length = distance_a_b.norm()
        
        spring_resting_length = r.spring_at_rest_length[spring_idx]
        
        # Applying the sinuisoidal function to have the piston of the motor (the cause of the movement be in that range)
        # spring_resting_length = spring_resting_length + 0.08 * spring_actuation[spring_idx] * tai.sin(0.9*time_step)
        
        # Newer version takes the motorized action form the NN. Keep value small
        spring_resting_length = spring_resting_length + piston_force * r.spring_actuation[spring_idx] * r.actuation[time_step, spring_idx]
        
        # Difference between current and supposed initial at that index
        spring_difference = curr_rest_length - spring_resting_length
        
        # Apply force proportionally to the difference between the at rest lengths. Normalized result by current distance
        # Turn the restoring force to a vector parallet to the vector connecting the two objects (by mult by the distance_a_b)
        # Big distances (denominator says) should NOT have big forces -> Swinging pendulum effect without stability
        # We would also need to add strength to the spring -> stiffness
        r.spring_restoring_forces[time_step, spring_idx] = (dt * spring_difference  * stiffness / curr_rest_length) * distance_a_b
        
        # Apply the force. - symbol means pulling force
        r.spring_forces_on_objects[time_step, object_a_index] +=  -3 * r.spring_restoring_forces[time_step, spring_idx]
        r.spring_forces_on_objects[time_step, object_b_index] +=  3 * r.spring_restoring_forces[time_step, spring_idx]
        
# -------------------------------------------------------------

@tai.kernel
def simulate_objects(time_step: tai.i32, r: tai.template()): # type: ignore
    
    for object_idx in range(r.n_objects):
        
        # Get old position and velocity
        old_pos = r.positions[time_step-1, object_idx]
        old_velocity = (damping * r.velocities[time_step-1, object_idx] +
                        dt * gravity  * tai.Vector([0,1]) + 
                        r.spring_forces_on_objects[time_step, object_idx]) # Change velocity as fn of gravity by dt and the spring forces
        
        # Detect collisions. And check that velocity is still moving as cause of motor
        if old_pos[1] <= (ground_height + 0.01) and old_velocity[1] < 0:
            
            old_velocity = tai.Vector([0,0])
        
        # Update position and velocity
        new_pos = old_pos + dt * old_velocity
        r.positions[time_step, object_idx] = new_pos
        
        new_velocity = old_velocity
        r.velocities[time_step, object_idx] = new_velocity
        
# -------------------------------------------------------------

def step_one(time_step: tai.i32, r):
    
    calculate_center_robot(time_step, r)
    simulate_neural_network_SH(time_step, r)
    simulate_neural_network_HM(time_step, r)
    simulate_springs(time_step, r)
    simulate_objects(time_step, r)
    
# -------------------------------------------------------------

def tune_hm_weights(r):
    for i in range(r.n_springs):
        for j in range(r.n_hidden_neurons):
            r.weightsHM[i, j] -= learning_rate * r.weightsHM.grad[i,j]
                
def tune_hidden_layer_biases(r):
    for i in range(r.n_hidden_neurons):
        r.bias_hidden[i] -= learning_rate * r.bias_hidden.grad[i]
        
def tune_sh_weights(r):
    for i in range(r.n_hidden_neurons):
        
        for j in tai.static(range(n_sin_waves)): 
            
            # Each of these variables has a gradient associated with it.
            r.weightsSH[i,j] -= learning_rate * r.weightsSH.grad[i,j]
            
        # Simulate the sensors inside the objects
        # First 2 sensors -> 'proprioceptive sensors'. Indicate position of that object wrt robots center of mass.
        for j in tai.static(range(r.n_objects)):

            # Add to i-th neuron, the horizontal dist between j-th object and bot's center
            r.weightsSH[i, j* 4 + n_sin_waves]        -= learning_rate * r.weightsSH.grad[i, j* 4 + n_sin_waves]
            
            # Add to i-th neuron, the vertical dist between j-th object and bot's center
            r.weightsSH[i, j* 4 + 1 + n_sin_waves]    -= learning_rate * r.weightsSH.grad[i, j* 4 + 1 + n_sin_waves] 
            
            r.weightsSH[i, j* 4 + 2 + n_sin_waves]    -= learning_rate * r.weightsSH.grad[i, j* 4 + 2 + n_sin_waves]
            r.weightsSH[i, j* 4 + 3 + n_sin_waves]    -= learning_rate * r.weightsSH.grad[i, j* 4 + 3 + n_sin_waves]
        
        # goal sensors -> how far the bot got?
        r.weightsSH[i, r.n_objects * 4 + n_sin_waves]       -= learning_rate * r.weightsSH.grad[i, r.n_objects * 4 + n_sin_waves]
        r.weightsSH[i, r.n_objects * 4 + n_sin_waves + 1]   -= learning_rate * r.weightsSH.grad[i, r.n_objects * 4 + n_sin_waves + 1] 
        
def tune_robots_brain(r):
    
    prev_w_SH = r.weightsSH
    prev_w_HM = r.weightsHM
    prev_w_hidden = r.hidden
    prev_w_bias_hidden = r.bias_hidden
    
    # Fine-tune hidden to motor layer
    tune_hm_weights(r)

    # Fine-tune the bias
    tune_hidden_layer_biases(r)
        
    # Fine-tune sensor to hidden layer
    tune_sh_weights(r)
    
    return prev_w_SH, prev_w_HM, prev_w_hidden, prev_w_bias_hidden

# -------------------------------------------------------------

# TODO: pass loss as parameter and move function to utils
# TODO: Save loss of re-simulated robot as well
def save_fitness_losses(r, robot_index):
    # Write information of the robot morphology to text
    with open(f"fitness/loss_{robot_index}.txt", 'w') as file:
        file.write(str(r.loss))
        file.close()
      
    # Track all losses over time independently  
    with open(f"trackers_loss/loss_{robot_index}.txt", 'a') as file:
        save_loss = str(r.loss) + "\n"
        file.write(save_loss)
        file.close()        
        
def save_controller_weights(r, robot_index, sim_step, opt_step, prev_w_SH, prev_w_HM, prev_w_hidden, prev_w_bias_hidden):
        
    # Get new optimized weights.
    weightsSH_arr = r.weightsSH.to_numpy()
    weightsHM_arr = r.weightsHM.to_numpy()
    hidden_arr = r.hidden.to_numpy()
    bias_hidden_arr = r.bias_hidden.to_numpy()
    
    # Check: 1st - Save always in first run. 2nd - If last loss is worse, Save controllers without updating.
    if sim_step > 1 and (opt_step == n_optimization_steps - 1): 
        prev_loss, last_loss = utils.check_last_and_prev_loss(robot_index)
        if last_loss > (prev_loss + 0.02):
            print(f"\nRobot {robot_index} not saving optimized controllers.")
            weightsSH_arr = prev_w_SH.to_numpy()
            weightsHM_arr = prev_w_HM.to_numpy()
            hidden_arr = prev_w_hidden.to_numpy()
            bias_hidden_arr = prev_w_bias_hidden.to_numpy()
            
    # Save controllers
    np.savez(f'controller/weights_{robot_index}.npz', weightsSH=weightsSH_arr, weightsHM=weightsHM_arr,
             hidden=hidden_arr, bias_hidden=bias_hidden_arr)
        
def load_controller_weights(r, robot_index):
    
    # Load the arrays from the .npz file
    data = np.load(f'controller/weights_{robot_index}.npz')

    # Access individual arrays by their keys
    r.weightsSH.from_numpy(data["weightsSH"])
    r.weightsHM.from_numpy(data["weightsHM"])
    r.hidden.from_numpy(data["hidden"])
    r.bias_hidden.from_numpy(data["bias_hidden"])
    
# -------------------------------------------------------------

def eliminate_individual(n_robot_population):
    
    # Lowest fitness is in our case the highest positive value because we want to minimize it.
    temp_lowest = -1000000
    idx_robot_delete = 0
    
    for robot_index in range(n_robot_population):
        with open(f"fitness/loss_{robot_index}.txt", 'r') as file:
            temp_loss = file.read()
            
            # Check that loss of robot is higher than previous highest.
            if float(temp_loss) > temp_lowest:
                
                # Save index of file to be deleted.
                idx_robot_delete = robot_index
                temp_lowest = float(temp_loss)
                print(f"\nNew Worst fitness robot {idx_robot_delete} -> {temp_lowest}")
    
    print(f"Eliminating robot {idx_robot_delete}\n")
    utils.track_values(idx_robot_delete)
    
    # Delete population, fitness and image files
    os.system(f"rm population/robot_{idx_robot_delete}.txt")
    os.system(f"rm fitness/loss_{idx_robot_delete}.txt")
    os.system(f"rm trackers_prob/prob_{idx_robot_delete}.txt")
    os.system(f"rm trackers_loss/loss_{idx_robot_delete}.txt")
    os.system(f"rm controller/weights_{idx_robot_delete}.npz")
    shutil.rmtree(f"images/robot_{idx_robot_delete}/")
    
    return idx_robot_delete
        
# -------------------------------------------------------------

def add_object(r, robot_index, is_spring_null):
    
    print(f"MUTATION: Adding Objects to robot {robot_index}")
    
    # Create between 1 and 3 new objects
    # Check of Mutation being to harsh on robots.
    if is_spring_null:
        n_new_objects = 2
    else:
        # n_new_objects = random.randint(1, 3)  # TODO: Add only 1 object at a time first
        n_new_objects = 1
    
    
    # Read from file existing objects
    with open(f"population/robot_{robot_index}.txt", 'r') as file:
        
        all_lines = file.readlines()
        
        # Get object positions
        old_obj_pos = eval(all_lines[0])
        
        # Get the largest_idx to start adding objects
        largest_idx, current_springs = utils.get_last_obj_index(all_lines[1:])
        new_largest_idx = largest_idx + 1
        
        # Generate objects
        new_obj_pos = utils.generate_obj_positions(n_new_objects)
        
        # Combine old and new objects
        total_obj_list = old_obj_pos + new_obj_pos
        r.startingObjectPositions = total_obj_list
        all_lines[0] = str(r.startingObjectPositions) + '\n'
        r.n_objects = len(r.startingObjectPositions)
        
        # Generate Springs. Randomly select if they are motorized or not.
        # TODO: There should be a randomization, where not all objects are connected between springs (no matter if they are motor or not)
        new_springs_robot = []
        for i in range(len(total_obj_list)):
            for j in range(new_largest_idx, len(total_obj_list)):
                if i < j:
                    is_motor = random.choice([0, 1])    
                    utils.create_spring(new_springs_robot, i, j, is_motor, total_obj_list)
        
        # TODO: Current springs are not given as 
        total_new_springs = current_springs + new_springs_robot
        r.springs = total_new_springs
        r.n_springs = len(r.springs)
    
    # Write the modified contents back to the file
    with open(f"population/robot_{robot_index}.txt", 'w') as file:
        
        # Write springs information
        for sublist in new_springs_robot:
            line = ' '.join(map(str, sublist)) + '\n'
            all_lines.append(line)
            
        file.writelines(all_lines)

def remove_object(r, robot_idx):
    
    # TODO: As simulation runs increase, number of possible objects to be removed increases as well.
    # max_obj_remove = int(math.sqrt(initial_robot_population - n_robot_population))
    # n_remove_objects = random.randint(1, max_obj_remove)
    n_remove_objects = 1
    
    # Check that there is at least 1 spring at all moments
    if r.n_springs <= 1:
        add_object(r, robot_idx, True)
        return
    
    with open(f"population/robot_{robot_idx}.txt", 'r') as file:
        
        # Delete randomly object index
        all_lines = file.readlines()
        
        # Get object positions
        old_obj_pos = eval(all_lines[0])
        r.startingObjectPositions = old_obj_pos
        original_obj_size = len(r.startingObjectPositions)
        
        # Check that there are more than 2 objects
        if original_obj_size <= 2:
            
            # Mutate by adding object
            add_object(r, robot_idx, True)
            return
        
        # Get springs from file
        spring_lines = all_lines[1:]
        
        for _ in range(n_remove_objects):

            # Remove Obj position
            object_index_remove = random.randint(2, len(r.startingObjectPositions)-1)   
            
            if object_index_remove <= len(r.startingObjectPositions):
                
                print(f"MUTATION: Removing Objects to robot {robot_idx}")
                
                # Remove Objects
                r.startingObjectPositions.pop(object_index_remove)
            
                # Remove links containing that index
                spring_lines = utils.check_object_index(spring_lines, object_index_remove)
                
            # Check there is still a robot spring
            if len(spring_lines) == 0:
                print(f"WARNING: Deleting all springs... Add Object to robot {robot_idx}")
                add_object(r, robot_idx, True)
                
                # Need to re-set spring lines
                temp_spring_lines = file.readlines()
                spring_lines = temp_spring_lines[1:]
                
                # Re-set objects
                old_obj_pos = eval(all_lines[0])
                r.startingObjectPositions = old_obj_pos
                print(f"Robot {robot_idx}, total springs {r.n_springs}, {len(spring_lines)}")
                   
        # Rewrite
        all_lines[0] = str(r.startingObjectPositions) + '\n'
        r.n_objects = len(r.startingObjectPositions)
        
    # Write the modified contents back to the file. 
    new_springs_robot = []
    with open(f"population/robot_{robot_idx}.txt", 'w') as file:
        file.write(all_lines[0])
        
        # Write springs information
        for line in spring_lines:
            file.write(line)
            line = line.rstrip('\n').split()
            
            # Modify data to be store in the robot spring info field.
            converted_parts = [int(line[0]), int(line[1]), float(line[2]), int(line[3])]
            new_springs_robot.append(converted_parts)
            
    r.springs = new_springs_robot
    r.n_springs = len(r.springs)
    print(f"Robot {robot_idx}, total springs {r.n_springs}, {len(spring_lines)}")

def mutate_population(r, n_robot_population):

    for robot_idx in range(n_robot_population):
        
        # Choose mutation action based on updated probability
        mutations = ["Add", "Remove", "Nothing"]
        add_prob, remove_prob, nothing_prob = utils.get_probabilities(robot_idx)
        mutation_action = random.choices([0, 1, 2], weights=[add_prob, remove_prob, nothing_prob])[0]
        print(f"Robot {robot_idx} - Mutation: {mutations[mutation_action]}. Probabilties are: {add_prob}, {remove_prob}, {nothing_prob}")
    
        if mutation_action == 0:
            # Add an object - Spring
            add_object(r, robot_idx, False)
   
        elif mutation_action == 1:
            # Remove an object - Spring
            remove_object(r, robot_idx)
            
        springs_population[robot_idx] = r.springs
        startingObjectPositions_population[robot_idx] = r.startingObjectPositions

# -------------------------------------------------------------

def set_fittest_robot_draw(robot_index):
    
    with open(f"population/robot_{robot_index}.txt", 'r') as file:
            
            all_lines = file.readlines()
            
            # Get object positions
            start_end_obj = eval(all_lines[0])
            
            # Get Springs
            spring_obj = all_lines[1:]
            _, spring_obj = utils.get_last_obj_index(spring_obj)
            
    springs_population[robot_index] = spring_obj
    startingObjectPositions_population[robot_index] = start_end_obj
        
# -------------------------------------------------------------

if __name__ == "__main__":
    # Clean files for simulation
    utils.remove_files_before_simulation()

    # Create population of robots
    springs_population, startingObjectPositions_population = create_population(n_robot_population)  

    for simulation_step in range(initial_robot_population):
        
        print(f"\nSIMULATION RUN {simulation_step+1}")
        robot_drawing = []
        
        for robot_idx in range(n_robot_population):
            print(f"\nWorking on robot {robot_idx}")
            
            # Get objects and springs individual robots
            springs = springs_population[robot_idx]
            startingObjectPositions = startingObjectPositions_population[robot_idx]

            r = Robot(springs, startingObjectPositions, max_steps)
            
            # TODO: There has to be a more efficient way to do this, than to have 2 different losses being calculated.
            # Create loss for that robot
            loss = tai.field(dtype=tai.f32, shape=(), needs_grad=True) # 0-D tensor
            tai.root.lazy_grad()
            
            # TODO: For each optimization step, there should only be an update of controllers when it is better, not at last
            for opt_step in range(n_optimization_steps):        
                
                if opt_step == 0:
                    Initialize_Neural_Network(r)
                else:
                    load_controller_weights(r, robot_idx)
                    
                # Init Robot
                Initialize(r)
                
                with tai.Tape(loss):
                
                    # Simulate
                    Simulate(r)
                    loss[None] = 0.0
                    
                    Compute_loss()
                
                # TODO: Find the solution to some robots having nan loss
                if loss[None] < r.loss[None]:
                    if np.isnan(loss[None]):
                        r.loss[None] = 100000
                    else:
                        r.loss[None] = float(loss[None])
                    
                print(f"Robot {robot_idx} - Opt Step {opt_step}. Loss: {r.loss[None]}")
                
                # Fine-tune the brain of the robot
                prev_w_SH, prev_w_HM, prev_w_hidden, prev_w_bias_hidden = tune_robots_brain(r)
                
                # Draw First Optimization Step
                if opt_step == 0 and simulation_step == 0:
                    os.system(f"rm images/robot_{robot_idx}/*.png")
                    Draw(r, 0, robot_idx)
                
                # Save the fitness loss
                if opt_step == n_optimization_steps - 1:
                    save_fitness_losses(r, robot_idx)
                
                # Save optimized steps Across simulation runs.
                save_controller_weights(r, robot_idx, simulation_step, opt_step, prev_w_SH, prev_w_HM, prev_w_hidden, prev_w_bias_hidden)  
             
        # Set new number of individuals in population
        n_robot_population -= 1
            
        if n_robot_population != 0:
            
            # Eliminate the lowest-ranked individual by fitness
            idx_robot_delete = eliminate_individual(n_robot_population)
            
            # Re-order file indices for simplicity
            utils.update_files()
            
            # Delete from list of springs and objects
            springs_population.pop(idx_robot_delete)
            startingObjectPositions_population.pop(idx_robot_delete)
            
            # Update Action probabilities
            utils.update_probabilities(robot_idx, simulation_step)
                
            # Mutate remaining individuals
            mutate_population(r, n_robot_population)
            
        else:
            print(f"\nEND SIMULATION")
            
            # Re-read the fittest robot objects and springs
            # set_fittest_robot_draw(0)
            time.sleep(2)
            
            # Debug check
            print(f"The final robot is:\n{springs_population[0]}\n{startingObjectPositions_population[0]}")
            
            # Track values for analyzing and plotting
            utils.track_values(0)
            
            # Draw final robot. 
            Draw(r, max_steps, 0)
            
            # Create video
            experiment_name = "X_7"
            utils.create_video(experiment_name, "fit")