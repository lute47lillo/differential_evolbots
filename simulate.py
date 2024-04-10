
import taichi as tai
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import random
from robot import Robot

# -------------------------------------------------------------
# move the object to max use the taichi's differentiable routine.
# TODO: Taichi's Fields documentation !

"""
    GLOBAL VARIABLES
"""
max_steps = 200
ground_height = 0.05
stiffness = 1000 # Strength of the spring in the example
dt = 0.01 # Amount of time that elapses between time steps.
gravity = -9.8
learning_rate = 1
x_offset = 0.1 # How far from left screen robot starts
damping = 0.6 # Is a constant that controls how much you slow the velocity of the object to which is applied. (1-damping) = X% reductions each time-step
n_hidden_neurons = 32
n_sin_waves = 10

"""
    UTIL FUNCTIONS
"""
# -----------------------------------------------------------------

def n_sensors(n_objects):
    return n_sin_waves + 4 * n_objects + 2

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

# TODO: Delete the files of population/ to be from sracth on every new run.
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
    startingObjectPositions.append([x_offset, ground_height])
    
    # How many more are created?
    total_objects = random.randint(1, 5)
    
    # Generate objects
    for _ in range(total_objects):
        
        # Generate random x_pos and y_pos
        obj_x_pos = random.uniform(0, 0.4)
        obj_y_pos = random.uniform(0, 0.4)
        
        # Check there is no object in same x and y.
        for created_obj in startingObjectPositions:
            x, y = created_obj
            # Add an arbitrary offset to undraw
            if x == obj_x_pos and y == obj_y_pos:
                obj_x_pos += 0.05
                obj_y_pos += 0.05
        
        # Add object
        startingObjectPositions.append([x_offset + obj_x_pos, ground_height + obj_y_pos])
        
    # Generate Springs. Randomly select if they are motorized or not.
    for i in range(len(startingObjectPositions)):
        for j in range(i+1, len(startingObjectPositions)):
            is_motor = random.choice([0, 1])
            create_spring(springs_robot, i, j, is_motor, startingObjectPositions)
 
    # Write information of the robot morphology to text
    with open(f"population/robot_{robot_index}.txt", 'w') as file:
        for sublist in springs_robot:
            line = ' '.join(map(str, sublist)) + '\n'
            file.write(line)
        # line = ' '.join(map(str, springs_robot)) + '\n'
        # file.write(line)
 
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
    # Second component of zeroth object. Loss = Height of 0th objects at last time_step
    loss[None] -= r.positions[r.max_steps-1, 0][1]
    
# -------------------------------------------------------------

@tai.kernel
def calculate_center_robot(time_step: tai.i32):
    
    for _ in range(1): #Taichi sugar code
        c = tai.Vector([0.0, 0.0])
        
        for i in range(r.n_objects):
            c += r.positions[time_step, i] # Position of i-th object at time_step
            
        r.center[time_step] = c / r.n_objects

# -------------------------------------------------------------

def Draw(frame_offset):
    
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
            
        # Draw the springs
 
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

                
        tai_gui.show(f"images/test_{frame_offset+time_step}.png")

# -------------------------------------------------------------

def Initialize():
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
        
def Initialize_Neural_Network():
    """
         Definition
        -----------
            Initialize the paramters of the Neural Network.
    """
    # Initialize sensor to hidden neurons
    for i in range(r.n_hidden_neurons):
        for j in range(n_sensors(r.n_objects)):
            r.weightsSH[i,j] = np.random.randn() * 0.2 - 0.1
    
    # Init bias for hidden neurons
    for i in range(r.n_hidden_neurons):
        r.bias_hidden[i] = np.random.randn() * 2 - 1
        
    # Init weights
    for i in range(r.n_springs):
        for j in range(r.n_hidden_neurons):
            r.weightsHM[i,j] = np.random.randn() * 0.2 - 0.1
            
# -------------------------------------------------------------

def Simulate():

    for time_step in range(1, r.max_steps):
        
        step_one(time_step)

# -------------------------------------------------------------

# transform senation into action
@tai.kernel
def simulate_neural_network_SH(time_step: tai.i32):
    
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
def simulate_neural_network_HM(time_step: tai.i32):
    
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
def simulate_springs(time_step: tai.i32):
    
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
        # TODO: Adapt the force_of_piston constant to be an actual variable
        # spring_resting_length = spring_resting_length + 0.08 * spring_actuation[spring_idx] * tai.sin(0.9*time_step)
        
        # Newer version takes the motorized action form the NN. Keep value small
        spring_resting_length = spring_resting_length + 0.07 * r.spring_actuation[spring_idx] *r. actuation[time_step, spring_idx]
        
        # Difference between current and supposed initial at that index
        spring_difference = curr_rest_length - spring_resting_length
        
        # Apply force proportionally to the difference between the at rest lengths. Normalized result by current distance
        # Turn the restoring force to a vector parallet to the vector connecting the two objects (by mult by the distance_a_b)
        # Big distances (denominator says) should NOT have big forces -> Swinging pendulum effect without stability
        # We would also need to add strength to the spring -> stiffness
        r.spring_restoring_forces[time_step, spring_idx] = (dt * spring_difference  * stiffness / curr_rest_length) * distance_a_b
        
        # Apply the force. - symbol means pulling force
        r.spring_forces_on_objects[time_step, object_a_index] +=  -1.5 * r.spring_restoring_forces[time_step, spring_idx]
        r.spring_forces_on_objects[time_step, object_b_index] +=  1.5 * r.spring_restoring_forces[time_step, spring_idx]
        
# -------------------------------------------------------------

@tai.kernel
def simulate_objects(time_step: tai.i32):
    
    for object_idx in range(r.n_objects):
        
        # Get old position and velocity
        old_pos = r.positions[time_step-1, object_idx]
        old_velocity = (damping * r.velocities[time_step-1, object_idx] +
                        dt * gravity  * tai.Vector([0,1]) + 
                        r.spring_forces_on_objects[time_step, object_idx]) # Change velocity as fn of gravity by dt and the spring forces
        
        # Detect collisions. And check that velocity is still moving as cause of motor
        if old_pos[1] <= ground_height and old_velocity[1] < 0:
            
            old_velocity = tai.Vector([0,0])
        
        # Update position and velocity
        new_pos = old_pos + dt * old_velocity
        r.positions[time_step, object_idx] = new_pos
        
        new_velocity = old_velocity
        r.velocities[time_step, object_idx] = new_velocity
        
# -------------------------------------------------------------

def step_one(time_step: tai.i32):
    
    calculate_center_robot(time_step)
    simulate_neural_network_SH(time_step)
    simulate_neural_network_HM(time_step)
    simulate_springs(time_step)
    simulate_objects(time_step)
    
# -------------------------------------------------------------

def tune_hm_weights():
    for i in range(r.n_springs):
        for j in range(r.n_hidden_neurons):
            r.weightsHM[i, j] -= learning_rate * r.weightsHM.grad[i,j]
                
def tune_hidden_layer_biases():
    for i in range(r.n_hidden_neurons):
        r.bias_hidden[i] -= learning_rate * r.bias_hidden.grad[i]
        
def tune_sh_weights():
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
        
def tune_robots_brain():
    
    # Fine-tune hidden to motor layer
    tune_hm_weights()

    # Fine-tune the bias
    tune_hidden_layer_biases()
        
    # Fine-tune sensor to hidden layer
    tune_sh_weights()
            
# -------------------------------------------------------------

# Create Video
def Create_video():
    os.system("rm movie.p4")
    os.system(" ffmpeg -i images/test_%d.png movie.mp4")

# -------------------------------------------------------------

# Create population of robots
n_robot_population = 2
os.system("rm population/*.txt")
springs_population, startingObjectPositions_population = create_population(n_robot_population)  
    
robot_drawing = []
for robot_idx in range(n_robot_population):
    print(f"Working on robot {robot_idx+1}")
    
    # Get objects and springs individual robots
    springs = springs_population[robot_idx]
    startingObjectPositions = startingObjectPositions_population[robot_idx]
        
    r = Robot(springs, startingObjectPositions, max_steps)
    
    # Create loss of that robot
    loss = tai.field(dtype=tai.f32, shape=(), needs_grad=True) # 0-D tensor
    tai.root.lazy_grad()
    
    for opt_step in range(2):        
        
        Initialize_Neural_Network()
        Initialize()
        
        with tai.Tape(loss):
        
            # Simulate
            Simulate()
            loss[None] = 0.0
            
            Compute_loss()
            
        r.loss = float(loss[None])
        print(f"Robot {robot_idx} - Opt Step {opt_step+1}. Loss: {r.loss}")
        
        # Fine-tune the brain of the robot
        # tune_robots_brain(weightsHM, weightsSH, bias_hidden)
        tune_robots_brain()
        
        if opt_step == 0:
            os.system("rm images/*.png")
            Draw(0)
            
    loss[None] = 0.0
    
Draw(max_steps)
    
# Create the video
Create_video()