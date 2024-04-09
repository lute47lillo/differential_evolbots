
import taichi as tai
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import random

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


"""
    UTIL FUNCTIONS
"""
# Objects connected by Springs
# springs = []
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

def create_population(n_robots_population):
    
    springs_population = []
    startingObjectPositions_population = []
    
    # Simulate individual robot and gather springs
    for idx_robot in range(n_robots_population):
        springs, startingObjectPositions = simulate_robot(idx_robot)
        springs_population.append(springs)   
        startingObjectPositions_population.append(startingObjectPositions) 
    
    return springs_population, startingObjectPositions_population

springs_population, startingObjectPositions_population = create_population(5)   

# Placeholder, get data of first robots  
springs = springs_population[0]
startingObjectPositions = startingObjectPositions_population[0]
# -------------------------------------------------------------

# Create a field w/ max_steps X n_objects entries.
# It is stored in positions. Needs to be defined previously
#Vector of length 2. Real Values
real = tai.f32
tai.init(default_fp = real) # Init TAI
vec =  lambda: tai.Vector.field(2, dtype=real) 

# -------------------------------------------------------------

n_objects = len(startingObjectPositions)
n_springs = len(springs)

# Store as Taichi fields
spring_anchor_a = tai.field(tai.i32)
spring_anchor_b = tai.field(tai.i32)
spring_at_rest_length = tai.field(tai.f32)
spring_actuation = tai.field(tai.i32) # Wheter or not the spring contains a piston motor that deals with the length. Binary value

# -------------------------------------------------------------
# NEURAL NETWORK 

# Sensor to Hidden neurons and weights
weightsSH = tai.field(tai.f32)

# Arbitrary choices
n_hidden_neurons = 32
n_sin_waves = 10

def n_sensors():
    # Simulate Central Pattern Generators (CPPNS). 4 sensors per objects. 2 global sensors (horizontal and vertical dist)
    return n_sin_waves + 4 * n_objects + 2

# Put weights from Sensors to hidden neurons
tai.root.dense(tai.ij, (n_hidden_neurons, n_sensors())).place(weightsSH)

# Hidden to Motor neurons and weights
weightsHM = tai.field(tai.f32)
tai.root.dense(tai.ij, (n_springs, n_hidden_neurons)).place(weightsHM)

# Capture motor value to be sent to every spring at every time_step
actuation = tai.field(tai.f32)
tai.root.dense(tai.ij, (max_steps, n_springs)).place(actuation)
# -------------------------------------------------------------
# Store positions of every object at every time step.
# Where each position is a vector of length 2. x and y.
positions = vec()
tai.root.dense(tai.i, max_steps).dense(tai.j, n_objects).place(positions)

# Gradients of position. Changing as a function of the loss per time step.
tai.root.dense(tai.i, max_steps).dense(tai.j, n_objects).place(positions.grad)


velocities = vec()
tai.root.dense(tai.i, max_steps).dense(tai.j, n_objects).place(velocities)

# Taichi Structure for springs. Turn Spring anchor A & B from integer into field
tai.root.dense(tai.i, n_springs).place(spring_anchor_a, spring_anchor_b, spring_at_rest_length, spring_actuation) 

# Forces of the springs
spring_restoring_forces = vec()
tai.root.dense(tai.i, max_steps).dense(tai.j, n_springs).place(spring_restoring_forces)

# Forces acting on the objects
spring_forces_on_objects = vec()
tai.root.dense(tai.i, max_steps).dense(tai.j, n_objects).place(spring_forces_on_objects)

# -------------------------------------------------------------
# Determines center of the bot by the time_step
center = vec()
tai.root.dense(tai.i, max_steps).place(center)

# Determine the goal of the robot.
goal = vec()
tai.root.place(goal)

# Create field for N hidden neurons at each time_step
hidden = tai.field(tai.f32)
tai.root.dense(tai.ij, [max_steps, n_hidden_neurons]).place(hidden)

# Create bias. One per each hidden neuron. Total N bias
bias_hidden = tai.field(tai.f32)
tai.root.dense(tai.i, n_hidden_neurons).place(bias_hidden)

# -------------------------------------------------------------
loss = tai.field(dtype=tai.f32, shape=()) # 0-D tensor
# Gradients
tai.root.lazy_grad()

# Execute by Taichi and not python by using decorater
@tai.kernel
def Compute_loss():
    
    # Focus on position of the objects to determine loss fn. Arbitrary choice
    # Second component of zeroth object. Loss = Height of 0th objects at last time_step
    loss[None] -= positions[max_steps-1, 0][1]
    
    # calculate the loss based on the distance travelled to the right
    
# -------------------------------------------------------------

@tai.kernel
def calculate_center_robot(time_step: tai.i32):
    
    for _ in range(1): #Taichi sugar code
        c = tai.Vector([0.0, 0.0])
        
        for i in range(n_objects):
            c += positions[time_step, i] # Position of i-th object at time_step
            
        center[time_step] = c / n_objects

# -------------------------------------------------------------

def Draw(frame_offset):
    
    for time_step in range(0, max_steps):
        # Draw the robot using Taichi's built-iGUI. (x,y) size of window
        tai_gui = tai.GUI("Robot", (512, 512),
                          background_color=0xFFFFFF, show_gui=False)

        # Draw the floow
        tai_gui.line(begin=(0, ground_height), end=(1, ground_height),
                     color=0x0, radius=3)

        # Draw the object
        for object_idx in range(n_objects):
            
            # Get at time step for each object -> (x,y) coords
            x = positions[time_step, object_idx][0]
            y = positions[time_step, object_idx][1]
            tai_gui.circle((x,y), color=0x0, radius=7)
            
        # Draw the springs
        for spring_idx in range(n_springs):
            object_a_index = spring_anchor_a[spring_idx]
            object_b_index = spring_anchor_b[spring_idx]
            
            # Get the positions of spring A at every time step
            position_a = positions[time_step, object_a_index]
            position_b = positions[time_step, object_b_index]
            
            has_motor = springs[spring_idx][3]
            
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
    for object_idx in range(n_objects):
        # Temp, just propagate positions.
        positions[0, object_idx] = startingObjectPositions[object_idx]
        
        # Set initial velocites
        velocities[0, object_idx] = [0, -0.1]
        
    for spring_idx in range(n_springs):
        s = springs[spring_idx] # Get spring
        spring_anchor_a[spring_idx]         = s[0] # the a object of that spring
        spring_anchor_b[spring_idx]         = s[1]
        spring_at_rest_length[spring_idx]   = s[2]
        spring_actuation[spring_idx]        = s[3]
        
    # Reset Positions (x,y) and velocities
    for i in range(1, max_steps):
        for j in range(n_objects):
            positions[i,j][0] = 0.0
            positions[i,j][1] = 0.0
            velocities[i,j][0] = 0.0
            velocities[i,j][1] = 0.0
    
    for i in range(1, max_steps):
        for j in range(n_springs):
            spring_restoring_forces[i,j][0] = 0.0   
            spring_restoring_forces[i,j][1] = 0.0 
            
    # Restore forces apply on the object by the springs
    for i in range(1, max_steps):
        for j in range(n_objects):
            spring_forces_on_objects[i,j][0] = 0.0   
            spring_forces_on_objects[i,j][1] = 0.0 
    
    # Reset values
    for i in range(1, max_steps):
        for j in range(n_hidden_neurons):
            hidden[i,j] = 0.0
            
    # Reset values of motor neurons
    for i in range(1, max_steps):
        for j in range(n_springs):
            actuation[i,j] = 0.0
            
    goal[None] = [0.9, 0.2]
        
def Initialize_Neural_Network():
    """
         Definition
        -----------
            Initialize the paramters of the Neural Network.
    """
    # Initialize sensor to hidden neurons
    for i in range(n_hidden_neurons):
        for j in range(n_sensors()):
            weightsSH[i,j] = np.random.randn() * 0.2 - 0.1
    
    # Init bias for hidden neurons
    for i in range(n_hidden_neurons):
        bias_hidden[i] = np.random.randn() * 2 - 1
        
    # Init weights
    for i in range(n_springs):
        for j in range(n_hidden_neurons):
            weightsHM[i,j] = np.random.randn() * 0.2 - 0.1
            
# -------------------------------------------------------------
def Simulate():
    
    for time_step in range(1, max_steps):
        
        # Update position of object.
        step_one(time_step)

# -------------------------------------------------------------

# transform senation into action
@tai.kernel
def simulate_neural_network_SH(time_step: tai.i32):
    
    # Propagate values
    for i in range(n_hidden_neurons):
        activation = 0.0
        
        # for each of the CPPNS
        for j in tai.static(range(n_sin_waves)): 
            # increment act of i-th neuron by the sinuoisoid of time_step. j is a phase offset
            activation += weightsSH[i,j] * tai.sin(30 * time_step*dt + \
                                                    2* math.pi / n_sin_waves * j) 
            
        # Simulate the sensors inside the objects
        # First 2 sensors -> 'proprioceptive sensors'. Indicate position of that object wrt robots center of mass.
        for j in tai.static(range(n_objects)):
            offset = positions[time_step, j] - center[time_step]
            
            # Add to i-th neuron, the horizontal dist between j-th object and bot's center
            activation += 0.25 * weightsSH[i, j* 4 + n_sin_waves] * offset[0]
            # Add to i-th neuron, the vertical dist between j-th object and bot's center
            activation += 0.25 * weightsSH[i, j* 4 + 1 + n_sin_waves] * offset[1]
            
            activation += 0.25 * weightsSH[i, j* 4 + 2 + n_sin_waves] * positions[time_step, j][1]
            activation += 0.25 * weightsSH[i, j* 4 + 3 + n_sin_waves] * positions[time_step, j][1]
        
        # goal sensors -> how far the bot got?
        activation += 0.25 * weightsSH[i, n_objects * 4 + n_sin_waves] * (goal[None][0] - center[time_step][0])
        activation += 0.25 * weightsSH[i, n_objects * 4 + n_sin_waves + 1] * (goal[None][1] - center[time_step][1])
            
        # Apply non-linearity
        activation += bias_hidden[i]
        activation = tai.tanh(activation)
        
        # Store in a hidden neuron at every time_step
        hidden[time_step, i] = activation
# -------------------------------------------------------------

@tai.kernel
def simulate_neural_network_HM(time_step: tai.i32):
    
    # For every spring..
    for i in range(n_springs):
        
        activation = 0.0 # Init for each motor neuron
        
        # Visit each hidden neuron. And sum up influence of all hidden neurons for each motor.
        for j in tai.static(range(n_hidden_neurons)):
            
            activation += weightsHM[i, j] * hidden[time_step, j] # pre-synaptic hidden neuron value
            
        activation = tai.tanh(activation)
        actuation[time_step, i] = activation
            

# -------------------------------------------------------------

@tai.kernel
def simulate_springs(time_step: tai.i32):
    # Simulate the physics of each springs at initial step
    for spring_idx in range(n_springs):
        object_a_index = spring_anchor_a[spring_idx]
        object_b_index = spring_anchor_b[spring_idx]
        
        # Get most recent position.
        position_a = positions[time_step-1, object_a_index]
        position_b = positions[time_step-1, object_b_index]
        
        # Compute distance between objects -> Length of spring at rest
        distance_a_b = position_a - position_b
        curr_rest_length = distance_a_b.norm()
        
        spring_resting_length = spring_at_rest_length[spring_idx]
        
        # Applying the sinuisoidal function to have the piston of the motor (the cause of the movement be in that range)
        # TODO: Adapt the force_of_piston constant to be an actual variable
        # spring_resting_length = spring_resting_length + 0.08 * spring_actuation[spring_idx] * tai.sin(0.9*time_step)
        
        # Newer version takes the motorized action form the NN. Keep value small
        spring_resting_length = spring_resting_length + 0.07 * spring_actuation[spring_idx] * actuation[time_step, spring_idx]
        
        # Difference between current and supposed initial at that index
        spring_difference = curr_rest_length - spring_resting_length
        
        # Apply force proportionally to the difference between the at rest lengths. Normalized result by current distance
        # Turn the restoring force to a vector parallet to the vector connecting the two objects (by mult by the distance_a_b)
        # Big distances (denominator says) should NOT have big forces -> Swinging pendulum effect without stability
        # We would also need to add strength to the spring -> stiffness
        spring_restoring_forces[time_step, spring_idx] = (dt * spring_difference  * stiffness / curr_rest_length) * distance_a_b
        
        # Apply the force. - symbol means pulling force
        spring_forces_on_objects[time_step, object_a_index] +=  -1.5 * spring_restoring_forces[time_step, spring_idx]
        spring_forces_on_objects[time_step, object_b_index] +=  1.5 * spring_restoring_forces[time_step, spring_idx]
        
# -------------------------------------------------------------

@tai.kernel
def simulate_objects(time_step: tai.i32):
    
    for object_idx in range(n_objects):
        
        # Get old position and velocity
        old_pos = positions[time_step-1, object_idx]
        old_velocity = (damping * velocities[time_step-1, object_idx] +
                        dt * gravity  * tai.Vector([0,1]) + 
                        spring_forces_on_objects[time_step, object_idx]) # Change velocity as fn of gravity by dt and the spring forces
        
        # Detect collisions. And check that velocity is still moving as cause of motor
        if old_pos[1] <= ground_height and old_velocity[1] < 0:
            
            old_velocity = tai.Vector([0,0])
        
        # Update position and velocity
        new_pos = old_pos + dt * old_velocity
        positions[time_step, object_idx] = new_pos
        
        new_velocity = old_velocity
        velocities[time_step, object_idx] = new_velocity
        
# -------------------------------------------------------------

def step_one(time_step: tai.i32):
    
    calculate_center_robot(time_step)
    simulate_neural_network_SH(time_step)
    simulate_neural_network_HM(time_step)
    simulate_springs(time_step)
    simulate_objects(time_step)
    
# -------------------------------------------------------------
def tune_hm_weights():
    for i in range(n_springs):
        for j in range(n_hidden_neurons):
            weightsHM[i, j] -= learning_rate * weightsHM.grad[i,j]
                
def tune_hidden_layer_biases():
    for i in range(n_hidden_neurons):
        bias_hidden[i] -= learning_rate * bias_hidden.grad[i]
        
def tune_sh_weights():
    for i in range(n_hidden_neurons):
        
        for j in tai.static(range(n_sin_waves)): 
            
            # Each of these variables has a gradient associated with it.
            weightsSH[i,j] -= learning_rate * weightsSH.grad[i,j]
            
        # Simulate the sensors inside the objects
        # First 2 sensors -> 'proprioceptive sensors'. Indicate position of that object wrt robots center of mass.
        for j in tai.static(range(n_objects)):

            # Add to i-th neuron, the horizontal dist between j-th object and bot's center
            weightsSH[i, j* 4 + n_sin_waves]        -= learning_rate * weightsSH.grad[i, j* 4 + n_sin_waves]
            
            # Add to i-th neuron, the vertical dist between j-th object and bot's center
            weightsSH[i, j* 4 + 1 + n_sin_waves]    -= learning_rate * weightsSH.grad[i, j* 4 + 1 + n_sin_waves] 
            
            weightsSH[i, j* 4 + 2 + n_sin_waves]    -= learning_rate * weightsSH.grad[i, j* 4 + 2 + n_sin_waves]
            weightsSH[i, j* 4 + 3 + n_sin_waves]    -= learning_rate * weightsSH.grad[i, j* 4 + 3 + n_sin_waves]
        
        # goal sensors -> how far the bot got?
        weightsSH[i, n_objects * 4 + n_sin_waves]       -= learning_rate * weightsSH.grad[i, n_objects * 4 + n_sin_waves]
        weightsSH[i, n_objects * 4 + n_sin_waves + 1]   -= learning_rate * weightsSH.grad[i, n_objects * 4 + n_sin_waves + 1] 
        
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

def run_simulation():
      
    # Initialize the robot.
    Initialize()

    # Automated Differentiation.
    with tai.Tape(loss):
        
        Simulate()
        
        loss[None] = 0.0
        Compute_loss()

    print(loss[None])


Initialize_Neural_Network()

# Run simulation
for opt_step in range(2):
    
    run_simulation()
    
    if opt_step == 0:
        os.system("rm images/*.png")
        Draw(0)
        
    tune_robots_brain()

Draw(max_steps)

    
# os.system("rm images/*.png")
# Draw(0)

# Create the video
Create_video()


