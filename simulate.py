
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

# Create a field w/ max_steps X n_objects entries.
# It is stored in positions. Needs to be defined previously
#Vector of length 2. Real Values
real = tai.f32
tai.init(default_fp = real) # Init TAI
vec =  lambda: tai.Vector.field(2, dtype=real) 

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

# TODO: Aa placeholder - Create a Population of 2 robots
# Delete old experimental population
os.system("rm population/*.txt")
springs_population, startingObjectPositions_population = create_population(2)   


# Placeholder, get data of first robots  
springs = springs_population[0]
startingObjectPositions = startingObjectPositions_population[0]

n_objects = len(startingObjectPositions)
n_springs = len(springs)

# -----------------------------------------------------------------

# def init_robot_objects_ds(startingObjectPositions, vec):
#     n_objects = len(startingObjectPositions)
    
#     # Store positions of every object at every time step.
#     # Where each position is a vector of length 2. x and y.
#     # positions = vec()
#     tai.root.dense(tai.i, max_steps).dense(tai.j, n_objects).place(positions)

#     # Gradients of position. Changing as a function of the loss per time step.
#     tai.root.dense(tai.i, max_steps).dense(tai.j, n_objects).place(positions.grad)

#     # velocities = vec()
#     tai.root.dense(tai.i, max_steps).dense(tai.j, n_objects).place(velocities)
    
#     # Forces acting on the objects
#     # spring_forces_on_objects = vec()
#     tai.root.dense(tai.i, max_steps).dense(tai.j, n_objects).place(spring_forces_on_objects)

    # return positions, velocities, spring_forces_on_objects

# positions, velocities, spring_forces_on_objects = init_robot_objects_ds(startingObjectPositions, vec)

# -----------------------------------------------------------------

def init_robot_springs_ds(springs, vec):
    n_springs = len(springs)
    
    # Store as Taichi fields
    spring_anchor_a = tai.field(tai.i32)
    spring_anchor_b = tai.field(tai.i32)
    spring_at_rest_length = tai.field(tai.f32)
    spring_actuation = tai.field(tai.i32)
    
    # Capture motor value to be sent to every spring at every time_step
    actuation = tai.field(tai.f32)
    tai.root.dense(tai.ij, (max_steps, n_springs)).place(actuation)
    
    # Taichi Structure for springs. Turn Spring anchor A & B from integer into field
    tai.root.dense(tai.i, n_springs).place(spring_anchor_a, spring_anchor_b, spring_at_rest_length, spring_actuation) 

    # Forces of the springs
    spring_restoring_forces = vec()
    tai.root.dense(tai.i, max_steps).dense(tai.j, n_springs).place(spring_restoring_forces)
    
    spring_anchors = (spring_anchor_a, spring_anchor_b)

    return spring_anchors, spring_at_rest_length, spring_actuation, spring_restoring_forces, actuation

# spring_anchors, spring_at_rest_length, spring_actuation, spring_restoring_forces, actuation = init_robot_springs_ds(springs, vec)
# spring_anchor_a, spring_anchor_b = spring_anchors

# -------------------------------------------------------------

# TODO: Make controller also modifiable in size
def init_robot_weights_ds(startingObjectPositions, n_hidden_neurons):
    
    # Attributes
    n_objects = len(startingObjectPositions)
    
    # Sensor to Hidden neurons and weights - Put weights from Sensors to hidden neurons
    weightsSH = tai.field(tai.f32)
    tai.root.dense(tai.ij, (n_hidden_neurons, n_sensors(n_objects))).place(weightsSH)

    # Hidden to Motor neurons and weights
    weightsHM = tai.field(tai.f32)
    tai.root.dense(tai.ij, (n_springs, n_hidden_neurons)).place(weightsHM)
    
    # Create field for N hidden neurons at each time_step
    hidden = tai.field(tai.f32)
    tai.root.dense(tai.ij, [max_steps, n_hidden_neurons]).place(hidden)

    # Create bias. One per each hidden neuron. Total N bias
    bias_hidden = tai.field(tai.f32)
    tai.root.dense(tai.i, n_hidden_neurons).place(bias_hidden)
    
    return weightsSH, weightsHM, hidden, bias_hidden

# weightsSH, weightsHM, hidden, bias_hidden = init_robot_weights_ds(startingObjectPositions, n_hidden_neurons)

# -------------------------------------------------------------
def init_robot_goal(vec):
    
    goal = vec()
    tai.root.place(goal)
    
    return goal

# TODO: Adapt based on the robot task
def init_robot_center(vec):
    
    center = vec()
    tai.root.dense(tai.i, max_steps).place(center)
    
    return center

# -------------------------------------------------------------
# -------------------------------------------------------------

# Execute by Taichi and not python by using decorater
@tai.kernel
def Compute_loss():
    
    # Focus on position of the objects to determine loss fn. Arbitrary choice
    # Second component of zeroth object. Loss = Height of 0th objects at last time_step
    loss[None] -= r.positions[r.max_steps-1, 0][1]
    # r.loss[None] = loss[None]
    # calculate the loss based on the distance travelled to the right
    
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

# def Initialize(positions: tai.f32, goal, velocities, spring_anchors, spring_at_rest_length, spring_actuation,
#                spring_restoring_forces, spring_forces_on_objects, hidden, actuation: tai.f32):
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
        s = springs[spring_idx] # Get spring
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
        for j in range(n_sensors(n_objects)):
            r.weightsSH[i,j] = np.random.randn() * 0.2 - 0.1
    
    # Init bias for hidden neurons
    for i in range(r.n_hidden_neurons):
        r.bias_hidden[i] = np.random.randn() * 2 - 1
        
    # Init weights
    for i in range(r.n_springs):
        for j in range(r.n_hidden_neurons):
            r.weightsHM[i,j] = np.random.randn() * 0.2 - 0.1
            
# -------------------------------------------------------------
# def Simulate(positions: tai.f32, center: tai.f32, goal: tai.f32, velocities, weightsSH, weightsHM, bias_hidden, hidden, actuation: tai.f32, spring_pack):
def Simulate():

    for time_step in range(1, r.max_steps):
        
        # Update position of object.
        # step_one(positions, center, goal, velocities, weightsSH, weightsHM, bias_hidden, hidden, actuation, spring_pack,time_step)
        step_one(time_step)
        
    print("Simulated")


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
        activation += 0.25 * r.weightsSH[i, n_objects * 4 + n_sin_waves] * (r.goal[None][0] - r.center[time_step][0])
        activation += 0.25 * r.weightsSH[i, n_objects * 4 + n_sin_waves + 1] * (r.goal[None][1] - r.center[time_step][1])
            
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

# def step_one(positions: tai.f32, center: tai.f32, goal: tai.f32, velocities, weightsSH, weightsHM, bias_hidden, hidden, actuation, spring_pack, time_step: tai.i32):
def step_one(time_step: tai.i32):

    # calculate_center_robot(positions, center, time_step)
    # simulate_neural_network_SH(positions, center, goal, weightsSH, bias_hidden, hidden, time_step)
    # simulate_neural_network_HM(weightsHM, hidden, actuation, time_step)
    # simulate_springs(positions, spring_pack, actuation, time_step)
    # simulate_objects(positions, velocities, spring_pack, time_step)
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
        r.weightsSH[i, n_objects * 4 + n_sin_waves]       -= learning_rate * r.weightsSH.grad[i, r.n_objects * 4 + n_sin_waves]
        r.weightsSH[i, n_objects * 4 + n_sin_waves + 1]   -= learning_rate * r.weightsSH.grad[i, r.n_objects * 4 + n_sin_waves + 1] 
        
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

#TODO: Figure out a way of calculating for each robot.
# Gradients - Loss


r = Robot(springs, startingObjectPositions, max_steps)
loss = tai.field(dtype=tai.f32, shape=(), needs_grad=True) # 0-D tensor
tai.root.lazy_grad()

def run_simulation():
    
    # Create population of robots
    n_robot_population = 1
    os.system("rm population/*.txt")
    springs_population, startingObjectPositions_population = create_population(n_robot_population)  
    
    robot_drawing = []
    for robot_idx in range(n_robot_population):
        
        
        # Get objects and springs individual robots
        springs = springs_population[robot_idx]
        startingObjectPositions = startingObjectPositions_population[robot_idx]
        
        # Init Objects
        # positions, velocities, spring_forces_on_objects = init_robot_objects_ds(startingObjectPositions, vec)
        
        # Init Springs
        # spring_anchors, spring_at_rest_length, spring_actuation, spring_restoring_forces, actuation = init_robot_springs_ds(springs, vec)
        
        # Init NNs parameters
        # weightsSH, weightsHM, hidden, bias_hidden = init_robot_weights_ds(startingObjectPositions, n_hidden_neurons)
        
        # Determines center of the bot by the time_step
        # center = init_robot_center(vec)

        # Determine the goal of the robot.
        # goal = init_robot_goal(vec)
        
        # Init NNs architecture
        # Initialize_Neural_Network(n_objects, weightsSH, bias_hidden, weightsHM)
        Initialize_Neural_Network()
        Initialize()
        
        # Initialize body of the individual robot
        # Initialize(positions, goal, velocities, spring_anchors, spring_at_rest_length, spring_actuation,
            #    spring_restoring_forces, spring_forces_on_objects, hidden, actuation)
        
        # Pack it up for parameter readability 
        # spring_pack = (spring_anchors, spring_at_rest_length, spring_actuation, spring_restoring_forces, spring_forces_on_objects)
        
        with tai.Tape(loss):
        
            # Simulate
            Simulate()
            loss[None] = 0.0
            
            Compute_loss()
            
        print(loss[None])
        
        # Fine-tune the brain of the robot
        # tune_robots_brain(weightsHM, weightsSH, bias_hidden)
        tune_robots_brain()
        
        # TODO: Eventually we might want to add the option to draw for every 'survival of the fittest' run.
        
        # robot_drawing.append([positions, spring_anchors])

    # # TODO: OLD
    # # Initialize the robot.
    # Initialize()

    # # Automated Differentiation.
    # with tai.Tape(loss):
        
    #     Simulate(positions, velocities, weightsSH, weightsHM, bias_hidden, hidden, actuation, spring_pack,)
        
    #     loss[None] = 0.0
    #     Compute_loss()

    # print(loss[None])

# Run simulation
for opt_step in range(2):
    
    run_simulation()
    # robot_drawings = run_simulation()
    # positions = robot_drawings[0][0]
    # spring_anchors = robot_drawings[0][1]
    
    if opt_step == 0:
        os.system("rm images/*.png")
        Draw(0)

Draw(max_steps)

    
# os.system("rm images/*.png")
# Draw(0)

# Create the video
Create_video()