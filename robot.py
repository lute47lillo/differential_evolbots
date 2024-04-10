import taichi as tai

class Robot:
    
    def __init__(self, springs, startingObjectPositions, max_steps):
        """
            Definition
            -----------
                Initialize a new Robot instance.

            Parameters
            -----------
                - springs (list): The springs and their information for the robot.
                - startingObjectPositions (list): The starting positions of the objects of the robot.
                - max_steps (int): Maximum number of steps for simulation of the robot.
                
            Returns
            -----------
                None
        """
        
        # Create a field w/ max_steps X n_objects entries. It is stored in positions. Needs to be defined previously
        real = tai.f32
        tai.init(default_fp = real) # Init TAI
        self.vec =  lambda: tai.Vector.field(2, dtype=real)  # Vector of length 2. Real Values
        self.loss = tai.field(dtype=tai.f32, shape=()) # 0-D tensor
        
        # -----------------------------------------------------------------
        
        # Get common attributes -> Springs, objects and max_steps
        self.springs = springs
        self.startingObjectPositions = startingObjectPositions
        self.max_steps = max_steps
        
        # Get number of .. for X robot
        self.n_objects = len(self.startingObjectPositions)
        self.n_springs = len(self.springs)
        
        # -----------------------------------------------------------------
        
        # Create vectors for positions, velocity, and other physical attributes
        self.positions = self.vec()
        self.velocities = self.vec()
        self.spring_forces_on_objects = self.vec()
    
        # Initialize them
        self.init_robot_objects_ds()

        # -----------------------------------------------------------------
        
        # Store as Taichi fields Spring physical attributes
        self.spring_anchor_a = tai.field(tai.i32)
        self.spring_anchor_b = tai.field(tai.i32)
        self.spring_at_rest_length = tai.field(tai.f32)
        self.spring_actuation = tai.field(tai.i32)
        self.spring_restoring_forces = self.vec()
        self.actuation = tai.field(tai.f32)
        
        # Initialize them
        self.init_robot_springs_ds()
        
        # -----------------------------------------------------------------
        
        # Create Neural Network attributes
        self.n_sensors = 10 + 4 * self.n_objects + 2
        self.n_hidden_neurons = 32
        self.weightsSH = tai.field(tai.f32)
        self.weightsHM = tai.field(tai.f32)
        self.hidden = tai.field(tai.f32)
        self.bias_hidden = tai.field(tai.f32)
    
        # Initialize them
        self.init_robot_weights_ds()
        
        # -----------------------------------------------------------------
        
        # Design center of mass of robot and goal
        self.center = self.vec()
        self.goal = self.vec()
        
        # Initialize
        self.init_robot_center()
        self.init_robot_goal()
        
        # -----------------------------------------------------------------
        
        # Set fields to requiere Grads
        tai.root.lazy_grad()
    
        
    def init_robot_springs_ds(self):
        
        # Capture motor value to be sent to every spring at every time_step
        tai.root.dense(tai.ij, (self.max_steps, self.n_springs)).place(self.actuation)
        
        # Taichi Structure for springs. Turn Spring anchor A & B from integer into field
        tai.root.dense(tai.i, self.n_springs).place(self.spring_anchor_a, self.spring_anchor_b, self.spring_at_rest_length, self.spring_actuation) 

        # Forces of the springs
        tai.root.dense(tai.i, self.max_steps).dense(tai.j, self.n_springs).place(self.spring_restoring_forces)

    
    def init_robot_objects_ds(self):
        
        # Store positions of every object at every time step.
        # Where each position is a vector of length 2. x and y.
        tai.root.dense(tai.i, self.max_steps).dense(tai.j, self.n_objects).place(self.positions)

        # Gradients of position. Changing as a function of the loss per time step.
        tai.root.dense(tai.i, self.max_steps).dense(tai.j, self.n_objects).place(self.positions.grad)

        tai.root.dense(tai.i, self.max_steps).dense(tai.j, self.n_objects).place(self.velocities)
        
        # Forces acting on the objects
        tai.root.dense(tai.i, self.max_steps).dense(tai.j, self.n_objects).place(self.spring_forces_on_objects)
    
    def init_robot_weights_ds(self):
        
        # Sensor to Hidden neurons and weights - Put weights from Sensors to hidden neurons
        tai.root.dense(tai.ij, (self.n_hidden_neurons, self.n_sensors)).place(self.weightsSH)

        # Hidden to Motor neurons and weights
        tai.root.dense(tai.ij, (self.n_springs, self.n_hidden_neurons)).place(self.weightsHM)
        
        # Create field for N hidden neurons at each time_step
        tai.root.dense(tai.ij, [self.max_steps, self.n_hidden_neurons]).place(self.hidden)

        # Create bias. One per each hidden neuron. Total N bias
        tai.root.dense(tai.i, self.n_hidden_neurons).place(self.bias_hidden)
    
    def init_robot_goal(self):
    
        tai.root.place(self.goal)

    # TODO: Adapt based on the robot task
    def init_robot_center(self):
        
        tai.root.dense(tai.i, self.max_steps).place(self.center)
