import taichi as tai

class Robot:
    
    def __init__(self, springs, startingObjectPositions, max_steps):
        """
        Initialize a new Robot instance.

        Parameters:
        - name (str): The name of the robot.
        - model (str): The model of the robot.
        - color (str): The color of the robot.
        """
        self.loss = tai.field(dtype=tai.f32, shape=(), needs_grad=True) # 0-D tensor
        
        self.springs = springs
        self.startingObjectPositions = startingObjectPositions
        self.max_steps = max_steps
        
        self.n_objects = len(self.startingObjectPositions)
        
        # Create a field w/ max_steps X n_objects entries.
        # It is stored in positions. Needs to be defined previously
        #Vector of length 2. Real Values
        real = tai.f32
        tai.init(default_fp = real) # Init TAI
        vec =  lambda: tai.Vector.field(2, dtype=real) 
        
        self.positions = vec()
        self.velocities = vec()
        self.spring_forces_on_objects = vec()
    
        # Store positions of every object at every time step.
        # Where each position is a vector of length 2. x and y.
        # positions = vec()
        tai.root.dense(tai.i, self.max_steps).dense(tai.j, self.n_objects).place(self.positions)

        # Gradients of position. Changing as a function of the loss per time step.
        tai.root.dense(tai.i, self.max_steps).dense(tai.j, self.n_objects).place(self.positions.grad)

        # velocities = vec()
        tai.root.dense(tai.i, self.max_steps).dense(tai.j, self.n_objects).place(self.velocities)
        
        # Forces acting on the objects
        # spring_forces_on_objects = vec()
        tai.root.dense(tai.i, self.max_steps).dense(tai.j, self.n_objects).place(self.spring_forces_on_objects)

        # You can add more attributes here as needed
        self.n_springs = len(self.springs)
    
        # Store as Taichi fields
        self.spring_anchor_a = tai.field(tai.i32)
        self.spring_anchor_b = tai.field(tai.i32)
        self.spring_at_rest_length = tai.field(tai.f32)
        self.spring_actuation = tai.field(tai.i32)
        
        # Capture motor value to be sent to every spring at every time_step
        self.actuation = tai.field(tai.f32)
        tai.root.dense(tai.ij, (self.max_steps, self.n_springs)).place(self.actuation)
        
        # Taichi Structure for springs. Turn Spring anchor A & B from integer into field
        tai.root.dense(tai.i, self.n_springs).place(self.spring_anchor_a, self.spring_anchor_b, self.spring_at_rest_length, self.spring_actuation) 

        # Forces of the springs
        self.spring_restoring_forces = vec()
        tai.root.dense(tai.i, self.max_steps).dense(tai.j, self.n_springs).place(self.spring_restoring_forces)
        
        self.spring_anchors = (self.spring_anchor_a, self.spring_anchor_b)
        
        self.n_sensors = 10 + 4 * self.n_objects + 2
        self.n_hidden_neurons = 32
    
        # Sensor to Hidden neurons and weights - Put weights from Sensors to hidden neurons
        self.weightsSH = tai.field(tai.f32)
        tai.root.dense(tai.ij, (self.n_hidden_neurons, self.n_sensors)).place(self.weightsSH)

        # Hidden to Motor neurons and weights
        self.weightsHM = tai.field(tai.f32)
        tai.root.dense(tai.ij, (self.n_springs, self.n_hidden_neurons)).place(self.weightsHM)
        
        # Create field for N hidden neurons at each time_step
        self.hidden = tai.field(tai.f32)
        tai.root.dense(tai.ij, [self.max_steps, self.n_hidden_neurons]).place(self.hidden)

        # Create bias. One per each hidden neuron. Total N bias
        self.bias_hidden = tai.field(tai.f32)
        tai.root.dense(tai.i, self.n_hidden_neurons).place(self.bias_hidden)
        
        
        self.center = vec()
        tai.root.dense(tai.i, self.max_steps).place(self.center)
        
        self.goal = vec()
        tai.root.place(self.goal)
        tai.root.lazy_grad()