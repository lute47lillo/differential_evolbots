
import taichi as tai
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# -------------------------------------------------------------
# move the object to max use the taichi's differentiable routine.
# TODO: Taichi's Fields documentation !

max_steps = 100
ground_height = 0.1
dt = 0.01 # Amount of time that elapses between time steps.
gravity = -9.8

# Objects connected by Springs
startingObjectPositions = []
n_objects = 4

for object_idx in range(n_objects):
    startingObjectPositions.append([np.random.random(), np.random.random()*0.9])

# -------------------------------------------------------------


# Create a field w/ max_steps X n_objects entries.
# It is stored in positions. Needs to be defined previously
#Vector of length 2. Real Values
real = tai.f32
tai.init(default_fp = real) # Init TAI
vec =  lambda: tai.Vector.field(2, dtype=real) 


positions = vec()
tai.root.dense(tai.i, max_steps).dense(tai.j, n_objects).place(positions)

# Gradients of position. Changing as a function of the loss per time step.
tai.root.dense(tai.i, max_steps).dense(tai.j, n_objects).place(positions.grad)


velocities = vec()
tai.root.dense(tai.i, max_steps).dense(tai.j, n_objects).place(velocities)


loss = tai.field(dtype=tai.f32, shape=(), needs_grad = True) # 0-D tensor

# -------------------------------------------------------------
# Execute by Taichi and not python by using decorater
@tai.kernel
def Compute_loss():
    
    # Focus on position of the objects to determine loss fn. Arbitrary choice
    # Second component of zeroth object. Loss = Height of 0th objects at last time_step
    loss[None] = positions[max_steps-1, 0][1]
    
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
                
        tai_gui.show(f"images/test_{frame_offset+time_step}.png")

# -------------------------------------------------------------
def Initialize():
    
    # Initialize the position for each object
    for object_idx in range(n_objects):
        # Temp, just propagate positions.
        positions[0, object_idx] = startingObjectPositions[object_idx]
        
        # Set initial velocites
        velocities[0, object_idx] = [0,-0.1]


# -------------------------------------------------------------
def Simulate():
    
    for time_step in range(1, max_steps):
        
        # Update position of object.
        step_one(time_step)

# -------------------------------------------------------------
@tai.kernel
def step_one(time_step: tai.i32):
    
    for object_idx in range(n_objects):
        
        # Get old position and velocity
        old_pos = positions[time_step-1, object_idx]
        old_velocity = velocities[time_step-1, object_idx] + dt * gravity  * tai.Vector([0,1]) # Change velocity as fn of gravity by dt
        
        # Detect collisions
        if old_pos[1] <= ground_height:
            
            old_velocity = tai.Vector([0,0])
        
        # Update position and velocity
        new_pos = old_pos + dt * old_velocity
        positions[time_step, object_idx] = new_pos
        
        new_velocity = old_velocity
        velocities[time_step, object_idx] = new_velocity
            
# -------------------------------------------------------------
# Create Video
def Create_video():
    os.system("rm movie.p4")
    os.system(" ffmpeg -i images/test_%d.png movie.mp4")


# -------------------------------------------------------------
Initialize()

# Automated Differentiation.
with tai.ad.Tape(loss):
    
    Simulate()
    
    # In our case dLoss / dPosition
    Compute_loss()
    
os.system("rm images/*.png")
Draw(0)


# Based on our loss. We have dLoss/dPosition, which is positions.grad[0,0][1]
# Update in the opposite direction of the gradient to minimize the loss
startingObjectPositions[0] += 0.1 * positions.grad[0,0]

Initialize()
Simulate()
Draw(max_steps)
Create_video()


