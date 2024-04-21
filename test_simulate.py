import os
import shutil
from robot import Robot
import taichi as tai
import utils
import simulate as sim
import numpy as np
tai.init()

max_steps = 200
ground_height = 0.1

def draw_fittest_robot(r, frame_offset, robot_index):
        
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

        if not os.path.exists(f"img_base/robot_{robot_index}"):
            os.makedirs(f"img_base/robot_{robot_index}")  
               
        tai_gui.show(f"img_base/robot_{robot_index}/image_{frame_offset+time_step}.png")
        
def copy_init_robot_files():
    """
        Definition
        -----------
            Copy images of the fittest robot at population init. to new directory.

    """
    # Define the source and destination directories
    src_dir = 'images/robot_0'
    dst_dir = 'img_base/robot_0'

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Get a list of files in the source directory
    files = os.listdir(src_dir)
    
    # Copy the first 200 files to the destination directory
    for file in files:
        filename, file_extension = os.path.splitext(file)
        if filename.startswith('image_') and 200 <= int(filename.split('_')[1]) <= 399:
            new_filename = 'image_' + str(int(filename.split('_')[1]) - 200)
            new_file = new_filename + file_extension
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, new_file)
            shutil.copyfile(src_file, dst_file)
     
def read_objects_springs_fit_robot(robot_index):
    
    with open(f"population/robot_{robot_index}.txt", 'r') as file:
            
            all_lines = file.readlines()
            
            # Get object positions
            start_end_obj = eval(all_lines[0])
            
            # Get Springs
            spring_obj = all_lines[1:]
            _, spring_obj = utils.get_last_obj_index(spring_obj)

    return spring_obj, start_end_obj

@tai.kernel
def compute_loss_baseline():
    
    loss[None] -= (1.2 * (r.positions[r.max_steps-1, 0][0]) - 0.1) \
                + (1.2 * (r.positions[r.max_steps-1, 1][0]) - 0.3) \
                + (0.7 * (r.goal[None][0] - r.positions[r.max_steps-1, 0][0])) \
                + (0.7 * (r.goal[None][0] - r.positions[r.max_steps-1, 1][0]))

# TODO: Check with the other loss (using -)
"""
    Init Simulation
"""
# TODO I do not think this is what you want to do
os.system("rm -rf img_base/*")
# copy_init_robot_files()

# Read springs and objects
springs, startingObjectPositions = read_objects_springs_fit_robot(0)

# Create Robot
r = Robot(springs, startingObjectPositions, max_steps)
    
# Create loss for that robot
loss = tai.field(dtype=tai.f32, shape=(), needs_grad=True) 
tai.root.lazy_grad() 

for opt_step in range(140):        
        
    if opt_step == 0:
        sim.Initialize_Neural_Network(r)
        
    # Init Robot
    sim.Initialize(r)
    
    with tai.Tape(loss):
    
        # Simulate
        sim.Simulate(r)
        loss[None] = 0.0
        
        # sim.Compute_loss()
        compute_loss_baseline()
    
    # TODO: Find the solution to some robots having nan loss
    if loss[None] < r.loss[None]:
        if np.isnan(loss[None]):
            r.loss[None] = 100000
        else:
            r.loss[None] = float(loss[None])
        
    print(f"Robot {0} - Opt Step {opt_step}. Loss: {loss[None]}")
    
    # Fine-tune the brain of the robot
    prev_w_SH, prev_w_HM, prev_w_hidden, prev_w_bias_hidden = sim.tune_robots_brain(r)
    
    if opt_step == 0:
        os.system(f"rm img_base/robot_0/*.png")
        draw_fittest_robot(r, 0, 0)
    
    if opt_step % 10 == 0:
        # TODO: save the weights if better loss
        print("WIP")
        
        # TODO: Save the fitness loss periodically, to compare later. So Ideally there will be 15 losses saved
        # sim.save_fitness_losses(0)
        
    

draw_fittest_robot(r, 200, 0)
utils.create_video("RE_7", "re")
