import os
import taichi as tai
import utils
import simulate as sim
import numpy as np
from robot import Robot

# Initialize taichi
tai.init()
print(tai.__version__)

# TODO: Move global variables to an independent file
max_steps = 200
ground_height = 0.1

def draw_fittest_robot(r, img_dir, frame_offset, robot_index):
        
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

        if not os.path.exists(f"{img_dir}/robot_{robot_index}"):
            os.makedirs(f"{img_dir}/robot_{robot_index}")  
               
        tai_gui.show(f"{img_dir}/robot_{robot_index}/image_{frame_offset+time_step}.png")

@tai.kernel
def compute_loss_baseline():
    
    loss[None] -= (1.2 * (r.positions[r.max_steps-1, 0][0]) - 0.1) \
                + (1.2 * (r.positions[r.max_steps-1, 1][0]) - 0.3) \
                + (0.7 * (r.goal[None][0] - r.positions[r.max_steps-1, 0][0])) \
                + (0.7 * (r.goal[None][0] - r.positions[r.max_steps-1, 1][0]))


"""
    Init Simulation
"""
if __name__ == "__main__":
    
    # Parse arguments
    n_pop, n_opt, name_experiment, variant_type = utils.parse_args_baseline()
    
    # Delete images from older run
    os.system("rm -rf img_random/*")
    os.system("rm -rf img_test/*")
    os.system(f"rm stats/baseline_loss_{variant_type}.txt")
    
    # Baseline simulation for test co-evolution or random Variant B for A/B testing.
    if variant_type == "random":
        # A/B Testing
        springs, startingObjectPositions = sim.simulate_robot(0, variant_type)
        image_dir = 'img_random'
        type_video = 'random'
    else:
        # Read springs and objects
        springs, startingObjectPositions = utils.read_objects_springs_fit_robot(0)
        image_dir = 'img_test'
        type_video = 'test'

    # Create Robot
    r = Robot(springs, startingObjectPositions, max_steps)
        
    # Create loss for that robot
    loss = tai.field(dtype=tai.f32, shape=(), needs_grad=True) 
    tai.root.lazy_grad() 

    # Track loss
    loss_tracker = []

    # Define optimization steps based on original simulation
    baseline_opt_steps = (n_pop - 1) * n_opt // 2
    
    for opt_step in range(n_pop):        
            
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
            
        print(f"Robot {0} - Opt Step {opt_step}. Loss: {r.loss[None]}")
        
        # Fine-tune the brain of the robot.
        _, _, _, _ = sim.tune_robots_brain(r)
        
        # Draw first. Initial drawing will be fittest morphology from main simulation.
        if opt_step == 0:
            os.system(f"rm {image_dir}/robot_0/*.png")
            draw_fittest_robot(r, image_dir, 0, 0)
        
        # TODO: Wathc out for modifications
        #if opt_step % n_opt == 0:
    
        # Save the loss
        loss_tracker.append(r.loss[None])
        
        if opt_step == (baseline_opt_steps - 1):
            loss_tracker.append(r.loss[None])
            
    # Save the loss tracker to analyze and plot.    
    utils.baseline_stat_loss_save(loss_tracker, variant_type)

    # Draw and crate video of the baseline - fittest robot
    draw_fittest_robot(r, image_dir, 200, 0)
    utils.create_video(f"{variant_type}_{name_experiment}_{n_pop}r_{n_opt}o", type_video)