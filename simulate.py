
import taichi as tai
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Objects connected by Springs
startingObjectPositions = []
ground_height = 0.1

startingObjectPositions.append([0.1, 0.2]) # Append one object. 0.1 to right from origin [0, 0] and 0.2 above ground


# Draw the robot using Taichi's built-iGUI. (x,y) size of window
tai_gui = tai.GUI("Robot", (512, 512), background_color=0xFFFFFF, show_gui=False)

# Draw the floow
tai_gui.line(begin=(0, ground_height), end=(1, ground_height), color=0x0, radius=3)

# Draw the object
for object in startingObjectPositions:
    x = object[0]
    y = object[1]
    tai_gui.circle((x,y), color=0x0, radius=7)
               
               
tai_gui.show("scene.png")


# Better showing of the GUI image.
fig = plt.imshow(mpimg.imread("scene.png"))


