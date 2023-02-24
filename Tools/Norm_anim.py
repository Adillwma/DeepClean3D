import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the original number and normalization range
num = 5
min_val = 0
max_val = 10

# Define the normalization function
def normalize(x, a, b):
    return (x - min_val) / (max_val - min_val) * (b - a) + a

# Define the plot
fig, ax = plt.subplots()
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
line, = ax.plot([], [], lw=2)

# Define the animation function
def update(frame):
    # Calculate the normalized value at the current frame
    norm_val = normalize(num, min_val, max_val * (frame + 1) / 100)
    # Update the plot
    line.set_data([0, norm_val], [0, 1])
    return line,

# Create the animation
anim = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# Show the animation
plt.show()