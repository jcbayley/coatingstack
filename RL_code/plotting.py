import matplotlib.pyplot as plt
import numpy as np

def plot_coating(state, fname=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid(True)
    depth_so_far = 0  # To keep track of where to plot the next bar
    colors = ["C0", "C1", "C2", "C3", "C4"]
    for i in range(len(state)):
        material_idx = np.argmax(state[i][1:]) 
        thickness = state[i][0]
        ax.bar(depth_so_far + thickness / 2, thickness, 
                width=thickness, 
                color=colors[material_idx])
        depth_so_far += thickness

    ax.set_xlim([0, depth_so_far * 1.01])
    ax.set_ylabel('Physical Thickness [nm]')
    ax.set_xlabel('Layer Position')
    ax.set_title('Generated Stack')

    if fname is not None:
        fig.savefig(fname)