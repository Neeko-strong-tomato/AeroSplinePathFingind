import matplotlib.pyplot as plt
import numpy as np

def visualize(grid, path):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="Greys", origin="upper")

    plt.plot(
        path[:, 1],
        path[:, 0],
        linestyle="--",
        marker="o",
        color="red",
        markersize=3
    )

    plt.title("Parcours du robot (RL)")
    plt.grid(True)
    plt.show()
