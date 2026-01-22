import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from visualization.colors import path_colors


class MeshPlotter:
    def __init__(self, mesh):
        self.mesh = mesh

    def plot(self, robot=None, show_path=True):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Mesh
        faces = self.mesh.vertices[self.mesh.faces]
        mesh_collection = Poly3DCollection(
            faces,
            facecolor=(0.8, 0.8, 0.8, 0.3),
            edgecolor="k",
            linewidth=0.1,
        )
        ax.add_collection3d(mesh_collection)

        # Robot path
        if robot and show_path:
            centers = self.mesh.triangles_center[robot.path]
            colors = path_colors(len(centers))

            for i in range(len(centers) - 1):
                ax.plot(
                    centers[i : i + 2, 0],
                    centers[i : i + 2, 1],
                    centers[i : i + 2, 2],
                    color=colors[i],
                    linewidth=2,
                )

            ax.scatter(
                centers[-1, 0],
                centers[-1, 1],
                centers[-1, 2],
                color="red",
                s=40,
                label="Robot"
            )

        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
        plt.legend()
        plt.show()
