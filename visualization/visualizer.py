import trimesh

class Visualizer:
    def __init__(self, mesh):
        self.scene = trimesh.Scene()
        self.scene.add_geometry(mesh)

    def animate(self, robot, path, step=10):
        """
        step = 10  → on affiche 1 point sur 10
        """

        # Préparer la scène AVANT affichage
        for i in range(0, len(path), step):
            robot.move_to(path[i])

            trail = robot.paint_trail()
            if trail:
                self.scene.add_geometry(trail)

        self.scene.show()
