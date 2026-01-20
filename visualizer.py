import trimesh
import time

class Visualizer:
    def __init__(self, mesh):
        self.scene = trimesh.Scene(mesh)

    def add_path(self, points):
        path = trimesh.load_path(points)
        self.scene.add_geometry(path)

    def add_robot(self, robot):
        self.scene.add_geometry(robot.body)

    def animate(self, robot, path, delay=0.1):
        for p in path:
            robot.move_to(p)
            trail = robot.paint_trail()
            if trail:
                self.scene.add_geometry(trail)

            self.scene.show()
            time.sleep(delay)
