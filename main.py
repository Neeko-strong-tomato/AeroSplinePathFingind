from mesh_env import MeshEnvironment
from pathfinding.astar import astar_path
from robot import Robot
from visualizer import Visualizer

# Charger mesh
env = MeshEnvironment("./3d_models/fan_uv_mapped.obj")

start_face = 2
goal_face = 2

# Pathfinding classique
faces_path = astar_path(env.graph, start_face, goal_face)

# Convertir faces → positions 3D
path_points = [env.face_center(f) for f in faces_path]

# Robot
robot = Robot()

# Visualisation
viz = Visualizer(env.mesh)
viz.add_path(path_points)
viz.scene.show()
