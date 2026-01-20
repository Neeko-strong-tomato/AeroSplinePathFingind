from mesh_env import MeshEnvironment
from visualizer import Visualizer
from robot import Robot
from config import *

from coverage.segmentation import segment_by_normals
from coverage.zigzag import zigzag_on_region

print("\n==============================")
print("🧠 MODE :", PLANNING_MODE.value.upper())
print("==============================\n")

# Charger mesh
env = MeshEnvironment("modele.stl")
mesh = env.mesh

# Segmentation
regions = segment_by_normals(
    mesh,
    angle_threshold_deg=ANGLE_THRESHOLD_DEG
)

# Coverage path global
global_path = []

for region in regions:
    path = zigzag_on_region(
        mesh,
        region,
        step=ZIGZAG_STEP
    )
    global_path.extend(path)

print(f"🎨 Points de trajectoire générés : {len(global_path)}")

# Robot & visualisation
robot = Robot()
viz = Visualizer(mesh)

robot = Robot()
viz = Visualizer(mesh)

viz.animate(robot, global_path)
