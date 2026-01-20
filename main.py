import random

from mesh_env import MeshEnvironment
from visualizer import Visualizer
from robot import Robot
from config import *

from coverage.segmentation import segment_by_normals
from coverage.rl_agent import QAgent
from coverage.rl_env import MeshRLEnv


print("\n==============================")
print("MODE : IA (RL Coverage Path Planning)")
print("==============================\n")

# --------------------------------------------------
# 1. Chargement du mesh (résolution réduite si fallback)
# --------------------------------------------------
env_mesh = MeshEnvironment("modele.stl")
mesh = env_mesh.mesh
graph = env_mesh.graph

# --------------------------------------------------
# 2. Segmentation du mesh
# --------------------------------------------------
regions = segment_by_normals(
    mesh,
    angle_threshold_deg=ANGLE_THRESHOLD_DEG
)

# --------------------------------------------------
# 3. Agent RL global
# --------------------------------------------------
agent = QAgent(
    alpha=0.1,
    gamma=0.9,
    epsilon=0.3
)

global_path = []

# --------------------------------------------------
# 4. Coverage Path Planning par région (RL borné)
# --------------------------------------------------
for region_id, region_faces in enumerate(regions):

    print(f"Région {region_id} | {len(region_faces)} faces")

    # ⚠️ ignorer les régions trop petites
    if len(region_faces) < 5:
        continue

    rl_env = MeshRLEnv(
        graph=graph,
        faces_in_region=region_faces
    )

    start_face = random.choice(region_faces)
    state = rl_env.reset(start_face)

    face_path = [start_face]
    done = False

    max_steps = min(len(region_faces) * 3, 300)
    steps = 0

    while not done and steps < max_steps:

        actions = rl_env.actions()
        if not actions:
            break

        action = agent.choose_action(state, actions)
        next_state, reward, done = rl_env.step(action)

        agent.update(
            state,
            action,
            reward,
            next_state,
            rl_env.actions()
        )

        state = next_state
        face_path.append(action)
        steps += 1

    # réduire l'exploration au fil des régions
    if region_id > 5:
        agent.epsilon = 0.05

    # --------------------------------------------------
    # 5. Conversion faces → points 3D
    # --------------------------------------------------
    for face_id in face_path:
        global_path.append(mesh.triangles_center[face_id])


print(f"\nPoints de trajectoire générés : {len(global_path)}")

# --------------------------------------------------
# 6. Visualisation
# --------------------------------------------------
robot = Robot()
viz = Visualizer(mesh)
viz.animate(robot, global_path)
