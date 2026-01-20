from mesh_env import MeshEnvironment
from visualizer import Visualizer
from robot import Robot
from config import PATHFINDING_MODE, PathFindingMode

from pathfinding.astar import astar_path
from pathfinding.rl_env import MeshRLEnv
from pathfinding.rl_agent import QAgent

# =====================
# Initialisation
# =====================

env = MeshEnvironment("modele.stl")

start_face = 0
goal_face = min(50, len(env.graph.nodes) - 1)

print("\n==============================")
print("🚀 DÉMARRAGE DU PATHFINDING")
print(f"🧠 Mode sélectionné : {PATHFINDING_MODE.value.upper()}")
print("==============================\n")

# =====================
# CHOIX DE L'ALGO
# =====================

if PATHFINDING_MODE == PathFindingMode.ASTAR:
    print("➡️ Utilisation de l'algorithme A*")

    faces_path = astar_path(
        env.graph,
        start_face,
        goal_face
    )

elif PATHFINDING_MODE == PathFindingMode.RL:
    print("➡️ Utilisation du PathFinding par Renforcement (RL)")

    rl_env = MeshRLEnv(env.graph, goal_face)
    agent = QAgent()

    # Entraînement simple
    for episode in range(200):
        state = rl_env.reset(start_face)
        done = False

        while not done:
            actions = rl_env.actions(state)
            action = agent.choose_action(state, actions)

            next_state, reward, done = rl_env.step(action)
            agent.update(
                state,
                action,
                reward,
                next_state,
                rl_env.actions(next_state)
            )
            state = next_state

    # Exécution finale
    faces_path = [start_face]
    state = start_face
    while state != goal_face:
        action = agent.choose_action(state, rl_env.actions(state))
        faces_path.append(action)
        state = action

else:
    raise ValueError("Mode de pathfinding inconnu")

# =====================
# Conversion faces → points
# =====================

path_points = [env.face_center(f) for f in faces_path]

# =====================
# Visualisation
# =====================

robot = Robot()
viz = Visualizer(env.mesh)

viz.add_path(path_points)
viz.add_robot(robot)
viz.animate(robot, path_points)
