from mesh_env import MeshEnvironment
from pathfinding.rl_agent import QAgent
from pathfinding.rl_env import UVMapRLEnv
from robot import Robot
from visualizer import Visualizer
import numpy as np
from uv_map import create_point_cloud
# Charger mesh
env = MeshEnvironment("./dome.stl")

# Récupérer la UV map 2D
face_uv_centers, _ = env.generate_uv_map()

# Position de départ en UV
start_uv = np.array([0.5, 0.5])

# Initialiser Q-Learning agent et environment UV
q_agent = QAgent(alpha=0.1, gamma=0.9, epsilon=0.2)  
uv_env = UVMapRLEnv(env, coverage_threshold=0.9, cell_size=0.02)

best_episode_reward = -float('inf')
best_episode_path = []

# Entraîner le Q-agent
print("🎓 Entraînement du Q-agent sur la UV map...")
num_episodes = 100
for episode in range(num_episodes):
    state = uv_env.reset(start_uv)
    done = False
    episode_reward = 0
    episode_path = [state.copy()]
    
    steps = 0
    while not done and steps < 200:
        actions = uv_env.actions(state)
        if not actions:
            break
        
        action_idx = q_agent.choose_action(str(state), list(range(len(actions))))
        action = actions[action_idx]
        
        next_state, reward, done = uv_env.step(action)
        episode_reward += reward
        episode_path.append(next_state.copy())
        
        next_actions = uv_env.actions(next_state)
        q_agent.update(str(state), action_idx, reward, str(next_state), list(range(len(next_actions))))
        
        state = next_state.copy()
        steps += 1
    
    coverage_pct = uv_env._get_coverage_ratio() * 100
    print(f"Episode {episode+1:2d}/{num_episodes}: Reward={episode_reward:7.1f}, Coverage={coverage_pct:5.1f}%, Steps={steps:3d}")
    
    if episode_reward > best_episode_reward:
        best_episode_reward = episode_reward
        best_episode_path = episode_path
        print(f"           ✓ Nouveau meilleur")

print(f"\n{'='*60}")
print(f"Entraînement terminé - Meilleur reward: {best_episode_reward:.2f}")
print(f"{'='*60}\n")

# Générer chemin greedy final
print("🎬 Génération du chemin GREEDY...")
uv_env_final = UVMapRLEnv(env, coverage_threshold=0.9, cell_size=0.02)
uv_path = [start_uv.copy()]
state = start_uv.copy()

for iteration in range(1000):
    coverage_pct = uv_env_final._get_coverage_ratio() * 100
    
    if coverage_pct >= 90:
        break
    
    actions = uv_env_final.actions(state)
    if not actions:
        break
    
    q_values = np.array([q_agent.q.get((str(state), i), 0.0) for i in range(len(actions))])
    
    if np.all(q_values == 0):
        action_idx = np.random.randint(0, len(actions))
    else:
        action_idx = np.argmax(q_values)
    
    state = actions[action_idx].copy()
    uv_path.append(state)
    
    cell = uv_env_final._uv_to_grid_cell(state)
    uv_env_final.visited_cells.add(cell)
    
    if iteration % 50 == 0:
        print(f"  Itération {iteration:3d}: Couverture={coverage_pct:5.1f}%")

final_coverage = uv_env_final._get_coverage_ratio() * 100
print(f"\n✅ Chemin généré: {len(uv_path)} points UV (couverture: {final_coverage:.1f}%)\n")

# Convertir chemin UV en positions 3D
print("🔄 Conversion UV → 3D...")
path_3d = []
for uv_pos in uv_path:
    distances = np.linalg.norm(face_uv_centers - uv_pos, axis=1)
    closest_face = np.argmin(distances)
    
    face_center_3d = env.mesh.triangles_center[closest_face]
    path_3d.append(face_center_3d)

print(f"✅ Chemin 3D créé avec {len(path_3d)} points\n")

# Robot et visualisation
robot = Robot()

print("🎨 Affichage de la scène...")
viz = Visualizer(env.mesh)
viz.add_path(path_3d)
viz.add_robot(robot)
viz.animate(robot, path_3d, delay=0.02)
