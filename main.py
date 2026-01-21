from mesh_env import MeshEnvironment
from visualizer import Visualizer
from robot import Robot
from config import *
import numpy as np

from coverage.segmentation import segment_by_normals
from coverage.zigzag import zigzag_on_region
from coverage.mesh_rl_env import MeshCoverageRLEnv
from coverage.dqn_agent import DQNAgent
from coverage.rl_trainer import RLTrainer

print("\n==============================")
print("🧠 MODE :", PLANNING_MODE.value.upper())
print("==============================\n")

# Charger mesh
env = MeshEnvironment("modele.stl")
mesh = env.mesh

if PLANNING_MODE == PlanningMode.COVERAGE:
    # ========== MODE COVERAGE (CLASSIQUE) ==========
    print("📍 Mode classique: Zigzag par région\n")
    
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
    viz.animate(robot, global_path)

elif PLANNING_MODE == PlanningMode.RL:
    # ========== MODE RL (APPRENTISSAGE) ==========
    print("🤖 Mode RL: Apprentissage par renforcement (DQN)\n")
    
    import torch
    
    # Initialiser trainer
    trainer = RLTrainer()
    trainer.initialize_agent()
    
    # Entraîner sur le mesh actuel
    print(f"📊 Entraînement sur {RL_EPISODES} épisodes...\n")
    history = trainer.train(n_episodes=RL_EPISODES)
    
    # Évaluer l'agent entraîné
    print("\n📈 Évaluation finale...")
    eval_stats = trainer.evaluate(mesh=mesh, n_runs=3)
    
    print(f"Coverage moyen: {eval_stats['coverage_mean']:.1f}% ± {eval_stats['coverage_std']:.1f}%")
    print(f"Path length moyen: {eval_stats['path_length_mean']:.2f}")
    print(f"Steps moyen: {eval_stats['steps_mean']:.0f}")
    
    # Générer trajet final avec agent entraîné
    print("\n🎬 Génération trajet avec agent entraîné...")
    
    rl_env = MeshCoverageRLEnv(mesh, max_steps=5000)
    state = rl_env.reset()
    done = False
    
    while not done:
        action = trainer.agent.choose_action(state, training=False)
        next_state, reward, done, info = rl_env.step(action)
        state = next_state
    
    # Convertir trajet en positions 3D
    global_path = np.array(rl_env.visited_sequence)
    
    print(f"🎨 Points de trajectoire générés : {len(global_path)}")
    print(f"Coverage final: {rl_env.get_coverage_percentage():.1f}%")
    print(f"Path length: {rl_env.get_path_length():.2f}")
    
    # Robot & visualisation
    robot = Robot()
    viz = Visualizer(mesh)
    viz.animate(robot, global_path)
