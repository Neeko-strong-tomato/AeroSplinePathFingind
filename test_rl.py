#!/usr/bin/env python3
"""Test simple du système RL"""

import numpy as np
import trimesh
from coverage.mesh_rl_env import MeshCoverageRLEnv
from coverage.dqn_agent import DQNAgent
from coverage.mesh_generator import ProceduralMeshGenerator

print("🔧 Initialisation...")

# Générer un mesh simple
mesh = ProceduralMeshGenerator.simple_sphere(subdivisions=2)

print(f"📊 Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

# Créer l'env
env = MeshCoverageRLEnv(mesh, coverage_radius=0.15, max_steps=500)

# Test reset
state = env.reset()
print(f"🎮 État dim: {len(state)}")
print(f"📍 Action space: 8 directions discrètes")

# Créer l'agent
agent = DQNAgent(state_dim=len(state), action_dim=8)

print("✅ Système RL prêt!")
print("\n🚀 Test d'un pas:")

# Test step
action = 0  # direction 0
state_next, reward, done, info = env.step(action)
print(f"   Récompense: {reward:.3f}")
print(f"   Couverture: {(env.coverage_map.sum() / env.n_faces)*100:.1f}%")
print(f"   Étapes: {env.step_count}/{env.max_steps}")

print("\n✨ Tous les composants fonctionnent!")
print("   Pour un entraînement complet: python coverage/rl_trainer.py")
