import numpy as np
import trimesh
from stable_baselines3 import PPO

from env.mesh_env import MeshEnv
from mesh.mesh_loader import load_mesh


def evaluate_model(mesh_path, model_path, max_steps=10_000):
    mesh = load_mesh(mesh_path)
    env = MeshEnv(mesh)

    model = PPO.load(model_path)

    obs = env.reset()
    done = False
    steps = 0

    visited_faces = set()

    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        visited_faces.add(env.current_face)
        steps += 1

    coverage = len(visited_faces) / len(mesh.faces)

    metrics = {
        "coverage_ratio": coverage,
        "steps": steps,
        "visited_faces": len(visited_faces),
        "total_faces": len(mesh.faces),
    }

    return metrics


if __name__ == "__main__":
    metrics = evaluate_model(
        mesh_path="out_fan_segments/fan_seg_000.stl",
        model_path="models/fan/fan_seg_000_ppo.zip"
    )

    print("=== EVALUATION RL ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
