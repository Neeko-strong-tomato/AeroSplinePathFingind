import os
import gym
import trimesh
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env.mesh_env import MeshEnv
from mesh.mesh_loader import load_mesh


def train_on_segment(mesh_path, model_path, timesteps=200_000):
    print(f"[RL] Training on segment: {mesh_path}")

    mesh = load_mesh(mesh_path)
    env = MeshEnv(mesh)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=timesteps)
    model.save(model_path)

    print(f"[RL] Model saved to {model_path}")


def train_on_folder(segments_dir, output_dir, timesteps=200_000):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(segments_dir):
        if not fname.endswith(".stl"):
            continue

        seg_path = os.path.join(segments_dir, fname)
        model_name = fname.replace(".stl", "_ppo")
        model_path = os.path.join(output_dir, model_name)

        train_on_segment(seg_path, model_path, timesteps)


if __name__ == "__main__":
    # Exemple
    train_on_folder(
        segments_dir="out_fan_segments",
        output_dir="models/fan",
        timesteps=150_000
    )
