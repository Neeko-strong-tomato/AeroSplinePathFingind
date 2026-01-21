import argparse
import numpy as np
import os

from stable_baselines3 import PPO

from env_surface_rl import SurfaceCoverageEnv
from visualize import visualize
from callbacks import TQDMCallback


def run_episode(env, model, deterministic=True):
    obs, _ = env.reset()
    done = False

    path = [env.pos.copy()]
    rotations = 0

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)

        if action in [1, 2]:
            rotations += 1

        obs, reward, done, _, _ = env.step(action)
        path.append(env.pos.copy())

    return np.array(path), rotations


import matplotlib.pyplot as plt

def moving_mean_std(data, window=20):
    data = np.array(data)

    means = []
    stds = []

    for i in range(len(data)):
        start = max(0, i - window)
        chunk = data[start:i + 1]

        means.append(chunk.mean())
        stds.append(chunk.std())

    return np.array(means), np.array(stds)


def plot_training_history(callback, window=20):
    plt.figure(figsize=(12, 6))

    for values, label in [
        (callback.losses, "Total loss"),
        (callback.value_losses, "Value loss"),
        (callback.policy_losses, "Policy loss")
    ]:
        if len(values) == 0:
            continue

        mean, std = moving_mean_std(values, window)

        x = range(len(mean))

        plt.plot(x, mean, label=label)
        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.25
        )

    plt.xlabel("Rollouts")
    plt.ylabel("Loss")
    plt.title("Historique d'entraînement PPO (moyenne ± variance)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main(args):
    surface = [
        "############....",
        "##############..",
        "##############..",
        "######.........."
    ]
    
    env = SurfaceCoverageEnv(surface_map=surface, max_steps=500)
    obs, _ = env.reset()
    print(obs["mask"].shape)

    # Chargement ou entraînement du modèle
    model_file = args.model_path + ".zip"

    if args.train or not os.path.exists(model_file):
        print("🚀 Entraînement du modèle RL...")
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=0,
            gamma=0.997,
            learning_rate=5e-5,
            n_steps=512,
            batch_size=256,
            clip_range=0.1,
            ent_coef=0.003
        )
        callback = TQDMCallback(args.timesteps)
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback
        )
        plot_training_history(callback)

        model.save(args.model_path)
        print(f"✅ Modèle sauvegardé : {model_file}")
    else:
        print(f"📦 Chargement du modèle : {model_file}")
        model = PPO.load(model_file)


    # Exécution d’un épisode
    path, rotations = run_episode(env, model)

    # Métriques
    coverage = env.covered / (env.size * env.size)

    print("\n📊 MÉTRIQUES")
    print(f"Surface couverte : {coverage:.2%}")
    print(f"Longueur du chemin : {len(path)}")
    print(f"Nombre de rotations : {rotations}")

    # Visualisation
    visualize(env.grid, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coverage Path Planning RL - 2D")

    parser.add_argument("--train", action="store_true",
                        help="Entraîner un nouveau modèle")

    parser.add_argument("--model-path", type=str, default="surface_coverage_rl",
                        help="Chemin du modèle RL")

    parser.add_argument("--size", type=int, default=20,
                        help="Taille de la surface 2D")

    parser.add_argument("--max-steps", type=int, default=500,
                        help="Nombre max de pas par épisode")

    parser.add_argument("--timesteps", type=int, default=300_000,
                        help="Nombre de pas d'entraînement")

    args = parser.parse_args()
    main(args)
