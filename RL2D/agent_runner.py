import numpy as np
from stable_baselines3 import PPO
from env_surface_rl import SurfaceCoverageEnv

env = SurfaceCoverageEnv(size=20)
model = PPO.load("surface_coverage_rl")

obs, _ = env.reset()
positions = [env.pos.copy()]

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    positions.append(env.pos.copy())

positions = np.array(positions)
