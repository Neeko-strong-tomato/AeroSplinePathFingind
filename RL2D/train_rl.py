from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env_surface_rl import SurfaceCoverageEnv

env = make_vec_env(lambda: SurfaceCoverageEnv(size=20), n_envs=4)

model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    gamma=0.99,
    n_steps=256,
    batch_size=128
)

model.learn(total_timesteps=300_000)
model.save("surface_coverage_rl")