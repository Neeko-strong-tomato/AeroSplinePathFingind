from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


class TQDMCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.pbar = None

        # Historique
        self.losses = []
        self.value_losses = []
        self.policy_losses = []

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training PPO")

    def _on_step(self) -> bool:
        self.pbar.update(self.locals["n_steps"])
        return True

    def _on_rollout_end(self):
        logger = self.model.logger.name_to_value

        if "train/loss" in logger:
            self.losses.append(logger["train/loss"])

        if "train/value_loss" in logger:
            self.value_losses.append(logger["train/value_loss"])

        if "train/policy_gradient_loss" in logger:
            self.policy_losses.append(logger["train/policy_gradient_loss"])

    def _on_training_end(self):
        self.pbar.close()
