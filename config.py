from enum import Enum

class PlanningMode(Enum):
    COVERAGE = "coverage"
    RL = "rl" 

PLANNING_MODE = PlanningMode.COVERAGE  # Changer à RL pour mode apprentissage

# ========== COVERAGE PARAMETERS ==========
ANGLE_THRESHOLD_DEG = 30
ZIGZAG_STEP = 0.05

# ========== RL PARAMETERS ==========
RL_EPISODES = 20  # Nombre d'épisodes d'entraînement

# DQN Configuration
RL_CONFIG = {
    'state_dim': 8,              # position(3) + orientation(3) + coverage(1) + steps(1)
    'action_dim': 8,             # 8 directions
    'learning_rate': 1e-3,
    'gamma': 0.99,               # discount factor
    'epsilon_start': 1.0,        # exploration rate
    'epsilon_min': 0.01,
    'epsilon_decay': 0.9999,
    'buffer_size': 50000,        # experience replay
    'batch_size': 32,
    'target_update_frequency': 10,
    'max_steps_per_episode': 5000,
    'coverage_radius': 0.1,      # rayon de couverture du robot
    'save_frequency': 5,
    'save_dir': './coverage_models'
}
