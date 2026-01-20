from enum import Enum

class PathFindingMode(Enum):
    ASTAR = "astar"
    RL = "rl"

# =====================
# HYPERPARAMÈTRE GLOBAL
# =====================

PATHFINDING_MODE = PathFindingMode.RL
# PATHFINDING_MODE = PathFindingMode.RL
