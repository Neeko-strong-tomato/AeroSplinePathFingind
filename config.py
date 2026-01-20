from enum import Enum

class PlanningMode(Enum):
    COVERAGE = "coverage"
    RL = "rl" 

PLANNING_MODE = PlanningMode.COVERAGE

# Coverage parameters
ANGLE_THRESHOLD_DEG = 30
ZIGZAG_STEP = 0.05
