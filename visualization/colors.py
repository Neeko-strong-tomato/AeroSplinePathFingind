import numpy as np
import matplotlib.cm as cm

def path_colors(n):
    cmap = cm.get_cmap("viridis")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]
