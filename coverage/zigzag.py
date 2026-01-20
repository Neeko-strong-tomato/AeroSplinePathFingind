import numpy as np

def discretize(points_2d, step=0.005):
    """
    Crée une grille binaire de couverture à partir des points 2D
    """
    min_xy = points_2d.min(axis=0)
    max_xy = points_2d.max(axis=0)
    size = np.ceil((max_xy - min_xy) / step).astype(int)

    grid = np.zeros(size, dtype=np.uint8)
    idx = ((points_2d - min_xy) / step).astype(int)
    grid[idx[:,0], idx[:,1]] = 1

    return grid, min_xy, step


def zigzag_path(grid):
    path = []
    rows, cols = grid.shape
    for i in range(rows):
        if i % 2 == 0:
            for j in range(cols):
                if grid[i,j]:
                    path.append((i,j))
        else:
            for j in reversed(range(cols)):
                if grid[i,j]:
                    path.append((i,j))
    return path


def reproject_to_3d(path_2d, min_xy, step, center, u, v):
    points_3d = []
    for i,j in path_2d:
        x = min_xy[0] + i*step
        y = min_xy[1] + j*step
        p = center + x*u + y*v
        points_3d.append(p)
    return np.array(points_3d)


import matplotlib.pyplot as plt

def visualize_path(points_3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points_3d[:,0], points_3d[:,1], points_3d[:,2], 'r.-')
    plt.show()
