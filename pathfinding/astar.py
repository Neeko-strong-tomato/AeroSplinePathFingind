import networkx as nx
import numpy as np

def astar_path(graph, start, goal):
    def heuristic(a, b):
        pa = graph.nodes[a]["pos"]
        pb = graph.nodes[b]["pos"]
        return np.linalg.norm(pa - pb)

    return nx.astar_path(
        graph,
        start,
        goal,
        heuristic=heuristic,
        weight="weight"
    )
