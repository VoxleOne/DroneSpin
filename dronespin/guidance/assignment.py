import numpy as np


def assign_drones_to_vertices(n_drones, n_vertices):
    """
    Assign N drones to M vertices. Handles N != M.
    Returns list of length n_drones where each element is a vertex index.

    Strategy:
    - If N <= M: assign each drone to a distinct vertex (first N vertices)
    - If N > M: assign vertices cyclically (round-robin)
    """
    return [i % n_vertices for i in range(n_drones)]
