import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import combinations


def plot_simulation(drones, formation, title="DroneSpin", show_forward=True, filename=None):
    """
    Plot drones and formation vertices in 3D.

    drones: list of DroneState
    formation: Formation
    filename: if set, save to file instead of showing
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    vertices = np.array(formation.vertices)
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
               c='blue', marker='^', s=100, label='Vertices', zorder=5)

    for i, j in combinations(range(len(vertices)), 2):
        xs = [vertices[i, 0], vertices[j, 0]]
        ys = [vertices[i, 1], vertices[j, 1]]
        zs = [vertices[i, 2], vertices[j, 2]]
        ax.plot(xs, ys, zs, 'b-', alpha=0.3, linewidth=0.8)

    if not drones:
        colors = np.empty((0, 4))
    else:
        colors = plt.cm.Set1(np.linspace(0, 1, len(drones)))
    for i, drone in enumerate(drones):
        p = drone.position
        ax.scatter([p[0]], [p[1]], [p[2]],
                   c=[colors[i]], marker='o', s=80, label=f'Drone {drone.drone_id}', zorder=5)

        if show_forward:
            q = drone.orientation  # [x,y,z,w]
            x, y, z, w = q
            fwd = np.array([
                2*(x*z + w*y),
                2*(y*z - w*x),
                1 - 2*(x*x + y*y)
            ])
            scale = 0.3
            ax.quiver(p[0], p[1], p[2], fwd[0]*scale, fwd[1]*scale, fwd[2]*scale,
                      color=colors[i], arrow_length_ratio=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=7)

    if filename:
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig
