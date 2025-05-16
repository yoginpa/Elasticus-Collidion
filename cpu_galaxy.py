import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from cpu_simulator import PhysicsSimulator

N = 100
positions = np.random.uniform(-1, 1, (N, 2))
velocities = np.random.uniform(-0.05, 0.05, (N, 2))
masses = np.random.uniform(1, 5, N)

sim = PhysicsSimulator(positions, velocities, masses)
dt = 0.01

fig, ax = plt.subplots()
sc = ax.scatter(sim.positions[:, 0], sim.positions[:, 1], s=10)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')


def update(frame):
    sim.step(dt)
    sc.set_offsets(sim.positions)
    return sc,


ani = FuncAnimation(fig, update, frames=300, interval=20, blit=False)
plt.show()
