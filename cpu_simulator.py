import numpy as np

class PhysicsSimulator:
    """Simple CPU-based physics simulator for N-body interactions."""

    def __init__(self, positions, velocities, masses, G=1.0, epsilon=1e-4):
        self.positions = np.asarray(positions, dtype=float)
        self.velocities = np.asarray(velocities, dtype=float)
        self.masses = np.asarray(masses, dtype=float)
        self.G = G
        self.epsilon = epsilon

    def _compute_accelerations(self):
        pos = self.positions
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2) + self.epsilon
        inv_dist3 = 1.0 / np.sqrt(dist_sq ** 3)
        factors = self.G * self.masses[np.newaxis, :] * inv_dist3
        accel = (diff * factors[:, :, np.newaxis]).sum(axis=1)
        return accel

    def step(self, dt):
        a = self._compute_accelerations()
        self.velocities += a * dt
        self.positions += self.velocities * dt
        return self.positions, self.velocities
