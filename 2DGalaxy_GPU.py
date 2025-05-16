import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rc
from IPython.display import HTML

rc('animation', writer='pillow')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_accelerations(positions, masses, G=1.0, eps=1e-4):
    N = positions.shape[0]
    # delta[i,j] = positions[j] - positions[i]
    delta = positions.unsqueeze(1) - positions.unsqueeze(0)
    dist_sqr = (delta ** 2).sum(-1) + eps
    # Prevent self interaction
    dist_sqr += torch.eye(N, device=device) * 1e9
    inv_dist3 = dist_sqr.pow(-1.5)
    forces = G * delta * inv_dist3.unsqueeze(-1) * masses.view(1, -1, 1)
    return forces.sum(1)

def step(positions, velocities, masses, dt):
    acc = compute_accelerations(positions, masses)
    velocities += acc * dt
    positions += velocities * dt

# Particle setup
N = 50
positions = (torch.rand(N, 2, device=device) * 2 - 1).float()
velocities = (torch.rand(N, 2, device=device) * 0.1 - 0.05).float()
masses = (torch.rand(N, device=device) * 4 + 1).float()

dt = 0.01

# Plot setup
fig, ax = plt.subplots()
sc = ax.scatter([], [], s=10)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')


def update(frame):
    step(positions, velocities, masses, dt)
    sc.set_offsets(positions.cpu().numpy())
    return [sc]

ani = FuncAnimation(fig, update, frames=300, interval=20, blit=False)
plt.show()
HTML(ani.to_jshtml())
