import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from matplotlib import rc
from IPython.display import HTML

rc('animation', writer='pillow')

class Particle:
    def __init__(self, x, y, velocity, mass):
        self.position = [x, y]
        self.velocity = [velocity[0], velocity[1]]
        self.mass = mass
    
    def update(self, x, y, velocity, mass):
        self.position = [x, y]
        self.velocity = [velocity[0], velocity[1]]
        self.mass = mass
    
    def step(self, dt):
        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt


def gravPull(particle1, particle2):
    dx = - particle1.position[0] + particle2.position[0]
    dy = - particle1.position[1] + particle2.position[1]
    epsilon = 1e-4
    dist_sqr = dx * dx + dy * dy + epsilon
    r = math.sqrt(dist_sqr)

    G = 1.0 # Gravitational Constant
    a_mag = G * particle2.mass / dist_sqr  # Acceleration magnitude

    # Normalize direction vector
    direction = [dx / r, dy / r]

    # Acceleration vector
    acceleration = [a_mag * direction[0], a_mag * direction[1]]
    return acceleration

def recalculate_all(particle_matrix):
    dt = 0.01
    N_particles = len(particle_matrix)
    for i in range(N_particles):
        total_acceleration = [0.0, 0.0]
        for j in range(N_particles):
            if (i == j): continue
            acceleration = gravPull(particle_matrix[i], particle_matrix[j])
            total_acceleration[0] += acceleration[0]
            total_acceleration[1] += acceleration[1]
        
        # Update velocity of particle i using total_acceleration
        particle_matrix[i].velocity[0] += total_acceleration[0] * dt
        particle_matrix[i].velocity[1] += total_acceleration[1] * dt

    # Finally call .step(dt) on each particle to update the actual positions of the particles
    for particle in particle_matrix:
        particle.step(dt)


## Particle Setup
particles = []
N = 50              # Starting with 50 to start with
for _ in range(N):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    vx = random.uniform(-0.05, 0.05)
    vy = random.uniform(-0.05, 0.05)
    mass = random.uniform(1, 5)
    particles.append(Particle(x, y, [vx, vy], mass))


## Plot Setup
fig, ax = plt.subplots()
sc = ax.scatter([], [], s=10)   # s=10 sets size
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')


## Animation update function
def update(frame):
    print(f"Frame: {frame}")  # Optional debug
    recalculate_all(particles)
    xs = [p.position[0] for p in particles]
    ys = [p.position[1] for p in particles]
    sc.set_offsets(list(zip(xs, ys)))
    return [sc]


## Create and run animation
ani = FuncAnimation(fig, update, frames=300, interval=20, blit=False)
plt.show()
#plt.close(fig)  # Hide the static plot
HTML(ani.to_jshtml())