import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Grid parameters
nx, ny = 101, 101
x = np.linspace(-500, 500, nx)
y = np.linspace(-500, 500, ny)
dx = dy = x[1] - x[0]

# Physical constants
g = 9.81  # Gravitational acceleration
h0 = 100.0  # Resting water depth
cfl = 0.9  # CFL condition factor

# Create the grid
X, Y = np.meshgrid(x, y)

# Initialize fields
h = np.ones((ny, nx)) * h0
u = np.zeros((ny, nx))  # x-component velocity
v = np.zeros((ny, nx))  # y-component velocity

# Apply Gaussian disturbance to h
Lx, Ly = 100, 100  # Characteristic length scales of the disturbance
eta_n = 5.0 * np.exp(-((X)**2 / (2 * Lx**2) + (Y)**2 / (2 * Ly**2)))  # Gaussian disturbance
h += eta_n  # Superimpose the disturbance

# Update function for the shallow water equations
def update(h, u, v, dt):
    H = h + h0  # Total water height

    # Compute height gradients
    dhdx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2 * dx)
    dhdy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) / (2 * dy)

    # Update velocities using height gradients
    u_new = u - g * dt * dhdx
    v_new = v - g * dt * dhdy

    # Compute fluxes
    flux_x = H * u_new
    flux_y = H * v_new

    # Update water height using flux divergence
    div_flux_x = (np.roll(flux_x, -1, axis=1) - np.roll(flux_x, 1, axis=1)) / (2 * dx)
    div_flux_y = (np.roll(flux_y, -1, axis=0) - np.roll(flux_y, 1, axis=0)) / (2 * dy)
    h_new = h - dt * (div_flux_x + div_flux_y)

    return h_new, u_new, v_new

# Animation function to create the visualization
def velocity_animation(X, Y, u_list, v_list, frame_interval, filename):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname="serif", fontsize=19)
    plt.xlabel("x [km]", fontname="serif", fontsize=16)
    plt.ylabel("y [km]", fontname="serif", fontsize=16)
    q_int = 3
    Q = ax.quiver(
        X[::q_int, ::q_int] / 1000.0, Y[::q_int, ::q_int] / 1000.0,
        u_list[0][::q_int, ::q_int], v_list[0][::q_int, ::q_int],
        scale=0.2, scale_units="inches",
    )

    def update_quiver(num):
        u = u_list[num]
        v = v_list[num]
        ax.set_title(
            "Velocity field $\mathbf{{u}}(x,y,t)$ after t = {:.2f} hours".format(
                num * frame_interval / 3600
            ),
            fontname="serif",
            fontsize=19,
        )
        Q.set_UVC(u[::q_int, ::q_int], v[::q_int, ::q_int])
        return Q,

    anim = FuncAnimation(fig, update_quiver, frames=len(u_list), interval=10, blit=False)
    anim.save(f"{filename}.mp4", fps=24, dpi=200)
    return anim

# Main simulation loop
u_list = []
v_list = []
num_frames = 200

for frame in range(num_frames):
    max_H = np.max(h + h0)  # Total height for CFL condition
    dt = cfl * dx / np.sqrt(g * max_H)
    h, u, v = update(h, u, v, dt)
    u_list.append(u.copy())
    v_list.append(v.copy())

velocity_animation(X, Y, u_list, v_list, frame_interval=10, filename="velocity_field")
