import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter as writer

# Grid parameters
nx, ny = 150, 150
Lx, Ly = 1E+6, 1E+6     # length of x and y domain
x = np.linspace(-Lx/2, Lx/2, nx)
y = np.linspace(-Ly/2, Ly/2, ny)
dx = dy = x[1] - x[0]

# Physical constants
g = 9.81  # Gravitational acceleration
h0 = 100.0  # Resting water depth
cfl = 0.9  # CFL condition factor
f_0 = 1.16E-4 # Fixed part of coriolis parameter

# Create the grid
X, Y = np.meshgrid(x, y)

# Initialize fields
h = np.ones((ny, nx)) * h0
u = np.zeros((ny, nx))  # x-component velocity
v = np.zeros((ny, nx))  # y-component velocity

# Apply Gaussian disturbance to h
Lr = np.sqrt(g*h)/(f_0*4)

# eta_n = np.exp(-((X)**2 / (Lr**2) + (Y)**2 / (Lr**2)))  # Gaussian disturbance
#offset_x = Lx/2.7
#offset_y = Ly/4.0
offset_x = 375000
offset_y = 250000
eta_n = np.exp(-((X-offset_x)**2/(2*(0.05E+6)**2) + (Y-offset_y)**2/(2*(0.05E+6)**2)))
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

    # Apply boundary conditions: set velocities to 0 at the edges
    u_new[:, 0] = 0.0       # Western boundary
    u_new[:, -1] = 0.0      # Eastern boundary

    v_new[0, :] = 0.0       # Southern boundary
    v_new[-1, :] = 0.0      # Northern boundary

    # Compute fluxes
    flux_x = H * u_new
    flux_y = H * v_new

    # Update water height using flux divergence
    div_flux_x = (np.roll(flux_x, -1, axis=1) - np.roll(flux_x, 1, axis=1)) / (2 * dx)
    div_flux_y = (np.roll(flux_y, -1, axis=0) - np.roll(flux_y, 1, axis=0)) / (2 * dy)
    h_new = h - dt * (div_flux_x + div_flux_y)

    return h_new, u_new, v_new


def draw_compl_islands(shape):
    """
    shape must be an array of vectors, where the outline is given in a clockwise manner.
    """
    edges = []
    edges_x = []
    edges_y = []
    for i in range(len(shape)-1, -2, -1):
        # edges_x.append(shape[i])
        edges.append([[shape[i][0], shape[i-1][0]],
                      [shape[i][1], shape[i-1][1]]])
        edges_x.append(shape[i][0])
        edges_y.append(shape[i][1])
    
    # test check if [-1, -1] is in island, should return True (it's in the island)
    print(check_island_bounds(np.array([-1, -1]), np.array(edges)))
    
    # plots island
    plt.plot(edges_x, edges_y)
    plt.show()


def check_island_bounds(point, island):
    """
    returns true if in island, otherwise false
    """
    dists = []
    ress = []
    for edge in island:
        dist = np.abs(np.linalg.norm(np.cross(edge[1]-edge[0], edge[0]- point))/np.linalg.norm(edge[1]-edge[0]))
        dists.append(dist)
        ress.append((point[0]-edge[0][0])*(edge[1][1])-edge[0][1])
        - ((point[1]-edge[0][1])*(edge[1][0]-edge[0][0]))
    # print("dists = ", dists)
    # print(ress)
    # print(min(dists), ress[np.argmin(dists)], (ress[np.argmin(dists)] < 0))
    return ress[np.argmin(dists)] < 0


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
    anim.save(f"{filename}.mp4", fps=60, dpi=200)
    return anim

def surface_animation(X, Y, u_list, v_list, frame_interval, filename):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname="serif", fontsize=19)
    plt.xlabel("x [km]", fontname="serif", fontsize=16)
    plt.ylabel("y [km]", fontname="serif", fontsize=16)

    surf = ax.plot_surface(X, Y, u_list[0], cmap=plt.cm.RdBu_r)


    def update_surf(num):

        z_list = np.array([[np.linalg.norm(np.array([u_list[num][x], v_list[num][y]])) for x in range(len(u_list[num]))] for y in range(len(v_list[num]))])

        ax.clear()
        surf = ax.plot_surface(X/1000, Y/1000, z_list, cmap = plt.cm.RdBu_r)
        ax.set_title("Surface elevation $\eta(x,y,t)$ after $t={:.2f}$ hours".format(
            num*frame_interval/3600), fontname = "serif", fontsize = 19, y=1.04)
        ax.set_xlabel("x [km]", fontname = "serif", fontsize = 14)
        ax.set_ylabel("y [km]", fontname = "serif", fontsize = 14)
        ax.set_zlabel("$\eta$ [m]", fontname = "serif", fontsize = 16)
        ax.set_zlim(-5, 20)
        plt.tight_layout()
        return surf,

    anim = FuncAnimation(fig, update_surf,
        frames = len(u_list), interval = 10, blit = False)
    mpeg_writer = writer(fps = 24, bitrate = 10000,
        codec = "libx264", extra_args = ["-pix_fmt", "yuv420p"])
    anim.save(f"{filename}.mp4", fps=24, dpi=200)
    # anim.save("{}.mp4".format(filename), writer = mpeg_writer)
    return anim,    # Need to return anim object to see the animation

# Main simulation loop
u_list = []
v_list = []
seconds = 10
num_frames = 24 * seconds

for frame in range(num_frames):
    max_H = np.max(h + h0)  # Total height for CFL condition
    dt = cfl * dx / np.sqrt(g * max_H)
    h, u, v = update(h, u, v, dt)
    u_list.append(u.copy())
    v_list.append(v.copy())

# velocity_animation(X, Y, u_list, v_list, frame_interval=10, filename="velocity_field")
draw_compl_islands([[1.5, 2.0],
                    [2.0, 1.0],
                    [2.0, -2.0],
                    [-1.0, -3.0],
                    [-3.0, -1.0],
                    [-1.0, 2.0]])
# surface_animation(X, Y, u_list, v_list, frame_interval=10, filename="surface_water")