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


islands = []
# store which grid points are on any island. makes drawing easier
taken_points = []


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

    # square in middle
    # for x in range(50, 101):
    #     for y in range(50, 101):
    #         u_new[x, y] = 0.0
    #         v_new[x, y] = 0.0

    update_islands(u_new, v_new)

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


def update_islands(u_new, v_new):
    for i in taken_points:
        u_new[i[0]][i[1]] = 0.0
        v_new[i[0]][i[1]] = 0.0

    return u_new, v_new


def grid_check():
    print(nx)
    for x in range(60, 120):
        print(x)
        for y in range(60, 140):
            # print(x, y)
            # for i in islands:
            if [x, y] not in taken_points:
                x_coords = (x-75)/150 * 1000
                y_coords = (y-75)/150 * 1000
                # print(x_coords, y_coords)
                if check_island_bounds((x_coords, y_coords), islands[0]):
                    taken_points.append([x, y])


def draw_compl_islands(ax):
    for i in islands:
        x_s = [edge[0] for edge in i]
        y_s = [edge[1] for edge in i]
        ax.plot(x_s, y_s)
    #     for edge in i:
    #         ax.plot([edge[0][0], edge[0][1]], [edge[1][0], edge[1][1]])
    # ax.plot([0], [0])


def create_compl_islands(shape):
    """
    shape must be an array of vectors, where the outline is given in a clockwise manner.
    """
    edges = []
    # edges_x = []
    # edges_y = []
    # for i in range(len(shape)-1, -2, -1):
    #     # edges.append([[shape[i][0], shape[i-1][0]],
    #     #               [shape[i][1], shape[i-1][1]]])
    #     # edges.append([[shape[i][0], shape[i][1]],
    #     #               [shape[i-1][0], shape[i-1][1]]])
    #     edges_x.append(shape[i][0])
    #     edges_y.append(shape[i][1])

    # test check if [-1, -1] is in island, should return True (it's in the island)
    # print(check_island_bounds(np.array([-1, -1]), np.array(edges)))

    # islands.append(edges)
    shape.append(shape[0])
    islands.append(shape)


# https://lazyjobseeker.github.io/en/posts/winding-number-algorithm/ used
def check_island_bounds(point, island):
    """
    returns true if in island, otherwise false
    """
    dists = []
    ress = []
    intersect_count = 0
    # island.append(island[0])

    line1 = ((point[0], point[1]), (500, point[1]))

    # island.append(island[0])

    for i in range(len(island) - 1):
        # print(i, len(island))
        # print(island[i], island[i-1])
        line2 = (island[i+1], island[i])
        
        if (line2[0][1] > line1[0][1] and line2[1][1] > line1[0][1] or
            line2[0][1] < line1[0][1] and line2[1][1] < line1[0][1]):
            continue
        if line2[0][0] < line1[0][0] and line2[1][0] < line1[0][0]:
            continue

        def det(a, b):
            return a[0]*b[1]-a[1]*b[0]

        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
        dett = det(xdiff, ydiff)

        if dett == 0:
            continue
        else:
            d = (det(line1[0], line1[1]), det(line2[0], line2[1]))
            x = det(d, xdiff)/dett
            y = det(d, ydiff)/dett
            if (x <= max([line2[0][0], line2[1][0]])
                and x >= min([line1[0][0], line2[0][0], line2[1][0]])
                and y <= max([line1[0][1]])
                and y >= min([line1[0][1]])):
                intersect_count = intersect_count + 1
                print(line1, line2, x, y)
                print("ymax = ",max([line1[0][1]]))
                print("ymin = ",min([line1[0][1]]))
                print("xmax = ", max([line2[0][0], line2[1][0]]))
                print("xmin = ", min([line2[0][0], line1[0][0], line2[1][0]]))
    # print(c_point)
    # px, py = c_point
    # px, py = point
    # for i, v in enumerate(island[:-1]):
    #     segm = island[i: i+2]
    #     x_coords = np.transpose(segm)[0]
    #     y_coords = np.transpose(segm)[0]

    #     if px > np.max(x_coords):
    #         continue
    #     if (y_coords[0]-py)*(y_coords[1]-py) < 0:
    #         intersect_count += 1
    #     if y_coords[0] == py:
    #         intersect_count += 0.5
    #     if y_coords[1] == py:
    #         intersect_count += 0.5

    print(intersect_count)
    return True if intersect_count % 2 != 0 else False

        # segm = np.asarray(island[])

        # e0 = np.array([edge[0][0], edge[1][0]])
        # e1 = np.array([edge[0][1], edge[1][1]])
        # print(edge[0], edge[1], point)
        # # dist = np.abs(np.linalg.norm(np.cross(edge[1]-edge[0], edge[0]-point))/np.linalg.norm(edge[1]-edge[0]))
        # dist = np.abs(np.linalg.norm(np.cross(e1-e0, e0-point))/np.linalg.norm(e1-e0))
        # # dist = np.abs(np.linalg.norm(np.cross(new_e_1-new_e_0, new_e_0-point))/np.linalg.norm(new_e_1-new_e_0))
        # dists.append(dist)
        # # ress.append((point[0]-new_p_00)*(new_p_11-new_p_01)
        #             # - ((point[1]-new_p_01)*(new_p_10-new_p_00)))
        # ress.append(([edge[0][1]]-edge[0][0])*(point[1]-edge[1][0]) - (edge[1][1]-edge[1][0])*(point[0]-edge[0][0]))
        # # ress.append((point[0]-edge[0][0])*(edge[1][1])-edge[0][1])
        # # - ((point[1]-edge[0][1])*(edge[1][0]-edge[0][0]))

    if np.array_equal(point, np.array([150, 150])):
        print("point is", point)
        print(dists)
        # print(island)
        print(ress)
        print(ress[np.argmin(dists)][0], (ress[np.argmin(dists)][0] > 0))
    # print("dists = ", dists)
    # print(ress)
    # print(min(dists), ress[np.argmin(dists)], (ress[np.argmin(dists)] < 0))
    return ress[np.argmin(dists)][0] > 0


# Animation function to create the visualization
def velocity_animation(X, Y, u_list, v_list, frame_interval, filename):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname="serif", fontsize=19)
    plt.xlabel("x [km]", fontname="serif", fontsize=16)
    plt.ylabel("y [km]", fontname="serif", fontsize=16)
    q_int = 3

    draw_compl_islands(ax)
    plt.show()
    return

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
fps = 24
num_frames = fps * seconds

create_compl_islands([(0, 450),
                      (30, 260),
                      (20, 100),
                      (10, -100),
                      (-50, -150),
                      (-100, 1),
                      (-75, 200)])
# create_compl_islands([(10, 10),
#                       (10, 15),
#                       (15, 15),
#                       (15, -20),
#                       (0, -20),
#                       (-10, 5)])

# print(check_island_bounds((150, 150), islands[0]))
print("start gridcheck")
# grid_check()
print(check_island_bounds((1, 1), islands[0]))
print(check_island_bounds((1, 201), islands[0]))
print(check_island_bounds((-101, -101), islands[0]))
print(check_island_bounds((-101, 201), islands[0]))
# print(check_island_bounds((0, 0), islands[0]))
# print(check_island_bounds((-60, 300), islands[0]))
# print(check_island_bounds((-10, 300), islands[0]))

# print("start updating")
for frame in range(num_frames):
    max_H = np.max(h + h0)  # Total height for CFL condition
    dt = cfl * dx / np.sqrt(g * max_H)
    h, u, v = update(h, u, v, dt)
    u_list.append(u.copy())
    v_list.append(v.copy())

velocity_animation(X, Y, u_list, v_list, frame_interval=10, filename="velocity_field")
# surface_animation(X, Y, u_list, v_list, frame_interval=10, filename="surface_water")

# create_compl_islands([[-300, -300],
#                       [-300, 300],
#                       [300, 300],
#                       [300, -300]])

# draw_compl_islands([[1.5, 2.0],
#                     [2.0, 1.0],
#                     [2.0, -2.0],
#                     [-1.0, -3.0],
#                     [-3.0, -1.0],
#                     [-1.0, 2.0]])