import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing
import os
import subprocess
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter as writer



USE_MULTIPROCESSING = True
VIDEO_NAME = "velocity_field"
FRAMERATE = 60

# Update function for the shallow water equations
def update(h, u, v, dt, h0, dx, dy, g, taken_points):
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

    update_islands(u_new, v_new, taken_points)

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


def update_islands(u_new, v_new, taken_points):
    for i in taken_points:
        u_new[i[0]][i[1]] = 0.0
        v_new[i[0]][i[1]] = 0.0

    return u_new, v_new


def grid_check(nx, ny, islands, taken_points):
    """
    For each point in the grid, check if it lies in an island.
    """
    for x in range(nx):
        for y in range(ny):
            for i in islands:
                if [y, x] not in taken_points:
                    # To translate the indeces of the grid to the drawn grid.
                    # It should be possible to not hardcode this.
                    x_coords = (x-75)/150 * 1000
                    y_coords = (y-75)/150 * 1000

                    if check_island_bounds((x_coords, y_coords), i):
                        # hoezo is dit geroteerd??
                        taken_points.append([y, x])
    return taken_points


def draw_compl_islands(ax, islands):
    """
        Draws all islands onto the plot.
    """
    for i in islands:
        x_s = [edge[0] for edge in i]
        y_s = [edge[1] for edge in i]
        ax.plot(x_s, y_s)


def create_compl_islands(shape, islands):
    """
    Shape can be given clockwise and counterclockwise. Stores an island in the
    islands array as a list of edge points.
    """

    # Dit is om de laatste lijn mee te nemen, anders wordt die niet getekend
    shape.append(shape[0])
    islands.append(shape)


# https://en.wikipedia.org/wiki/Point_in_polygon used the winding number
# algorithm
def check_island_bounds(point, island):
    """
    Checks if a given point is in the island using the winding number
    algorithm. This algorithm draws a line from the point towards infinity
    (or the edge of the plot in this case), and counts the amount of edges it
    intersects with. When the number of intersections is an even number, the
    point is outside the island, otherwise it is inside.

    returns true if a point is in the island, otherwise false.
    """

    # TODO plot different colours for inside and outside islands.
    intersect_count = 0

    # creates the line from the point to the edge of the plot.
    line1 = ((float(point[0]), float(point[1])), (500.0, float(point[1])))

    for i in range(len(island) - 1):
        line2 = ((float(island[i+1][0]), float(island[i+1][1])),
                 (float(island[i][0]), float(island[i][1])))

        # filter out the lines that are completely to the left of the point, or
        # are entirely above or below it.
        if (line2[0][1] > line1[0][1] and line2[1][1] > line1[0][1] or
           line2[0][1] < line1[0][1] and line2[1][1] < line1[0][1]):
            continue
        if line2[0][0] < line1[0][0] and line2[1][0] < line1[0][0]:
            continue

        # If the line intersects with the edge's endpoint, add a half. This
        # causes the connecting line to do the same, making it count as one
        # intersection
        if ((line1[0][0] <= line2[0][0] and line1[0][1] == line2[0][1]) or
           (line1[0][0] <= line2[1][0] and line1[0][1] == line2[1][1])):
            intersect_count = intersect_count + 0.5
            continue

        # Returns the determinant.
        def det(a, b):
            return a[0]*b[1]-a[1]*b[0]

        # Calculates the determinant of the two lines.
        # Formula taken from:
        # https://lazyjobseeker.github.io/en/posts/winding-number-algorithm/
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
        dett = det(xdiff, ydiff)

        # if the determinant is 0, the point is on the line. Otherwise there
        # is an intersection with the line, but not necessarily the line
        # segment.
        if dett == 0:
            continue
        else:
            # Formula used to determine the intersection point. This may not
            # necessarily lie on the line segment.
            d = (det(line1[0], line1[1]), det(line2[0], line2[1]))
            x = det(d, xdiff)/dett
            y = det(d, ydiff)/dett

            # if The intersection is to the left of the point, ignore it.
            if x < line1[0][0]:
                continue

            # Checks if the intersection is on the line segment.
            if (x <= max([line2[0][0], line2[1][0]])
                and x >= min([line1[0][0], line2[0][0], line2[1][0]])
                and y <= max([line2[0][1], line2[1][1]])
               and y >= min([line2[0][1], line2[1][1]])):

                intersect_count = intersect_count + 1

    return True if intersect_count % 2 != 0 else False


# Animation function to create the visualization
def velocity_animation(X, Y, u_list, v_list, frame_interval, filename, islands, title_offset = 0):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days",
              fontname="serif", fontsize=19)
    plt.xlabel("x [km]", fontname="serif", fontsize=16)
    plt.ylabel("y [km]", fontname="serif", fontsize=16)
    q_int = 3

    # Draws the islands to the plot
    draw_compl_islands(ax, islands)

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
                (num * frame_interval / 3600) + title_offset
            ),
            fontname="serif",
            fontsize=19,
        )
        Q.set_UVC(u[::q_int, ::q_int], v[::q_int, ::q_int])
        return Q,

    anim = FuncAnimation(fig, update_quiver, frames=len(u_list), interval=10,
                         blit=False)
    anim.save(f"{filename}.mp4", fps=FRAMERATE, dpi=200)
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


def main():
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
    # offset_x = Lx/2.7
    # offset_y = Ly/4.0
    offset_x = 375000
    offset_y = 250000
    eta_n = np.exp(-((X-offset_x)**2/(2*(0.05E+6)**2) + (Y-offset_y)**2/(2*(0.05E+6)**2)))
    h += eta_n  # Superimpose the disturbance

    # Stores islands
    islands = []
    # store which grid points are on any island. makes drawing easier
    taken_points = []


    # Main simulation loop
    t = time.time()
    u_list = []
    v_list = []
    seconds = 20
    num_frames = FRAMERATE * seconds

    # is nu nogal gehardcoded, maar ik zie geen makkelijkere oplossing
    create_compl_islands([(0, 450),
                        (30, 260),
                        (20, 100),
                        (10, -100),
                        (-50, -150),
                        (-100, 1),
                        (-75, 200)],
                        islands)

    # Checks which grid points are in an island. Only needs to be executed once.
    taken_points = grid_check(nx, ny, islands, taken_points)

    for _ in range(num_frames):
        max_H = np.max(h + h0)  # Total height for CFL condition
        dt = cfl * dx / np.sqrt(g * max_H)
        h, u, v = update(h, u, v, dt, h0, dx, dy, g, taken_points)
        u_list.append(u.copy())
        v_list.append(v.copy())

    print("Math took {:.2f} seconds".format(time.time() - t))

    t = time.time()

    if USE_MULTIPROCESSING:
        print("Using multiprocessing")
        num_procs = 4   # Don't use too many or your laptop will explode
        procs = []
        for i in range(num_procs):
            startindex = i * (num_frames // num_procs)
            endindex = (i + 1) * (num_frames // num_procs)
            u_list_section = u_list[startindex:endindex]
            v_list_section = v_list[startindex:endindex]
            title_offset = startindex * 10 / 3600
            p = multiprocessing.Process(
                target=velocity_animation,
                args=(X, Y, u_list_section, v_list_section, 10, f"{VIDEO_NAME}_{i}", islands, title_offset),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        # Stitch videos with ffmpeg. Make sure ffmpeg is callable within the terminal.
        with open("videos.txt", "w") as f:
            for i in range(num_procs):
                f.write(f"file {VIDEO_NAME}_{i}.mp4\n")   # Write the video filenames to a text file so ffmpeg doesn't skip any videos.
        command = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "videos.txt", "-c", "copy", f"{VIDEO_NAME}.mp4"]
        subprocess.run(command)

        # Remove temporary video and text files
        for i in range(num_procs):
            os.remove(f"{VIDEO_NAME}_{i}.mp4")
        os.remove("videos.txt")

    else:
        print("Not using multiprocessing")
        velocity_animation(X, Y, u_list, v_list, frame_interval=10,
                    filename=VIDEO_NAME, islands=islands)

    print("Animation took {:.2f} seconds".format(time.time() - t))


if __name__ == "__main__":
    main()