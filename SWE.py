import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing
import threading
import os
import sys
import subprocess
import argparse
from matplotlib.animation import FuncAnimation
import matplotlib
from islands import australia as island_array


VIDEO_NAME_HEATMAP = "velocity_heatmap"
VIDEO_NAME = "velocity_field"
COLOUR = "limegreen"
OUTLINE_COLOUR = "red"
FRAMERATE = 60


# Updates the height (h), and velocities (u, v) of the shallow water equations
# for a single timestep using finite difference methods. Enforces boundary
# conditions and ensures velocity is set to zero on islands.
def update(h, u, v, dt, h0, dx, dy, g, taken_points):
    H = h + h0  # Total water height

    # Compute height gradients
    dhdx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2 * dx)
    dhdy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) / (2 * dy)

    # Update velocities using height gradients
    # u is the horizontal while v is the vertical movement (x and y vectors)
    u_new = u - g * dt * dhdx
    v_new = v - g * dt * dhdy

    # Apply boundary conditions: set velocities to 0 at the edges
    u_new[:, 0] = 0.0       # Western boundary
    u_new[:, -1] = 0.0      # Eastern boundary
    v_new[0, :] = 0.0       # Southern boundary
    v_new[-1, :] = 0.0      # Northern boundary

    update_islands(u_new, v_new, taken_points)

    # Compute fluxes
    flux_x = H * u_new
    flux_y = H * v_new

    # Update water height using flux divergence
    div_flux_x = (np.roll(flux_x, -1, axis=1) - np.roll(
        flux_x, 1, axis=1)) / (2 * dx)
    div_flux_y = (np.roll(flux_y, -1, axis=0) - np.roll(
        flux_y, 1, axis=0)) / (2 * dy)
    h_new = h - dt * (div_flux_x + div_flux_y)

    return h_new, u_new, v_new


velocity_magnitude_list = []


# Updates velocities on island points to zero, ensuring no water movement.
# Computes velocity magnitude for visualization purposes and appends it to
# the global velocity_magnitude_list for later use in animations.
def update_islands(u_new, v_new, taken_points):
    current_velocity_magnitude = np.zeros((150, 150))
    for i in taken_points:
        velocity_magnitude = np.sqrt(
            v_new[i[0]][i[1]]**2 + u_new[i[0]][i[1]]**2)
        current_velocity_magnitude[i[0]][i[1]] = velocity_magnitude
        u_new[i[0]][i[1]] = 0.0
        v_new[i[0]][i[1]] = 0.0
    velocity_magnitude_list.append(current_velocity_magnitude)

    return u_new, v_new


def grid_check(nx, ny, islands, taken_points):
    """
    Checks whether each grid point lies within any of the defined islands.
    If so, it marks the point as occupied and adds it to the taken_points list.
    This is used to enforce boundary conditions for water movement and
    visualization.
    """
    for x in range(nx):
        for y in range(ny):
            for i in islands:
                if [x, y] not in taken_points:
                    # To translate the indeces of the grid to the drawn grid.
                    # It should be possible to not hardcode this.
                    x_coords = (x-75)/150 * 1000
                    y_coords = (y-75)/150 * 1000

                    if check_island_bounds((y_coords, x_coords), i):
                        taken_points.append([x, y])
    return taken_points


def draw_compl_islands(ax, islands):
    """
    Draws all islands onto the plot.
    """
    for i in islands:
        x_s = [edge[0] for edge in i]
        y_s = [edge[1] for edge in i]
        ax.plot(x_s, y_s, color=OUTLINE_COLOUR)


def create_compl_islands(shape, islands):
    """
    Takes the outline of an island (as a list of points) and ensures it forms a
    closed polygon by appending the first point at the end. Stores the
    processed shape in the islands array for use in grid checks and plotting.
    """

    # This code is mainly used to draw the final line of the island shape
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


def plot_velocity_map(collisions_v, collisions_u):
    plt.scatter(collisions_v, collisions_u)
    plt.show()


# Creates an animation of the velocity field over time. The quiver plot shows
# the velocity vectors scaled and sampled at intervals defined by q_int.
# title_offset adjusts the time displayed in the animation title.
def velocity_animation(X, Y, u_list, v_list, frame_interval,
                       filename, islands, island_points, title_offset=0,
                       use_colour=False):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    plt.title("Velocity field $\\mathbf{{u}}(x,y)$ after 0.0 days",
              fontname="serif", fontsize=19)
    plt.xlabel("x [km]", fontname="serif", fontsize=16)
    plt.ylabel("y [km]", fontname="serif", fontsize=16)
    q_int = 3

    # Fill in the islands
    if use_colour:
        for i in island_points:
            ax.plot(i[1], i[0], "o", color=COLOUR)

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
            "Velocity field $\\mathbf{{u}}(x,y,t)$ after t = {:.2f}".format(
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


def plot_velocity_heatmap(velocity_magnitude_list, video_name):
    # Determine global min and max values for consistent scaling
    global_min = np.min([np.min(frame) for frame in velocity_magnitude_list])
    global_max = np.max([np.max(frame) for frame in velocity_magnitude_list])

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    cmap = matplotlib.colormaps.get_cmap('hot')
    cmap.set_bad(color='black')  # Set bad values to black

    # Use global min and max for consistent color scaling
    cax = ax.imshow(velocity_magnitude_list[0], cmap='hot',
                    interpolation='nearest', origin='lower',
                    vmin=global_min, vmax=global_max)
    fig.colorbar(cax, label='Velocity Magnitude')
    plt.title('Heatmap of Wave Impact on Island')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    def update(frame):
        cax.set_array(velocity_magnitude_list[frame])
        return cax,

    anim = FuncAnimation(fig, update, frames=len(
        velocity_magnitude_list), blit=True)
    anim.save(video_name, fps=FRAMERATE, dpi=200)


def print_loading_message(message, stop_event):
    while not stop_event.is_set():
        for i in range(4):
            if stop_event.is_set():
                break
            sys.stdout.write(f"\r{message}{'.' * i}   ")
            sys.stdout.flush()
            time.sleep(0.5)
    sys.stdout.write("\r" + " " * (len(message) + 4) + "\r")


def main(args: argparse.Namespace):
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

    # Create the grid
    X, Y = np.meshgrid(x, y)

    # Initialize fields
    h = np.ones((ny, nx)) * h0
    u = np.zeros((ny, nx))  # x-component velocity
    v = np.zeros((ny, nx))  # y-component velocity

    offset_x = -450 * 1000
    offset_y = 300 * 1000
    eta_n = np.exp(-((X-offset_x)**2/(2*(0.05E+6)**2) +
                   (Y-offset_y)**2/(2*(0.05E+6)**2)))
    h += eta_n  # Superimpose the disturbance

    # Stores islands
    islands = []
    # store which grid points are on any island. makes drawing easier
    taken_points = []

    # Main simulation loop
    t = time.time()
    total_time = time.time()
    u_list = []
    v_list = []
    seconds = args.seconds
    num_frames = FRAMERATE * seconds

    if args.verbose:
        print("Starting simulation calculations...\n")

    stop_event = threading.Event()
    loading_thread = threading.Thread(target=print_loading_message, args=(
        "Loading", stop_event))
    loading_thread.start()

    # Scales normalized island coordinates to the simulation grid's range,
    # ensuring they align with the physical domain boundaries.
    def scale_to_map(normalized_coords, min_val, max_val):
        scale_range = max_val - min_val
        return [
            (coord[0] * scale_range + min_val,
             -coord[1] * scale_range + max_val)  # Flip y-coordinate
            for coord in normalized_coords
        ]

    create_compl_islands(scale_to_map(island_array, -350, 350),
                         islands)

    # Checks if grid points are in an island. Only needs to be executed once.
    taken_points = grid_check(nx, ny, islands, taken_points)
    # Convert taken points to array of actual coordinates on plot
    pixels_in_islands = [[
        (x-75)/150 * 1000, (y-75)/150 * 1000] for x, y in taken_points]

    for _ in range(num_frames):
        max_H = np.max(h + h0)  # Total height for CFL condition
        dt = cfl * dx / np.sqrt(g * max_H)
        h, u, v = update(h, u, v, dt, h0, dx, dy, g, taken_points)
        u_list.append(u.copy())
        v_list.append(v.copy())

    stop_event.set()
    loading_thread.join()
    print("Math took {:.2f} seconds".format(time.time() - t))

    if args.make_heatmap and args.verbose:
        s = "Creating heatmap"
        r1 = "using multiprocessing..."
        r2 = "without multiprocessing..."
        r = r1 if args.use_multiprocessing else r2
        print(f"\n{s} {r}")

    stop_event = threading.Event()
    loading_thread = threading.Thread(target=print_loading_message, args=(
        "Loading", stop_event))
    loading_thread.start()

    t = time.time()

    # Generates the heatmap of velocity magnitudes. When multiprocessing is
    # enabled, splits the workload among multiple processes to improve
    # efficiency. Uses ffmpeg to combine partial videos into a final output.
    if args.make_heatmap:
        if args.use_multiprocessing:
            num_procs = args.num_procs
            procs = []
            for i in range(num_procs):
                startindex = i * (len(velocity_magnitude_list) // num_procs)
                endindex = (i + 1) * (len(
                    velocity_magnitude_list) // num_procs)

                velocity_magnitude_list_section = velocity_magnitude_list[
                    startindex:endindex]
                p = multiprocessing.Process(
                    target=plot_velocity_heatmap,
                    args=(velocity_magnitude_list_section,
                          f"{VIDEO_NAME_HEATMAP}_{i}.mp4",),
                )
                p.start()
                procs.append(p)

            for p in procs:
                p.join()

            # Stitch videos with ffmpeg.
            # Make sure ffmpeg is callable within the terminal.
            with open("videos.txt", "w") as f:
                for i in range(num_procs):
                    # Write the video filenames to a text file so ffmpeg
                    # doesn't skip any videos.
                    f.write(f"file {VIDEO_NAME_HEATMAP}_{i}.mp4\n")
            command = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i",
                       "videos.txt", "-c", "copy", f"{VIDEO_NAME_HEATMAP}.mp4"]
            # Hides the output, for debugging remove:
            # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            subprocess.run(command, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

            # Remove temporary video and text files
            for i in range(num_procs):
                os.remove(f"{VIDEO_NAME_HEATMAP}_{i}.mp4")
            os.remove("videos.txt")

        else:
            plot_velocity_heatmap(velocity_magnitude_list,
                                  f"{VIDEO_NAME_HEATMAP}.mp4")

    stop_event.set()
    loading_thread.join()

    if args.make_heatmap:
        print("Heatmap took {:.2f} seconds to make".format(time.time() - t))

    if args.make_video and args.verbose:
        s = "Creating velocity field video"
        c = f"{s} with colour" if args.use_colour else s
        r1 = "using multiprocessing..."
        r2 = "without multiprocessing..."
        r = r1 if args.use_multiprocessing else r2
        print(f"\n{c} {r}")

    stop_event = threading.Event()
    loading_thread = threading.Thread(target=print_loading_message, args=(
        "Loading", stop_event))
    loading_thread.start()

    t = time.time()

    # Create the velocity field video
    if args.make_video:
        if args.use_multiprocessing:
            num_procs = args.num_procs
            procs = []
            for i in range(num_procs):
                startindex = i * (num_frames // num_procs)
                endindex = (i + 1) * (num_frames // num_procs)
                u_list_section = u_list[startindex:endindex]
                v_list_section = v_list[startindex:endindex]
                title_offset = startindex * 10 / 3600
                p = multiprocessing.Process(
                    target=velocity_animation,
                    args=(X, Y, u_list_section, v_list_section, 10,
                          f"{VIDEO_NAME}_{i}", islands, pixels_in_islands,
                          title_offset, args.use_colour),
                )
                p.start()
                procs.append(p)

            for p in procs:
                p.join()

            # Stitch videos with ffmpeg.
            # Make sure ffmpeg is callable within the terminal.
            with open("videos.txt", "w") as f:
                for i in range(num_procs):
                    # Write the video filenames to a text file so ffmpeg
                    # doesn't skip any videos.
                    f.write(f"file {VIDEO_NAME}_{i}.mp4\n")

            command = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i",
                       "videos.txt", "-c", "copy", f"{VIDEO_NAME}.mp4"]
            # Hides the output, for debugging remove:
            subprocess.run(command, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

            # Remove temporary video and text files
            for i in range(num_procs):
                os.remove(f"{VIDEO_NAME}_{i}.mp4")
            os.remove("videos.txt")

        else:
            velocity_animation(X, Y, u_list, v_list, frame_interval=10,
                               filename=VIDEO_NAME, island_points=taken_points,
                               islands=islands, use_colour=args.use_colour)
    stop_event.set()
    loading_thread.join()
    if args.make_video:
        print("Animation took {:.2f} seconds".format(time.time() - t))

    print("Total calculation time: {:.2f} seconds".format(
        time.time() - total_time))


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--use_colour", action="store_true",
                        help="Use colour to indicate islands", default=False,
                        required=False)
    parser.add_argument("-mp", "--use_multiprocessing", action="store_true",
                        help="Use multiprocessing for video creation "
                             "(requires ffmpeg)",
                        required=False)
    parser.add_argument("-np", "--num_procs", type=int,
                        help="Number of processes to use for video creation",
                        default=4, required=False)
    parser.add_argument("-hm", "--make_heatmap", action="store_true",
                        help="Make heatmap of wave impact on island",
                        default=False, required=False)
    parser.add_argument("-mv", "--make_video", action="store_true",
                        help="Make video of velocity field",
                        default=False, required=False)
    parser.add_argument("-s", "--seconds", type=int,
                        help="Number of seconds to simulate",
                        default=20, required=False)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print more information",
                        default=False, required=False)

    args = parser.parse_args()
    if not args.make_heatmap and not args.make_video:
        print("Error: Please specify at least one of the following flags: "
              "--make_heatmap, --make_video")
        exit(1)

    if args.verbose:
        print(args)

    if args.use_colour and args.make_video and args.verbose:
        print("Warning: Adding colour to the velocity video will make the "
              "video much slower to create.")

    main(args)


if __name__ == "__main__":
    parse()
