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
COLOUR = "limegreen"
OUTLINE_COLOUR = "red"
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
        ax.plot(x_s, y_s, color=OUTLINE_COLOUR)


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
def velocity_animation(X, Y, u_list, v_list, frame_interval, filename, islands, taken_points, title_offset = 0):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days",
              fontname="serif", fontsize=19)
    plt.xlabel("x [km]", fontname="serif", fontsize=16)
    plt.ylabel("y [km]", fontname="serif", fontsize=16)
    q_int = 3
    # Draw every coordinate in taken_points as lime green
    for i in taken_points:
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
            "Velocity field $\mathbf{{u}}(x,y,t)$ after t = {:.2f}".format(
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


# def surface_animation(X, Y, u_list, v_list, frame_interval, filename):
#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

#     plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname="serif", fontsize=19)
#     plt.xlabel("x [km]", fontname="serif", fontsize=16)
#     plt.ylabel("y [km]", fontname="serif", fontsize=16)

#     surf = ax.plot_surface(X, Y, u_list[0], cmap=plt.cm.RdBu_r)


#     def update_surf(num):

#         z_list = np.array([[np.linalg.norm(np.array([u_list[num][x], v_list[num][y]])) for x in range(len(u_list[num]))] for y in range(len(v_list[num]))])

#         ax.clear()
#         surf = ax.plot_surface(X/1000, Y/1000, z_list, cmap = plt.cm.RdBu_r)
#         ax.set_title("Surface elevation $\eta(x,y,t)$ after $t={:.2f}$ hours".format(
#             num*frame_interval/3600), fontname = "serif", fontsize = 19, y=1.04)
#         ax.set_xlabel("x [km]", fontname = "serif", fontsize = 14)
#         ax.set_ylabel("y [km]", fontname = "serif", fontsize = 14)
#         ax.set_zlabel("$\eta$ [m]", fontname = "serif", fontsize = 16)
#         ax.set_zlim(-5, 20)
#         plt.tight_layout()
#         return surf,

#     anim = FuncAnimation(fig, update_surf,
#         frames = len(u_list), interval = 10, blit = False)
#     mpeg_writer = writer(fps = 24, bitrate = 10000,
#         codec = "libx264", extra_args = ["-pix_fmt", "yuv420p"])
#     anim.save(f"{filename}.mp4", fps=24, dpi=200)
#     # anim.save("{}.mp4".format(filename), writer = mpeg_writer)
#     return anim,    # Need to return anim object to see the animation


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

    # Define the scaling function for the new range of -500 to 500
    def scale_to_map(normalized_coords, min_val, max_val):
        scale_range = max_val - min_val
        return [
            (coord[0] * scale_range - max_val, coord[1] * -scale_range - min_val)
            for coord in normalized_coords
        ]

    # is nu nogal gehardcoded, maar ik zie geen makkelijkere oplossing
    island_array = [(0.717679658270422, 0.0),
  (0.7455672452453883, 0.09700176366843033),
  (0.7623211446740858, 0.09012844257579476),
  (0.7819147451042582, 0.11199294532627865),
  (0.7980922098569158, 0.19923237067678937),
  (0.8129812588142483, 0.21428571428571427),
  (0.8044515103338633, 0.21900317290201513),
  (0.8513513513513513, 0.2523305834713295),
  (0.8665461412092842, 0.2495590828924162),
  (0.8696343402225755, 0.2646114404164104),
  (0.8600953895071543, 0.26659911571970296),
  (0.8820734169368777, 0.31305114638447973),
  (0.9060795991012179, 0.31305114638447973),
  (0.9109697933227345, 0.34632642636428806),
  (0.9528343716068041, 0.3950617283950617),
  (0.9642289348171701, 0.380090483563259),
  (0.9697933227344993, 0.3902457148932713),
  (0.9633493627413621, 0.4567901234567901),
  (0.9721780604133545, 0.4487695146380274),
  (0.9762950049339078, 0.4982363315696649),
  (0.9602543720190779, 0.5811129529297945),
  (0.8926868044515104, 0.7163690105507444),
  (0.889101991048093, 0.7733686067019401),
  (0.8393591153432709, 0.7865961199294532),
  (0.8060413354531002, 0.8264514036667873),
  (0.7623211446740858, 0.7980044329072273),
  (0.7376788553259142, 0.8177153252599119),
  (0.6868044515103339, 0.8017778212552287),
  (0.6480965021056434, 0.7619047619047619),
  (0.6400245945798757, 0.7204585537918872),
  (0.6090423314943891, 0.7098765432098766),
  (0.6185725406379814, 0.6940035273368607),
  (0.6120826709062003, 0.6692968499287226),
  (0.6018397949659097, 0.6975308641975309),
  (0.5805934360508039, 0.7001763668430335),
  (0.5939654680018455, 0.6860670194003528),
  (0.6057233704292527, 0.6268579024049271),
  (0.5620031796502385, 0.6768885875152357),
  (0.5682121521123694, 0.6931216931216931),
  (0.5524642289348172, 0.6907338366696137),
  (0.5182829888712241, 0.613606351618743),
  (0.4507154213036566, 0.5836850396285376),
  (0.3298910212070368, 0.6067019400352733),
  (0.2890988655462573, 0.6287477954144621),
  (0.26311524591771285, 0.6649029982363316),
  (0.1875993640699523, 0.6586321348116491),
  (0.14149443561208266, 0.6960240548289761),
  (0.10810810810810811, 0.6940234248019046),
  (0.07333967508048679, 0.6728395061728395),
  (0.0688231962794798, 0.6437389770723104),
  (0.08517647546438518, 0.6358024691358025),
  (0.08505564387917329, 0.5860992343357482),
  (0.06680604797326009, 0.5114638447971781),
  (0.020433131229066093, 0.4091710758377425),
  (0.025558400483274336, 0.4021164021164021),
  (0.04292527821939587, 0.4329551508758885),
  (0.037360890302066775, 0.40328626276318774),
  (0.05097412273736858, 0.427689594356261),
  (0.03149254161073876, 0.37389770723104054),
  (0.04531001589825119, 0.301112117362918),
  (0.05564387917329094, 0.31582497902439055),
  (0.11287758346581876, 0.2573562568558455),
  (0.1287758346581876, 0.2681726256910778),
  (0.21303656597774245, 0.2355692667583467),
  (0.2410696237188864, 0.19753086419753085),
  (0.2384737678855326, 0.16996396295501076),
  (0.2591414944356121, 0.14787211147486873),
  (0.2702702702702703, 0.1757503995300284),
  (0.2671407627232371, 0.14462081128747795),
  (0.28855325914149443, 0.150027095023087),
  (0.2845786963434022, 0.12169000061378973),
  (0.30842607313195547, 0.11206148933292367),
  (0.307631160572337, 0.09703622118979555),
  (0.3267090620031797, 0.09713009880726012),
  (0.32909379968203495, 0.08308618082186937),
  (0.35135135135135137, 0.07861374213506135),
  (0.3815580286168522, 0.10963278112195454),
  (0.40858505564387915, 0.10779271135076071),
  (0.40790747543049055, 0.08641975308641975),
  (0.4340411835660621, 0.042328042328042326),
  (0.48171701112877585, 0.03402291873155214),
  (0.4610884953330721, 0.01675485008818342),
  (0.48171701112877585, 0.004109592856271445),
  (0.4843241948773641, 0.018518518518518517),
  (0.5421303656597775, 0.029492767293394607),
  (0.5397456279809221, 0.03798104302329329),
  (0.5816666560405857, 0.007054673721340388),
  (0.5609374085569244, 0.03439153439153439),
  (0.5763116057233705, 0.024463257394355224),
  (0.5834658187599364, 0.035115644631934013),
  (0.5666076591169094, 0.07671957671957672),
  (0.5839097747878484, 0.07848324514991181),
  (0.5862834212707586, 0.09347442680776014),
  (0.5635930047694754, 0.08365163005356596),
  (0.5508744038155803, 0.11148197590632258),
  (0.6157034629083403, 0.15873015873015872),
  (0.6478537360890302, 0.15082986593517092),
  (0.6383147853736089, 0.1741080089409874),
  (0.6664335859649221, 0.18342151675485008),
  (0.6869213994505023, 0.1437389770723104),
  (0.7036521716531111, 0.0)]

    create_compl_islands(scale_to_map(island_array, -350, 350),
                        islands)

    # Checks which grid points are in an island. Only needs to be executed once.
    taken_points = grid_check(nx, ny, islands, taken_points)
    # Convert taken points to array of actual coordinates on plot
    to_red = [[(x-75)/150 * 1000, (y-75)/150 * 1000] for x, y in taken_points]

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
        num_procs = 8   # Don't use too many or your laptop will explode
        procs = []
        for i in range(num_procs):
            startindex = i * (num_frames // num_procs)
            endindex = (i + 1) * (num_frames // num_procs)
            u_list_section = u_list[startindex:endindex]
            v_list_section = v_list[startindex:endindex]
            title_offset = startindex * 10 / 3600
            p = multiprocessing.Process(
                target=velocity_animation,
                args=(X, Y, u_list_section, v_list_section, 10, f"{VIDEO_NAME}_{i}", islands, to_red, title_offset),
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
                    filename=VIDEO_NAME, taken_points=taken_points, islands=islands)

    print("Animation took {:.2f} seconds".format(time.time() - t))


if __name__ == "__main__":
    main()
