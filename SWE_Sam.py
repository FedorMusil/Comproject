import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing
import os
import subprocess
from matplotlib.animation import FuncAnimation


USE_MULTIPROCESSING = True

# Update function for the shallow water equations
def update(h, u, v, dt, h0, dx, dy, g):
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

# Animation function to create the visualization
def velocity_animation(X, Y, u_list, v_list, frame_interval, filename, title_offset=0):
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
                (num * frame_interval / 3600) + title_offset
            ),
            fontname="serif",
            fontsize=19,
        )
        Q.set_UVC(u[::q_int, ::q_int], v[::q_int, ::q_int])
        return Q,

    anim = FuncAnimation(fig, update_quiver, frames=len(u_list), interval=10, blit=False)
    anim.save(f"{filename}.mp4", fps=60, dpi=200)
    return anim

def main():
    nx, ny = 250, 250
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

    offset_x = 375000
    offset_y = 250000
    eta_n = np.exp(-((X-offset_x)**2/(2*(0.05E+6)**2) + (Y-offset_y)**2/(2*(0.05E+6)**2)))
    h += eta_n  # Superimpose the disturbance
    # Main simulation loop

    u_list = []
    v_list = []
    seconds = 60
    num_frames = 60 * seconds

    t = time.time()
    for _ in range(num_frames):
        max_H = np.max(h + h0)  # Total height for CFL condition
        dt = cfl * dx / np.sqrt(g * max_H)
        h, u, v = update(h, u, v, dt, h0, dx, dy, g)
        u_list.append(u.copy())
        v_list.append(v.copy())
    print("Time taken to math:", time.time() - t)

    t = time.time()

    if USE_MULTIPROCESSING:
        print("Using multiprocessing")
        num_procs = 8
        procs = []
        for i in range(num_procs):
            startindex = i * len(u_list) // num_procs
            endindex = (i + 1) * len(u_list) // num_procs
            u_list_section = u_list[startindex:endindex]
            v_list_section = v_list[startindex:endindex]
            title_offset = startindex * 10 / 3600
            p = multiprocessing.Process(target=velocity_animation, args=(X, Y, u_list_section, v_list_section, 10, f"velocity_field_{i}", title_offset))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        # Stitch videos with ffmpeg
        with open("videos.txt", "w") as f:
            for i in range(num_procs):
                f.write(f"file 'velocity_field_{i}.mp4'\n")
        command = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "videos.txt", "-c", "copy", "velocity_field.mp4"]
        subprocess.run(command)

        # Remove temporary video and text files
        for i in range(num_procs):
            os.remove(f"velocity_field_{i}.mp4")
        os.remove("videos.txt")
    else:
        print("Not using multiprocessing")
        velocity_animation(X, Y, u_list, v_list, 10, "velocity_field")

    print("\nTime taken to animate:", time.time() - t)


if __name__ == "__main__":
    main()
