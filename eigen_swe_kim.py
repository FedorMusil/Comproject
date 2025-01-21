import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow


# using the non-conservative form on wikipedia:
# constants:
g = 9.81 # the acceleration due to gravity
f = 1.16E-4 #  the Coriolis coefficient associated with the Coriolis force. On Earth, f is equal to 2Ω sin(φ), where Ω is the angular rotation rate of the Earth (π/12 radians/hour), and φ is the latitude
k = 0 # the viscous drag coefficient
v = 1 # the kinematic viscosity mm^2/sec

# variables:
u = 0 # velocity in x
v = 0 # velocity in y
H = 5 # mean height of horizontal pressure surface
h = 0 # height deviation of horizontal pressure surface from mean height
b = 0 #  topographical height from a reference D, where b: H(x, y) = D + b(x,y)


class swe:
    def __init__(self):
        self.grid_force = np.zeros((100, 100), dtype=np.float64)
        self.grid_dir = np.zeros((100, 100), dtype=np.float64)
        self.grid = np.array([[np.zeros(2, dtype=np.float64) for _ in range(100)] for _ in range(100)])
        print(self.grid[0][0])

    def loop(self, timesteps):
        "go to the next timestep"
        # direction in degrees

        self.initial_impact((50,51), 5.0)
        print(self.grid[50][50])

        print("start")
        cur_time = 0.0

        plt.ion()
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(right=100)
        self.ax.set_ylim(top=100)
        self.grid[10][10] = np.array([10, 10])
        self.current_arrows = []

        # 10 timesteps
        for _ in range(timesteps):
            print(cur_time)

            plt.title("time = " + str(cur_time))

            self.clear_arrs()

            for x in range(len(self.grid)):
                for y in range(len(self.grid[0])):
                    # teken alleen pijlen die een kracht > 0 hebben

                    if not np.array_equal(self.grid[x][y], np.zeros(2)):
                        arrow = Arrow(x, y, self.grid[x][y][0], self.grid[x][y][1], width=1, fc='r', ec='r')
                        self.current_arrows.append(arrow)
                        self.ax.add_patch(arrow)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            self.update_cell((100,100))

            time.sleep(0.1)
            cur_time += 0.1
        print("done")

    def func1(self):
        # delta u over t

        pass

    def clear_arrs(self):
        for arrow in self.current_arrows:
            try:
                arrow.remove()
            # These errors happen if the arrow is already removed, so we can ignore them.
            # Should only happen when moving the canvas between the redraw timer ending and the arrows updating.
            except ValueError:
                pass
            except NotImplementedError:
                pass


    def angle_check(self, i):
        if i == [1, 0]:
            angle = 0
        elif i == [1, 1]:
            angle = 45
        elif i == [0, 1]:
            angle = 90
        elif i == [-1, 1]:
            angle = 135
        elif i == [-1, 0]:
            angle = 180
        elif i == [-1, -1]:
            angle = 225
        elif i == [0, -1]:
            angle = 270
        elif i == [1, -1]:
            angle = 315

        return angle

    def check_neighbours(self, dims, x, y):
        # eenheidscirkel tegen de klok in

        added_vecs = np.array([0,0], dtype=np.float64)
        mean_num = 0

        for i in [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]:
            # neighbour is out of bounds, so ignore
            if (x + i[0] < 0 or x + i[0] >= dims[0] or y + i[1] < 0 or y + i[1] >= dims[1]) :
                # VERVANG DIT! is nu gehardcoded
                continue

            else:
                vec = self.grid[x+i[0]][y+i[1]]
                if np.array_equal(np.array([0,0]), vec):
                    continue

                dir_vec =  np.array([x, y]) - np.array([x+i[0], y+i[1]])

                v1 = vec / np.linalg.norm(vec)
                v2 = dir_vec / np.linalg.norm(dir_vec)

                res = np.rad2deg(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

                # wanneer het op 90 graden controleerd, blijft het zichzelf actief houden op een of andere manier. (pijlen verdwijnen nooit helemaal)
                # if res > 45 or res < -45:
                if res >= 90 or res <= -90:
                    continue

                added_vecs += vec
                mean_num += 1

        if mean_num != 0:
            return added_vecs/mean_num
        return added_vecs


    def update_cell(self, dims):
        """
        get new direction from own direction and from the influence of the neighbours
        """
        # stores new values, still need old values for calculations
        self.grid[10][10] = np.zeros(2)
        updated_force =  np.copy(self.grid)
        for x in range(dims[0]):
            for y in range(dims[1]):
                updated_force[x][y] = self.check_neighbours(dims, x, y)

        # update whole map
        self.grid = updated_force

    def initial_impact(self, location, force):
        """
        simulate the initial impact
        """

        # update grid_force
        self.grid_force[location[0]][location[1]] = force

        self.grid[location[0]+1][location[1]] = np.array([force, 0])
        self.grid[location[0]-1][location[1]] = np.array([-force, 0])
        self.grid[location[0]][location[1]+1] = np.array([0, force])
        self.grid[location[0]][location[1]-1] = np.array([0, -force])
        self.grid[location[0]+1][location[1]+1] = np.array([np.sin(np.pi/4)*force, np.sin(np.pi/4)*force])
        self.grid[location[0]+1][location[1]-1] = np.array([np.sin(np.pi/4)*force, -np.sin(np.pi/4)*force])
        self.grid[location[0]-1][location[1]+1] = np.array([-np.sin(np.pi/4)*force, np.sin(np.pi/4)*force])
        self.grid[location[0]-1][location[1]-1] = np.array([-np.sin(np.pi/4)*force, -np.sin(np.pi/4)*force])


print(np.rad2deg(np.arctan2(1, 1)))
# loop(10)
swe_test = swe()
swe_test.loop(20)