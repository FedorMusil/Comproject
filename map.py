
"""
Will contain code to plot the map and show arrows on the map.
"""

import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.patches import Arrow
from PIL import Image


WATER_COLOUR = [174, 204, 240]


def degree_to_x_y_direction(degrees: int) -> tuple:
    """Converts degrees to x and y direction."""
    return np.cos(np.radians(degrees)), np.sin(np.radians(degrees))


class WorldMap:
    def __init__(self):
        self.img = Image.open("world.jpg")
        self.img = np.array(self.img)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.img)

        self.setup_arrows(40, 40)
        self.plot_arrows()
        plt.show()

    def setup_arrows(self, amount_x: int = 20, amount_y: int = 20):
        """
        Stores arrows in a list. We will place at most amount_x * amount_y arrows on the map, in a grid.
        Arrows must only show up on water.
        """
        self.arrowpos = []
        for x in range(amount_x):
            for y in range(amount_y):
                new_x = x * self.img.shape[1] // amount_x
                new_y = y * self.img.shape[0] // amount_y
                if list(self.img[new_y, new_x]) == WATER_COLOUR:
                    self.arrowpos.append((new_x, new_y))

    def plot_arrows(self):
        """Plots all arrows on the map."""
        for x, y in self.arrowpos:
            # TODO replace random direction with calculated one (formulas to be implemented)
            x_dir, y_dir = degree_to_x_y_direction(random.randint(0, 360))
            arrow = Arrow(x, y, x_dir * 20, y_dir * 20, width=5, fc='r', ec='r')
            self.ax.add_patch(arrow)
        self.fig.canvas.draw()


if __name__ == "__main__":
    WorldMap()