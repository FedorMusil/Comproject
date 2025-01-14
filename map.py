
"""
Will contain code to plot the map and show arrows on the map.
"""

import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.patches import Arrow
from math import ceil
from PIL import Image


WATER_COLOUR = [174, 204, 240]

# TODO: Update the length of arrows when the map is zoomed in/out (or prevent zooming in/out)

def degree_to_x_y_direction(degrees: int) -> tuple:
    """Converts degrees to x and y direction."""
    return np.cos(np.radians(degrees)), np.sin(np.radians(degrees))


class WorldMap:
    def __init__(self, image: str = "world.jpg"):
        self.img = Image.open(image)
        self.img = np.array(self.img)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.img)
        self.current_arrows = []

        # Absolute coords are the coordinates of the corners of the map, in order of top left, top right, bottom left, bottom right.
        self.absolute_coords = [(0, 0), (self.img.shape[1], 0), (0, self.img.shape[0]), (self.img.shape[1], self.img.shape[0])]
        self.absolute_coords_old = [(0, 0), (self.img.shape[1], 0), (0, self.img.shape[0]), (self.img.shape[1], self.img.shape[0])]
        self.fig.canvas.mpl_connect('draw_event', self.update_coords)

        self.setup_arrows(40, 40)
        self.plot_arrows()
        plt.show()

    def setup_arrows(self, amount_x: int = 20, amount_y: int = 20):
        """
        Stores arrows in a list. We will place at most amount_x * amount_y arrows on the map, in a grid.
        Arrows must only show up on water.
        """
        top_left, top_right, bottom_left, bottom_right = self.absolute_coords
        left = top_left[0]
        right = top_right[0]
        top = top_left[1]
        bottom = bottom_left[1]
        self.arrowpos = []
        for x in range(left, right, (right - left) // amount_x):
            for y in range(top, bottom, (bottom - top) // amount_y):
                if np.isclose(list(self.img[y, x]), WATER_COLOUR, atol=5).all():    # use isclose to account for slight variations in colour
                    self.arrowpos.append((x, y))

    def plot_arrows(self):
        """Plots all arrows on the map."""
        # First clear all current arrows
        for arrow in self.current_arrows:
            arrow.remove()
        self.current_arrows = []

        for x, y in self.arrowpos:
            # TODO replace random direction with calculated one (formulas to be implemented)
            # Probably will have to calculate the direction at the current position?
            # Or maybe change to a system to place an arrow wherever the current changes more than a threshold.
            # Maybe change width and colour depending on power of the current, if possible. For now, red will do, though.
            x_dir, y_dir = degree_to_x_y_direction(random.randint(0, 360))
            arrow = Arrow(x, y, x_dir * 40, y_dir * 40, width=5, fc='r', ec='r')
            self.current_arrows.append(arrow)
            self.ax.add_patch(arrow)
        self.fig.canvas.draw_idle()

    def update_coords(self, event):
        """callback for resize event, update coords in case of resize"""
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        x_min, x_max = ceil(x_min), ceil(x_max)
        y_min, y_max = ceil(y_min), ceil(y_max)

        top_left = (x_min, y_max)
        top_right = (x_max, y_max)
        bottom_left = (x_min, y_min)
        bottom_right = (x_max, y_min)
        self.absolute_coords = [top_left, top_right, bottom_left, bottom_right]

        if self.absolute_coords != self.absolute_coords_old:
            self.setup_arrows()
            self.plot_arrows()
            self.absolute_coords_old = self.absolute_coords


if __name__ == "__main__":
    WorldMap()