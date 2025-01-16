
"""
Will contain code to plot the map and show arrows on the map.
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import threading
from matplotlib.patches import Arrow
from math import ceil
from PIL import Image


def degree_to_x_y_direction(degrees: int) -> tuple[float, float]:
    """Converts degrees to x and y direction."""
    degrees += 90
    return np.cos(np.radians(degrees)), np.sin(np.radians(degrees))

def radians_to_x_y_directions(radians: float) -> tuple[float, float]:
    """Converts radians to x and y direction."""
    return np.cos(radians), np.sin(radians)


def point_away_from_point(x1: int, y1: int, x2: int, y2: int) -> tuple[float, float]:
    """Returns the x and y direction of a point away from another point."""
    x_dir = x2 - x1
    y_dir = -(y2 - y1)
    x_dir, y_dir = x_dir / np.sqrt(x_dir ** 2 + y_dir ** 2), y_dir / np.sqrt(x_dir ** 2 + y_dir ** 2)
    return x_dir, y_dir


class WorldMap:
    def __init__(self, image: str = "world.jpg", use_image: bool = False):
        if use_image:
            self.img = Image.open(image)
            self.img = np.array(self.img)
            self.water_colour = [174, 204, 240] # world.jpg water colour
        else:   # If not using an image, create a white image.
            self.img = np.ones((1000, 1000, 3)) * 255
            self.water_colour = [255, 255, 255]
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.img)
        self.current_arrows: list[Arrow] = []
        self.redraw_timer = None

        # Put the square here. objects is an array of arrays, where each object contains its boundaries.
        self.objects = []
        self.objects.append(self.draw_object())

        # Absolute coords are the coordinates of the corners of the map, in order of top left, top right, bottom left, bottom right.
        self.absolute_coords = [(0, 0), (self.img.shape[1], 0), (0, self.img.shape[0]), (self.img.shape[1], self.img.shape[0])]
        # self.fig.canvas.mpl_connect('draw_event', self.on_draw)

        self.setup_arrows()
        self.plot_arrows()
        plt.show()

    def draw_object(self):
        """
        Draw a square in the center of the map. No arrows can be drawn in the square.
        can be changed to make different shapes using a voxel method.
        """
        fig_size = (self.img.shape[0],self.img.shape[1])
        quarter = fig_size[0]/4
        objects_bounds = [(1*quarter, 1*quarter), (3*quarter, 1*quarter), (3*quarter, 3*quarter), (1*quarter, 3*quarter)]

        # draw square on map:
        x = [i[0] for i in objects_bounds]
        y = [i[1] for i in objects_bounds]
        x.append(x[0])
        y.append(y[0])
        plt.plot(x, y)
        
        return objects_bounds

    def setup_arrows(self, amount_x: int = 40, amount_y: int = 40):
        """
        Stores arrows in a list. We will place at most amount_x * amount_y arrows on the map, in a grid.
        Arrows will only show up on water.
        """
        top_left, top_right, bottom_left, bottom_right = self.absolute_coords
        left = top_left[0]
        right = top_right[0]
        top = top_left[1]
        bottom = bottom_left[1]

        self.arrowpos = []
        # Iterate over the grid such that we get amount_x * amount_y arrows at most.
        for x in range(left, right, (right - left) // amount_x):
            for y in range(top, bottom, (bottom - top) // amount_y):
                if x > 0 and y > 0 and x < self.img.shape[1] and y < self.img.shape[0]: # Don't do anything if out of bounds
                    if np.isclose(list(self.img[y, x]), self.water_colour, atol=7).all():    # use isclose to account for slight variations in colour
                        self.arrowpos.append((x, y))

    def check_objects(self, x, y):
        """
        checks if an arrow is going to be drawn in an object. returns true if the arrow intersects an object
        
        Only works for squares/ rectangles at the moment.
        """
        for i in self.objects:
            min_bounds = (i[0][0], i[0][1])
            max_bounds = (i[2][0], i[2][1])
            if (x > min_bounds[0] and x < max_bounds[0]) and (y > min_bounds[1] and y < max_bounds[1]):
                return True
        return False

    def plot_arrows(self):
        """Plots all arrows on the map. Removes old arrows first."""
        for arrow in self.current_arrows:
            try:
                arrow.remove()
            # These errors happen if the arrow is already removed, so we can ignore them.
            # Should only happen when moving the canvas between the redraw timer ending and the arrows updating.
            except ValueError:
                pass
            except NotImplementedError:
                pass
        self.current_arrows = []

        for x, y in self.arrowpos:
            # Make arrows point away from center
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()

            # check object intersection
            if self.check_objects(x, y) == True:
                continue

            mid_x, mid_y = (x_min + x_max) // 2, (y_min + y_max) // 2
            x_dir, y_dir = point_away_from_point(mid_x + 250, mid_y + 250, x, y)
            #x_dir, y_dir = degree_to_x_y_direction(random.randint(0, 360))

            # Adjust arrow size by the size of the map
            x_dir *= (x_max - x_min) / 30
            y_dir *= (y_max - y_min) / 30
            arrow = Arrow(x, y, x_dir, y_dir, width=5, fc='r', ec='r')
            self.current_arrows.append(arrow)
            self.ax.add_patch(arrow)
        self.fig.canvas.draw_idle()

    def on_draw(self, event):
        """
        callback for resize event, redraws arrows after 0.4 seconds of inactivity.
        Should help somewhat with performance when moving the map around.
        """
        if self.redraw_timer is not None:
            self.redraw_timer.cancel()

        self.redraw_timer = threading.Timer(0.4, self.redraw_arrows)
        self.redraw_timer.start()

    def redraw_arrows(self):
        """Redraws arrows on the map. Updates absolute coordinates of current canvas before drawing."""
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        x_min, x_max = ceil(x_min), ceil(x_max)
        y_min, y_max = ceil(y_min), ceil(y_max)

        top_left = (x_min, y_max)
        top_right = (x_max, y_max)
        bottom_left = (x_min, y_min)
        bottom_right = (x_max, y_min)
        new_coords = [top_left, top_right, bottom_left, bottom_right]

        if self.absolute_coords != new_coords:    # Comparison is necessary, otherwise it will loop infinitely.
            self.absolute_coords = new_coords
            self.setup_arrows()
            self.plot_arrows()


if __name__ == "__main__":
    WorldMap()