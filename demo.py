

import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.patches import Arrow
from PIL import Image

global img
global fig
global ax

def plot_img(img):
    global fig
    global ax
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


def main():
    global img
    img = Image.open("world.jpg")
    img = np.array(img.resize((img.size[0] // 2, img.size[1] // 2)))
    plot_img(img)


def onclick(event):
    global img  # Usually this is a function parameter, but matplotlib does not do nicely with that.
    global fig
    global ax
    if event.xdata is None or event.ydata is None:
        return
    if list(img[int(event.ydata), int(event.xdata)]) == [174, 204, 240]:    # if click on water
        arrow = Arrow(event.xdata, event.ydata, random.random() * 100, random.random() * 100, width=5, fc='r', ec='r')
        a = ax.add_patch(arrow)
        fig.canvas.draw()


if __name__ == "__main__":
    main()
