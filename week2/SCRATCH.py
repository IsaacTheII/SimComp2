#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Name:      Simon Padua
   Email:     simon.padua@uzh.ch
   Date:      15/03/2020
   Course:    
   Semester:  
   Week:      
   Thema:     
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

nx = 5
ny = 5

fig = plt.figure()
plt.axis([0, nx, 0, ny])
ax = plt.gca()
ax.set_aspect(1)


def init():
    # initialize an empty list of cirlces
    return []


def animate(i):
    global ax
    ax.clear()
    plt.axis([0, nx, 0, ny])
    # draw circles, select to color for the circles based on the input argument i.
    someColors = ['r', 'b', 'g', 'm', 'y']
    patches = []
    for x in range(0, nx):
        for y in range(0, ny):
            patches.append(ax.add_patch(plt.Circle((x + 0.005 * i, y + 0.005 * i), 0.45, color=someColors[i % 5])))
    return patches


def do_anim():
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=50, interval=20, blit=True)
    anim.save("test.mp4", fps=10)
    plt.show()


def numpy_add():
    a = np.array([1, 2, 3])
    b = np.array([1, -2, 1])
    for _ in range(2):
        a += b
    print(a)


def main():
    numpy_add()


if __name__ == "__main__":
    main()
