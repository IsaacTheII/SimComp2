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

from time import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from week2.Particle import Particle
from week2.TreeCode_simp import Cell


def gen_particle(N, type="random"):
    if type == "random":
        return [Particle(np.random.rand(), np.random.rand(), 0, 0, 1) for _ in range(N)]

    if type == "pseudo_coherent":
        particles = []

        s = min(N, np.random.randint(20, 50))
        for _ in range(s):
            x, y = np.random.rand(), np.random.rand()
            for _ in range(N // s):
                particles.append(
                    Particle((x + np.random.normal(scale=1.6) / s) % 1, (y + np.random.normal(scale=1.6) / s) % 1, 0, 0,
                             1))
        return particles


def draw_N_clostest_notupdated(root: Cell, N, ax):
    close_test = root.N_closest(root.particles[0], N)
    x_close_test = []
    y_close_test = []
    for p in close_test:
        x_close_test.append(p[0].r[0])
        y_close_test.append(p[0].r[1])
    ax.scatter(x_close_test, y_close_test, c='orange', alpha=.5, linewidths=5)
    ax.scatter(root.particles[0].r[0], root.particles[0].r[1], c='yellow', alpha=1, linewidths=5)


def draw_ball_walk_notupdated(root: Cell, pos_vec, range, ax):
    range_test = root.ballwalk(np.array(pos_vec), range)
    x_range_test = []
    y_range_test = []
    for p in range_test:
        x_range_test.append(p[0].r[0])
        y_range_test.append(p[0].r[1])
    ax.scatter(x_range_test, y_range_test, c='red', alpha=.5)


def make_new_tree(size, rand_ver):
    particles = gen_particle(size, rand_ver)
    root = Cell([0, 0], [1, 1], 20, None)
    root.insert(particles)
    return root


def make_tree_from_file(filename):
    particles = read_data(filename)
    root = Cell([-5, -5, -0.5], [5, 5, 0.5], 8, None)
    root.insert(particles)
    return root


def read_data(filename: str) -> list:
    data = open("esc202-planetesimals/ESC202-planetesimals.dat", 'r')
    particles = []
    line = data.readline().split()
    while line != []:
        particles.append(Particle(float(line[0]), float(line[1]), float(line[2]),
                                  float(line[3]), float(line[4]), float(line[5]),
                                  float(line[6]), float(line[7])))
        for _ in range(1):
            line = data.readline().split()
    return particles


def main():
    print("MAIN")
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot()
    # ax1.set_xlim(-0.1, 1.1)
    # ax1.set_ylim(-0.1, 1.1)

    root = make_tree_from_file("esc202-planetesimals/ESC202-planetesimals.dat")
    print("start")

    def init():
        return []

    def update(frame):
        print(frame)
        t = time()
        ax1.clear()
        for _ in range(1):
            root.leapfrog(.1, np.pi / 3)
        artist = []
        root.ani_Cells(ax1, artist)
        print(time() - t)
        return artist

    ani = animation.FuncAnimation(fig, update, init_func=init, frames=50)
    ani.save("orbit_full.mp4", fps=10)

    print("END")


if __name__ == '__main__':
    main()
