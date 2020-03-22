#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Name:      Simon Padua
   Email:     simon.padua@uzh.ch
   Date:      12.03.2020
   Course:    ESC202
   Week:      3
   Thema:     Gravity Trees
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt
from Particle import Particle
from TreeCode_newForce import Cell


def gen_particle(N, gen_type="random"):
    if gen_type == "random":
        return [Particle(np.random.rand(), np.random.rand(), 0, 0, 1) for _ in range(N)]

    if gen_type == "pseudo_coherent":
        particles = []

        s = min(N, np.random.randint(20, 50))
        for _ in range(s):
            x, y = np.random.rand(), np.random.rand()
            for _ in range(N // s):
                particles.append(
                    Particle((x + np.random.normal(scale=1.6) / s) % 1, (y + np.random.normal(scale=1.6) / s) % 1, 0, 0,
                             1))
        return particles


"""
def draw_N_clostest(root: Cell, N, ax):
    close_test = root.N_closest(root.particles[0], N)
    x_close_test = []
    y_close_test = []
    for p in close_test:
        x_close_test.append(p[0].r[0])
        y_close_test.append(p[0].r[1])
    ax.scatter(x_close_test, y_close_test, c='orange', alpha=.5, linewidths=5)
    ax.scatter(root.particles[0].r[0], root.particles[0].r[1], c='yellow', alpha=1, linewidths=5)

def draw_ball_walk(root: Cell, pos_vec, range, ax):
    range_test = root.ballwalk(np.array(pos_vec), range)
    x_range_test = []
    y_range_test = []
    for p in range_test:
        x_range_test.append(p[0].r[0])
        y_range_test.append(p[0].r[1])
    ax.scatter(x_range_test, y_range_test, c='red', alpha=.5)
"""


def make_new_tree(size, rand_ver):
    particles = gen_particle(size, rand_ver)
    root = Cell([0, 0], [1, 1], 20, None)
    root.insert(particles)
    return root


def make_tree_from_file(filename, size):
    particles = read_data(filename)
    root = Cell([-3, -3, -0.01], [3, 3, 0.01], size, None)
    root.insert(particles)
    return root


def read_data(filename: str) -> list:
    data = open(filename, 'r')
    particles = []
    line = data.readline().split()
    while line:
        particles.append(Particle(float(line[0]), float(line[1]), float(line[2]),
                                  float(line[3]), float(line[4]), float(line[5]),
                                  float(line[6]), float(line[7])))
        for _ in range(1):
            line = data.readline().split()
    particles.append(Particle(.0, .0, .0, .0, .0, .0, 10., .0))
    return particles

def read_make_binary_system(filename: str) -> list:
    data = open(filename, 'r')
    particles = []
    line = data.readline().split()
    while line:
        particles.append(Particle(float(line[0]), float(line[1]), float(line[2]),
                                  float(line[3]), float(line[4]), float(line[5]),
                                  float(line[6]), float(line[7])))
        for _ in range(1):
            line = data.readline().split()
    particles.append(Particle(.0, .0, .0, .0, .0, .0, 10., .0))
    data.close()
    data = open(filename, 'r')
    line = data.readline().split()
    while line:
        particles.append(Particle(float(line[0]+10), float(line[1]+10), float(line[2]),
                                  float(line[3]), float(line[4]), float(line[5]),
                                  float(line[6]), float(line[7])))
        for _ in range(1):
            line = data.readline().split()
    particles.append(Particle(10.0, 10.0, .0, .0, .0, .0, 10., .0))
    data.close()
    return particles


def init_tree(particles):
    x, y, z = [], [], []
    for p in particles:
        x.append(p.r[0])
        y.append(p.r[1])
        z.append(p.r[2])
    ymin, ymax = min(x), max(x)
    xmin, xmax = min(y), max(y)
    zmin, zmax = min(z), max(z)
    root = Cell([ymin, xmin, zmin], [ymax, xmax, zmax], 8, None)
    root.insert(particles)
    return root

def main_leapfrog_speed():
    print("MAIN")
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot()
    # ax1.set_xlim(-3, 3)
    # ax1.set_ylim(-3, 3)


    particles = read_data("esc202-planetesimals/ESC202-planetesimals.dat")
    root = init_tree(particles)
    print("start")
    t1 = time()
    root.leapfrog(particles, .5, 4)
    root.leapfrog(particles, .5, 4)
    root.leapfrog(particles, .5, 4)
    root.leapfrog(particles, .5, 4)
    root.leapfrog(particles, .5, 4)
    print("integration ", time()/5 - t1/5)
    root.draw_Cells(ax1)
    plt.show()
    print("end ", time() - t1)


def main_Forces():
    print("MAIN")
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot()
    ax1.set_title("Direct Forces")
    ax1.set_ylabel(r'$Force  [\frac{L}{s^{2}}]$')
    ax1.set_xlabel("Radius [L]")
    ax1.set_yscale('log')

    particles = read_data("esc202-planetesimals/ESC202-planetesimals.dat")
    root: Cell
    root = init_tree(particles)
    root.calc_bary_weight()
    x_radius, y_force = [],[]
    for p in particles:
        x_radius.append(np.linalg.norm(p.r))
        y_force.append(np.linalg.norm(root.get_forces(p, 8)))

    print(x_radius, y_force)
    ax1.plot(x_radius, y_force, 'o', c='black', markersize=.5)
    plt.show()
    plt.savefig("Force_Radius.png")
    print("END")


def main_Leaf_Cell_Performance():
    print("MAIN")
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot()
    ax1.set_title("Integration Performance Dependant On Tree Cell Loadings")
    ax1.set_ylabel("time [s]")
    ax1.set_xlabel("Leaf Cell Size [#particles]")
    y_time = []
    x_size = []

    for s in range(2, 20, 2):
        print("start")
        root: Cell
        root = make_tree_from_file("esc202-planetesimals/ESC202-planetesimals.dat", s)
        particles = root.get_particles()
        t1 = time()
        root.leapfrog(particles, .5, 1)
        root.leapfrog(particles, .5, 1)
        root.leapfrog(particles, .5, 1)
        print("integration ", time()/3 - t1/3)
        x_size.append(s)
        y_time.append(time()/3 - t1/3)
        print("end ", time() - t1)
    print(x_size, y_time)
    ax1.plot(x_size, y_time)
    plt.show()
    plt.savefig("Leaf_Size_Analysis.png")

    print("END")


if __name__ == '__main__':
    main_Forces()
