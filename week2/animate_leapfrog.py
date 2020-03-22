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
from week2.TreeCode_newForce import Cell


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
    root = Cell([-3, -3, -0.5], [3, 3, 0.5], 8, None)
    root.insert(particles)
    return root


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
    particles.append(Particle(.0, .0, .0, .0, .0, .0, 1., float(2e-6)))
    return particles

def read_make_binary_system(filename: str) -> list:
    data = open(filename, 'r')
    particles = []
    line = data.readline().split()
    while line:
        particles.append(Particle(float(line[0]), float(line[1]), float(line[2]),
                                  float(line[3]), float(line[4]), float(line[5]),
                                  float(line[6]), float(line[7])))
        for _ in range(2):
            line = data.readline().split()
    particles.append(Particle(3.0, 3.0, .5, .0, .0, .0, 1., .0))
    data.close()
    data = open(filename, 'r')
    line = data.readline().split()
    line = data.readline().split()
    while line:
        particles.append(Particle(float(line[0])+6, float(line[1])+6, float(line[2])+1,
                                  float(line[3]), float(line[4]), float(line[5]),
                                  float(line[6]), float(line[7])))
        for _ in range(2):
            line = data.readline().split()
    # particles.append(Particle(0.0, 0.0, 1.0, .0, .0, .0, .5, .0))
    data.close()
    return particles


def get_particle_pos(particle_array):
    x, y, z = [], [], []
    for p in particle_array:
        x.append(p.r[0])
        y.append(p.r[1])
        z.append(p.r[2])
    return x, y, z


def main():
    dt = 10
    theta = 12

    print("MAIN")
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot([0], [0], [0], 'o', markersize=5, color='yellow')
    p_plot, = ax.plot([], [], [], 'o', markersize=.5, color='black')
    text = ax.text2D(0.08, 0.08, "day: 0", transform=ax.transAxes, fontsize=20)
    ax.set_xlim(-3., 9.)
    ax.set_ylim(-3., 9.)
    ax.set_zlim(-.5, 1.5)
    ax.set_title("LeapFrog Integration: Theta: {:.1f}, dt: {:d}".format(theta, dt), fontsize=20)
    ax.set_xlabel('x [L]')
    ax.set_ylabel('y [L]')
    ax.set_zlabel('z [L]')

    print("setup tree:")
    t = time()
    particles = read_make_binary_system("esc202-planetesimals/ESC202-planetesimals.dat")
    root = init_tree(particles)
    print("tree finished in ", time() - t)

    print("start")

    def init():
        return []

    def update(frame):
        print(frame)
        t1 = time()
        iterations = 1
        for _ in range(iterations):
            root.leapfrog(particles, dt, theta)
        print("integration ", (time() - t1) / iterations)
        x, y, z = get_particle_pos(particles)
        p_plot.set_data(x, y)
        p_plot.set_3d_properties(z)
        text.set_text("day: %d" % (frame * dt * iterations))
        print(time() - t1)
        return

    ani = animation.FuncAnimation(fig, update, init_func=init, frames=100)
    ani.save("orbit_new_tree_direct_forces.mp4", fps=15)

    print("END")


if __name__ == '__main__':
    main()
