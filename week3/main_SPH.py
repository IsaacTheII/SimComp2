#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Name:      Simon Padua
   Email:     simon.padua@uzh.ch
   Date:      01/04/2020
   Course:    ESC202
   Semester:  
   Week:      3
   Thema:     SPH
"""

import numpy as np
import matplotlib.pyplot as plt
from week3.Particle import Particle
from week3.TreeCode_SPH import Cell


def gen_particle(N, type="random"):
    if type == "random":
        return [
            Particle(np.random.random(3) * [1, 1, 0] * 6 - 3, np.random.random(3) * [1, 1, 0] * 6 - 3, 0.001287, 1e-16)
            for
            _ in range(N)]

    if type == "pseudo_coherent":
        particles = []

        s = min(N, np.random.randint(20, 50))
        for _ in range(s):
            x, y, z = np.random.rand(), np.random.rand(), np.random.rand()
            for _ in range(N // s):
                particles.append(
                    Particle(np.array([(x + np.random.normal(scale=1.6) / s) % 1,
                                       (y + np.random.normal(scale=1.6) / s) % 1,
                                       (y + np.random.normal(scale=1.6) / s) % 1]) * [1, 1, 0],
                             np.array([0, 0, 0]), 0.0001278, 0))
        return particles

    if type == "Sedov-Taylor-Explosion":
        particles = []
        # spaceing = 1 / (N - 1)
        spaceing = 1 / (N)
        Iter = N
        N = N ** 2
        for x in range(Iter):
            for y in range(Iter):
                if x == N / 2 + .5 and y == N / 2 + .5:
                    particles.append(Particle([x * spaceing + spaceing / 2, y * spaceing + spaceing / 2, 0],
                                              [0, 0, 0],
                                              1 / N, .00001275, 100, 2 ** .5))
                else:
                    particles.append(Particle([x * spaceing + spaceing / 2, y * spaceing + spaceing / 2, 0],
                                              [0, 0, 0],
                                              1 / N, .00001275, 1, 2 ** .5))
        return particles


def get_particle_prop(particle_array):
    x, y, z, d = [], [], [], []
    for p in particle_array:
        x.append(p.r[0])
        y.append(p.r[1])
        z.append(p.r[2])
        d.append(p.density)
    return x, y, z, d


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
        particles.append(Particle([float(line[0]), float(line[1]), float(line[2])],
                                  [float(line[3]), float(line[4]), float(line[5])],
                                  float(line[6]), float(line[7])))
        for _ in range(1):
            line = data.readline().split()
    # particles.append(Particle([.0, .0, .0], [.0, .0, .0], 10., .0))
    return particles


def init_tree(particles):
    root = Cell([0.0, 0.0, 0.], [1., 1., 0.], 8, None)
    root.insert(particles)
    return root


def main():
    print("MAIN")
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot()
    # ax1.set_xlim(-0.1, 1.1)
    # ax1.set_ylim(-0.1, 1.1)

    # particles = read_data("esc202-planetesimals/ESC202-planetesimals.dat") # gen_particle(1000, "random") +
    # particles = gen_particle(100000, "pseudo_coherent")
    # particles = gen_particle(1000, "random")
    particles = gen_particle(41, "Sedov-Taylor-Explosion")

    root = init_tree(particles)
    # root.NN_density(particles)
    root.NN_density(particles)

    x, y, z, d = get_particle_prop(particles)
    print(min(d), max(d))
    print(max(d) / min(d))
    print(d)
    ax1.scatter(x, y, c=plt.cm.coolwarm(d / max(d)), s=50)

    # n_closest = root.N_closest(particles[221],4)
    # # n_closest = root.N_closest(particles[221], 20)
    # x, y, z, d = get_particle_prop([p for (p,_) in n_closest])
    # print(min(d), max(d))
    # ax1.scatter(x, y, c='red', linewidths=.2)

    plt.show()
    # plt.savefig("SPH_Density.png")
    print("END")


if __name__ == "__main__":
    main()
