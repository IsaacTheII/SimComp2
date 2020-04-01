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

def main():
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot()
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)

    size, n_closest = 10000, 1000

    root = make_new_tree(size, "pseudo_coherent")
    root.draw_Cells(ax1)
    draw_N_clostest(root, n_closest, ax1)
    draw_ball_walk(root, [.75, .5], .25, ax1)

    print("QuadTree with ", len(root), " particles.")
    print("Marked in red are point .25 distance from position (.75, .5).")
    print("Marked in orange are the 1000 clostest points from the yellow marked point.")

    plt.show()
    # plt.savefig("QTree_100000.png")
    print("END")


if __name__ == "__main__":
    main()

