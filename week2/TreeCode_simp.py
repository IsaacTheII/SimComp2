#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Name:      Padua, Simon
   Email:     simon.padua@uzh.ch
   Date:      11 March, 2020
   Kurs:      ESC202
   Semester:  FS20
   Week:      4
   Thema:     Gravity Tree (Note: code from quad tree but simpler and in strict binary-tree form)
"""

import numpy as np
import matplotlib.pyplot as plt
from week2.Particle import Particle


class Cell:
    def __init__(self, lowerleft, upperright, max_size, parent):
        self.bot = np.array(lowerleft)
        self.top = np.array(upperright)
        self.mass = 0
        # self.length = 0             TODO: is it faster to have a dedicated variable or to implemnetn len()
        self.max_s = max_size
        self.max_radius = np.linalg.norm(self.top - self.bot) / 2
        self.parent = parent
        self.particles = []
        self.isLeaf = True
        self.has_bary_weight = False

    def __len__(self):
        return self.particles.__len__()

    def re_center_weight(self):
        self.has_bary_weight = False
        if not self.isLeaf:
            self.child_bot.re_center_weight()
            self.child_top.re_center_weight()

    def calc_bary_weight(self):
        self.bary = np.zeros_like(self.top)
        self.mass = 0
        for p in self.particles:
            self.bary += p.r * p.mass
            self.mass += p.mass
        self.bary /= self.mass

    def split(self):
        self.isLeaf = False
        self.split_Dim = np.argmax((self.top - self.bot)) ** 2

        self.pivot, i = float(0), int(0)
        for p in self.particles:
            i += 1
            self.pivot += p.r[self.split_Dim]
        self.pivot /= i

        particle_bot, particle_top = [], []
        for p in self.particles:
            if p.r[self.split_Dim] < self.pivot:
                particle_bot.append(p)
            else:
                particle_top.append(p)
        # only works if insert return self
        self.child_bot = Cell(self.bot, np.copy(self.top).put(self.split_Dim, self.pivot)).insert(particle_bot)
        self.child_top = Cell(np.copy(self.bot).put(self.split_Dim, self.pivot), self.top).insert(particle_top)

    def insert(self, particles_array):
        for p in particles_array:
            self.mass += p.mass
        self.particles = np.concatenate((self.particles, particles_array))
        if not self.isLeaf:
            particle_bot, particle_top = [], []
            for p in self.particles:
                if p.r[self.split_Dim] < self.pivot:
                    particle_bot.append(p)
                else:
                    particle_top.append(p)

            self.child_bot.insert(particle_bot)
            self.child_top.insert(particle_top)
        else:
            if (len(self) > self.max_s):
                self.split()
        return self

    """
    def ballwalk(self, pos_vec: np.ndarray, max_dist: float) -> list:
        particles_inrange = []
        diag_dist_vec = self.top - self.bot
        cell_dist_vec = self.pivot - pos_vec  # TODO for weighted tree better to approximate circle as square
        if np.dot(cell_dist_vec, cell_dist_vec) < max_dist + np.dot(diag_dist_vec, diag_dist_vec) * .25:
            if self.isLeaf:
                for p in self.particles:
                    dist_vec = pos_vec - p.r
                    sq_dist = np.dot(dist_vec, dist_vec)
                    if sq_dist < max_dist ** 2:
                        particles_inrange.append([p, np.sqrt(sq_dist)])
                return particles_inrange
            else:
                for cell in self.child.values():
                    particles_inrange += cell.ballwalk(pos_vec, max_dist)
                return particles_inrange
        return []

    def cell_min_dist(self, pos_vec):
        dist = np.array(pos_vec) - self.pivot
        return np.sqrt(np.dot(dist, dist)) - self.max_radius

    def N_closest(self, particle: Particle, N: int):

        def insert_into_sort_list(list_tup, tup):
            for i in range(len(list_tup)):
                if list_tup[i][1] > tup[1]:
                    return list_tup[:i] + [tup] + list_tup[i:]
            return list_tup + [tup]

        sort_cell_queue = [(self, self.cell_min_dist(particle.r))]
        N_closest = [(particle, float('inf')) for _ in range(N + 1)]  # N+1 since particle is part of tree
        # replaced None with paritcle to avoid drawing error. !!! TODO: handle exeptions/invalid inputs
        while (N_closest[-1][1] > sort_cell_queue[0][1]):
            cell_tup = sort_cell_queue.pop(0)
            if not cell_tup[0].isLeaf:
                for cell in cell_tup[0].child.values():
                    sort_cell_queue = insert_into_sort_list(sort_cell_queue, (cell, cell.cell_min_dist(particle.r)))
            else:
                for p in cell_tup[0].particles:
                    distance = p.distance(particle.r)
                    if N_closest[-1][1] > distance:
                        N_closest.pop()
                        N_closest = insert_into_sort_list(N_closest, (p, distance))
            if len(sort_cell_queue) <= 0:
                break
        return N_closest
    """

    def draw_Cells(self, ax):
        if self.isLeaf:
            x, y = [], []
            for p in self.particles:
                x.append(p.r[0])
                y.append(p.r[1])
            ax.scatter(x, y, c='blue', alpha=.5)
            with_height = self.top - self.bot
            ax.add_patch(plt.Rectangle((self.bot[0], self.bot[1]), with_height[0], with_height[1], facecolor='none',
                                       edgecolor='lightgreen'))
        else:
            for cell in self.child.values():
                cell.draw_Cells(ax)
        return

    def get_forces(self, particle, theta):
        # theta should be in rad
        def get_theta(vec_center, vec_corner):
            return np.arccos(np.dot(vec_center, vec_corner) / (np.linalg.norm(vec_center) * np.linalg.norm(vec_corner)))

        # this fucntion assumes that the tree has been weight and barycenters has been determend
        cell_queue = [self]
        force = np.zeros_like(particle)
        while len(cell_queue) > 0:
            cell = cell_queue.pop(0)
            # TODO better theta cone check
            # if theta > 2 * np.arctan(cell.max_radius/np.linalg.norm(((cell.bot + cell.top) / 2) - particle.r)):

            # this should perform better. if the box is long and thin towards the direction of the particle
            # TODO verify what subset of corners is enough to guarantee to find the largest theta
            # TODO check if approximation as sphere is faster in the general case.
            cell_center = (cell.top + cell.bot) / 2
            t = max(get_theta(cell_center - particle.r, cell.top - particle.r),
                    get_theta(cell_center - particle.r, cell.top.put(0, cell.bot[0]) - particle.r),
                    get_theta(cell_center - particle.r, cell.top.put(1, cell.bot[1]) - particle.r),
                    get_theta(cell_center - particle.r, cell.bot.put(2, cell.top[2]) - particle.r),
                    get_theta(cell_center - particle.r, cell.bot - particle.r),
                    get_theta(cell_center - particle.r, cell.bot.put(0, cell.top[0]) - particle.r),
                    get_theta(cell_center - particle.r, cell.bot.put(1, cell.top[1]) - particle.r),
                    get_theta(cell_center - particle.r, cell.top.put(2, cell.bot[2]) - particle.r))

            if theta > 2 * t:
                force += cell.mass * particle.mass / np.linalg.norm(cell.bary - particle.r)**2

            else:
                if cell.isLeaf:
                    force += particle.get_force(cell.particles)
                else:
                    cell_queue.append(cell.child_bot)
                    cell_queue.append(cell.child_top)

    def leapfrog(self, dt):
        self.re_center_weight()

        return
