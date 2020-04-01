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
from Particle import Particle
from time import time


class Cell:
    def __init__(self, lowerleft, upperright, max_size, parent):
        self.bot = np.array(lowerleft)
        self.top = np.array(upperright)
        self.mass = 0
        self.max_s = max_size
        self.max_radius = np.linalg.norm(self.top - self.bot) / 2
        self.parent = parent
        self.particles = []
        self.isLeaf = True
        self.pivot: float
        self.split_Dim: int
        self.has_bary_weight = False  # probalby better to just calc for all branches since every cell will
        self.center = None  # need to be calc at some point.
        self.child_bot: Cell
        self.child_top: Cell

    def __len__(self):
        return self.particles.__len__()

    def get_particles(self):
        particles_array = []
        cell_queue = [self]
        while cell_queue:
            cell = cell_queue.pop()
            if cell.isLeaf:
                particles_array += cell.particles
            else:
                cell_queue += [cell.child_bot, cell.child_top]
        return particles_array

    def calc_bary_weight(self):
        if self.isLeaf:
            self.mass = 0
            self.center, i = float(0), int(0)
            p: Particle
            for p in self.particles:
                self.mass += p.mass
                self.center += p.r * p.mass
            if self.mass == 0:
                print("error")
                pass
            self.center /= self.mass
        else:
            bot_mass, bot_center = self.child_bot.calc_bary_weight()
            top_mass, top_center = self.child_top.calc_bary_weight()
            self.mass = bot_mass + top_mass
            self.center = (bot_center * bot_mass + top_center * top_mass) / self.mass
        return self.mass, self.center

    def split(self, particle_arr):
        self.isLeaf = False
        self.split_Dim = np.argmax((self.top - self.bot) ** 2)

        self.pivot, i = float(0), int(0)
        for p in particle_arr:
            i += 1
            self.pivot += p.r[self.split_Dim]
        self.pivot /= i

        particle_bot, particle_top = [], []
        for p in particle_arr:
            if p.r[self.split_Dim] < self.pivot:
                particle_bot.append(p)
            else:
                particle_top.append(p)
        # only works if insert return self
        new_upper = np.copy(self.top)
        new_upper.put(self.split_Dim, self.pivot)
        new_lower = np.copy(self.bot)
        new_lower.put(self.split_Dim, self.pivot)
        self.child_bot = Cell(self.bot, new_upper, self.max_s, self).insert(particle_bot)
        self.child_top = Cell(new_lower, self.top, self.max_s, self).insert(particle_top)

    def insert(self, particles_array):
        if len(particles_array) > self.max_s:
            self.split(particles_array)
        else:
            self.isLeaf = True
            self.particles = particles_array
        return self

    def reorder_tree(self):
        particles_array = self.get_particles()
        self.insert(particles_array)
        return self

    # this code is not yet update to the new tree structure
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
            self.child_bot.draw_Cells(ax)
            self.child_top.draw_Cells(ax)
        return

    def ani_Cells(self, ax, artist):
        if self.isLeaf:
            x, y = [], []
            for p in self.particles:
                x.append(p.r[0])
                y.append(p.r[1])
            artist.append(ax.scatter(x, y, c='blue', alpha=.5))
            # with_height = self.top - self.bot
            # artist.append(
            #     ax.add_patch(plt.Rectangle((self.bot[0], self.bot[1]), with_height[0], with_height[1], facecolor='none',
            #                                edgecolor='lightgreen')))
        else:
            artist.append(self.child_bot.draw_Cells(ax))
            artist.append(self.child_top.draw_Cells(ax))
        return artist

    def get_forces(self, particle, theta):
        # theta should be in rad
        def get_theta(vec_center, vec_corner):
            return np.arccos(np.dot(vec_center, vec_corner) / (np.linalg.norm(vec_center) * np.linalg.norm(vec_corner)))

        # this function assumes that the tree has been weight and barycenter has been determent
        cell_queue = [self]
        force = np.zeros_like(particle.v)
        while cell_queue:
            cell = cell_queue.pop(0)
            # TODO better theta cone check
            # this should perform better. if the box is long and thin towards the direction of the particle
            # TODO verify what subset of corners is enough to guarantee to find the largest theta
            # TODO check if approximation as sphere is faster in the general case.
            # cell_center = (cell.top + cell.bot) / 2
            # t = max(get_theta(cell_center - particle.r, cell.top - particle.r),
            #         get_theta(cell_center - particle.r, np.array([cell.bot[0], cell.top[1], cell.top[2]]) - particle.r),
            #         get_theta(cell_center - particle.r, np.array([cell.top[0], cell.bot[1], cell.top[2]]) - particle.r),
            #         get_theta(cell_center - particle.r, np.array([cell.top[0], cell.top[1], cell.bot[2]]) - particle.r),
            #         get_theta(cell_center - particle.r, cell.bot - particle.r),
            #         get_theta(cell_center - particle.r, np.array([cell.top[0], cell.bot[1], cell.bot[2]]) - particle.r),
            #         get_theta(cell_center - particle.r, np.array([cell.bot[0], cell.top[1], cell.bot[2]]) - particle.r),
            #         get_theta(cell_center - particle.r, np.array([cell.bot[0], cell.bot[1], cell.top[2]]) - particle.r))
            #
            # if theta > 2 * t:
            #     force += cell.mass * particle.mass / np.linalg.norm(cell.bary - particle.r) ** 2

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # the circle approximation is way faster even though it will miss interpret some boxes. exact calculation
            # has to much overhead and cant compensate. To be examined is the performance for unequally distributed
            # systems that may have differently shaped boxes.
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # if theta > 2 * np.arctan(cell.max_radius / np.linalg.norm(((cell.bot + cell.top) / 2) - particle.r)):
            #     force -= cell.mass * particle.mass * (cell.center - particle.r) / np.linalg.norm(
            #         cell.center - particle.r) ** 3

            if theta > self.max_radius / np.linalg.norm(((cell.bot + cell.top) / 2) - particle.r):
                force -= cell.mass * (cell.center - particle.r) / np.linalg.norm(cell.center - particle.r) ** 3

            else:
                if not cell.isLeaf:
                    cell_queue.append(cell.child_bot)
                    cell_queue.append(cell.child_top)
                # if cell.isLeaf:
                #     # force -= particle.get_force(cell.particles)
                #     pass
                # else:
                #     cell_queue.append(cell.child_bot)
                #     cell_queue.append(cell.child_top)
        return force * 0.01720209895 ** 2

    # def leapfrog(self, dt, theta):
    #     # self.re_center_weight()
    #     self.calc_bary_weight()
    #     # map(lambda p: p.update_vel(self.get_forces(p, theta), dt / 2), self.particles)
    #     [p.update_vel(self.get_forces(p, theta), dt / 2) for p in self.particles]
    #     # map(lambda p: p.update_pos(dt), self.particles)
    #     [p.update_pos(dt) for p in self.particles]
    #     # self.re_center_weight()
    #     self.calc_bary_weight()
    #     # map(lambda p: p.update_vel(self.get_forces(p, theta), dt / 2), self.particles)
    #     [p.update_vel(self.get_forces(p, theta), dt / 2) for p in self.particles]
    #     self.reorder_tree()
    #     return

    def leapfrog(self, dt, theta):
        self.calc_bary_weight()
        particles_array = self.get_particles()
        for p in particles_array:
            p.update_vel(self.get_forces(p, theta), dt / 2)
        for p in particles_array:
            p.update_pos(dt)
        self.reorder_tree()
        self.calc_bary_weight()
        for p in particles_array:
            p.update_vel(self.get_forces(p, theta), dt / 2)
        # t = time()
        self.reorder_tree()
        # print("reorder", time()-t)
        return
