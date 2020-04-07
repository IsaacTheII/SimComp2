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
from week3.Particle import Particle
from copy import copy
from time import time, perf_counter

global gamma
gamma = 5 / 3


class Cell:
    def __init__(self, lowerleft, upperright, max_size, parent):
        self.bot = np.array(lowerleft)
        self.top = np.array(upperright)
        self.mass = 0
        self.max_s = max_size
        self.center_of_volume = (self.bot + self.top) / 2
        diag = self.top - self.bot
        self.max_radius = np.sqrt(diag.dot(diag)) / 2
        self.parent = parent
        self.particles = []
        self.isLeaf = True
        self.pivot: float
        self.split_Dim: int
        self.has_bary_weight = False  # probalby better to just calc for all branches since every cell will
        self.center_of_mass = None  # need to be calc at some point.
        self.child_bot = None
        self.child_top = None

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

    def split(self, particle_arr):
        self.isLeaf = False
        self.split_Dim = np.argmax((self.top - self.bot) ** 2)
        # this can be used to balance the tree. but turns out it comes at heavy o
        # self.pivot, i = float(0), int(0)
        # for p in particle_arr:
        #     i += 1
        #     self.pivot += p.r[self.split_Dim]
        # self.pivot /= i
        self.pivot = ((self.top + self.bot) / 2)[self.split_Dim]

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
            # this modification would allow to have minimal volume in leaf cell without breaking ordering
            # r_stack = np.empty([len(particles_array), 3])
            # for i in range(len(particles_array)):
            #     r_stack[i] = particles_array[i].r
            # t = r_stack.min(axis=0)
            # self.bot = r_stack.min(axis=0)
            # self.top = r_stack.max(axis=0)
        return self

    def reorder_tree(self, particles):
        self = Cell([-0.0000001, -0.000001, 0.], [1.0000001, 1.00000001, 0.], self.max_s, self.parent)
        self.insert(particles)
        return self

    # this code is not yet updated to the new tree structure
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
        dist = np.array(pos_vec) - self.center_of_volume
        return np.sqrt(dist.dot(dist)) - self.max_radius

    def N_closest(self, particle: Particle, N: int):
        def insert_into_sort_list(list_tup, tup):
            for i in range(len(list_tup)):
                if list_tup[i][1] > tup[1]:
                    return list_tup[:i] + [tup] + list_tup[i:]
            return list_tup + [tup]

        sort_cell_queue = [(self, self.cell_min_dist(particle.r))]
        N_closest = [(particle, float('inf')) for _ in range(N + 1)]  # N+1 since particle is part of tree
        # replaced None with paritcle to avoid drawing error. !!! TODO: handle exeptions/invalid inputs
        while N_closest[-1][1] > sort_cell_queue[0][1]:
            (cell, _) = sort_cell_queue.pop(0)
            if cell.isLeaf:
                for p in cell.particles:
                    distance = p.dist(particle)
                    if N_closest[-1][1] > distance:
                        N_closest.pop()
                        N_closest = insert_into_sort_list(N_closest, (p, distance))
            else:
                if cell.child_bot is not None:
                    sort_cell_queue = insert_into_sort_list(sort_cell_queue,
                                                            (cell.child_bot, cell.child_bot.cell_min_dist(particle.r)))
                if cell.child_top is not None:
                    sort_cell_queue = insert_into_sort_list(sort_cell_queue,
                                                            (cell.child_top, cell.child_top.cell_min_dist(particle.r)))

            if len(sort_cell_queue) <= 0:
                break

        return N_closest

    def cell_min_dist_pb(self, pos_vec, off_vec):
        dist = pos_vec - (self.center_of_volume + off_vec)
        return np.sqrt(dist.dot(dist)) - self.max_radius

    def N_closest_periodic_boundary(self, particle: Particle, N: int):
        def insert_into_sort_list(list_tup, tup):
            for i in range(len(list_tup)):
                if list_tup[i][1] > tup[1]:
                    return list_tup[:i] + [tup] + list_tup[i:]
            return list_tup + [tup]

        # sort_cell_queue = [(self, self.cell_min_dist(particle.r))]
        sort_cell_queue = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                # for z in [-1, 0, 1]:
                off_vec = (self.top - self.bot) * [x, y, 0]
                # sort_cell_queue.append((self, self.cell_min_dist_pb(particle.r, off_vec), off_vec))
                sort_cell_queue = insert_into_sort_list(sort_cell_queue,
                                                        (self,
                                                         self.cell_min_dist_pb(particle.r, off_vec),
                                                         off_vec))

        N_closest = [(particle, float('inf')) for _ in range(N + 1)]  # N+1 since particle is part of tree
        # replaced None with paritcle to avoid drawing error. !!! TODO: handle exeptions/invalid inputs
        while (N_closest[-1][1] > sort_cell_queue[0][1]):
            (cell, _, off_vec) = sort_cell_queue.pop(0)
            if cell.isLeaf:
                for p in cell.particles:
                    distance = np.linalg.norm((p.r + off_vec) - particle.r)
                    if N_closest[-1][1] > distance:
                        N_closest.pop()
                        N_closest = insert_into_sort_list(N_closest, (p, distance))
            else:
                if cell.child_bot is not None:
                    sort_cell_queue = insert_into_sort_list(sort_cell_queue,
                                                            (cell.child_bot,
                                                             cell.child_bot.cell_min_dist_pb(particle.r, off_vec),
                                                             off_vec))
                if cell.child_top is not None:
                    sort_cell_queue = insert_into_sort_list(sort_cell_queue,
                                                            (cell.child_top,
                                                             cell.child_top.cell_min_dist_pb(particle.r, off_vec),
                                                             off_vec))

            if len(sort_cell_queue) <= 0:
                break

        return N_closest

    def draw_Cells(self, ax):
        if self.isLeaf:
            x, y = [], []
            for p in self.particles:
                x.append(p.r[0])
                y.append(p.r[1])
            ax.scatter(x, y, c='blue', alpha=.5)
            with_height = self.top - self.bot
            ax.add_patch(plt.Rectangle((self.bot[0], self.bot[1]), with_height[0], with_height[1], facecolor='none',
                                       edgecolor='red'))
        else:
            self.child_bot.draw_Cells(ax)
            self.child_top.draw_Cells(ax)
        return

    def NN_density(self, particle_array):
        p: Particle
        for p in particle_array:
            p.n_closest = self.N_closest_periodic_boundary(p, 64)
            # p.n_closest = self.N_closest(p, 32)
            p.h = p.n_closest[-1][1] * .5
        for p in particle_array:
            p.rho = 0
            for (other, dist) in p.n_closest:
                # test = other.mass * p.monoghan_kernel(other, dist)
                p.rho += other.mass * p.monoghan_kernel(other, dist)

    def calc_sound(self, particle_array):
        p: Particle
        for p in particle_array:
            p.c = np.sqrt(gamma * (gamma - 1) * p.e_pred)

    def NN_SPH_Force(self, particle_array):
        p: Particle
        for p in particle_array:
            P_a = p.c ** 2 / (gamma * p.rho)
            other: Particle
            for (other, dist) in p.n_closest:
                P_b = other.c ** 2 / (gamma * other.rho)
                pi_ab = p.viscosity(other)
                v_a_b = np.sqrt(p.v.dot(p.v)) - np.sqrt(other.v.dot(other.v))
                F_ab = P_a + P_b + pi_ab

                p.a -= .5 * other.mass * F_ab * p.gradient_monoghan_kernel(other, dist)
                other.a += .5 * p.mass + F_ab * other.gradient_monoghan_kernel(p, dist)

                p.e_dot += other.mass * (P_a + pi_ab) * v_a_b * p.gradient_monoghan_kernel(other, dist)
                other.e_dot += p.mass * (P_b + pi_ab) * v_a_b * other.gradient_monoghan_kernel(p, dist)

    def calc_force(self, particle_array):
        self.reorder_tree(particle_array)
        for p in particle_array:
            p.a = np.zeros(3, dtype=float)
            p.e_dot = 0
        self.NN_density(particle_array)
        self.calc_sound(particle_array)
        self.NN_SPH_Force(particle_array)

    def drift_1(self, particle_array, dt):
        p: Particle
        for p in particle_array:
            p.update_pos(dt)
            p.v_pred = p.v + p.a * dt
            p.e_pred = p.e + p.e_dot * dt

    def kick(self, particle_array, dt):
        p: Particle
        for p in particle_array:
            p.update_vel(dt)
            p.update_e(dt)

    def drift_2(self, particle_array, dt):
        p: Particle
        for p in particle_array:
            p.update_pos(dt)

    def SPH_leapfrog(self, particle_array, dt):
        self.drift_1(particle_array, dt)
        self.calc_force(particle_array)
        self.kick(particle_array, dt)
        self.drift_2(particle_array, dt)
