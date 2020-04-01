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


class Particle:
    def __init__(self, x, y, z, vx, vy, vz, m, r):
        self.r = np.array([x, y, z])
        self.v = np.array([vx, vy, vz])
        self.a = np.zeros_like(self.v)
        self.c = None
        self.e = None
        self.density = 0.0
        self.n_closest = []
        self.mass = m
        self.radius = r
        self.h = 0.0

    def dist(self, other):
        return np.linal.norm(self.r - other.r)

    def update_vel(self, acc_vec, dt):
        self.v += acc_vec * dt

    def update_pos(self, dt):
        self.r += self.v * dt

    def get_force(self, particle_arr):
        force = np.zeros_like(self.v)
        for p in particle_arr:
            # if p == self:
            #     continue
            force += p.mass * (self.r - p.r) / (np.linalg.norm(self.r - p.r) ** 3 + np.finfo(float).eps)
        return force * self.mass

    def monoghan_kernel(self, other):
        dist = self.dist(other)
        sigma = 8 / np.pi
        sim_h = (self.h + other.h) / 2  # TODO: check if better to symmetries the kernel
        dist_h = dist / sim_h
        if dist_h < .5:
            return sigma / sim_h ** 3 * (6 * dist_h ** 3 - 6 * dist_h ** 2 + 1)
        elif dist_h <= 1:
            return sigma / sim_h ** 3 * 2 * (1 - dist_h) ** 3
        else:
            return 0    # we need to check if dist/h is larger then 1 since we symmetrizing it.
