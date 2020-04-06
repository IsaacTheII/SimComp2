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
    def __init__(self, pos_vec, vel_vec, m, r, _e=None, _c=None):
        self.r = np.array(pos_vec)
        self.v = np.array(vel_vec)
        self.v_pred = np.array(vel_vec)
        self.a = np.zeros_like(self.v)
        self.e = _e
        self.e_pred = _e
        self.e_dot = 0.0
        self.c = _c
        self.density = 0.0
        self.n_closest = []
        self.mass = m
        self.radius = r
        self.h = 0.0

    def dist(self, other):
        dist = self.r - other.r
        return np.sqrt(dist.dot(dist))
        # return np.linalg.norm(self.r - other.r)

    def update_vel(self, acc_vec, dt):
        self.v += acc_vec * dt

    def update_pos(self, dt):
        self.r += self.v * dt
        self.r %= 1 + np.finfo(float).eps

    def get_force(self, particle_arr):
        force = np.zeros_like(self.v)
        for p in particle_arr:
            # if p == self:
            #     continue
            force += p.mass * (self.r - p.r) / (np.linalg.norm(self.r - p.r) ** 3 + np.finfo(float).eps)
        return force * self.mass

    def monoghan_kernel(self, other, dist):
        # dist = self.dist(other)
        sigma = 8 / np.pi
        # sim_h = (self.h + other.h) / 2  # TODO: check if better to symmetries the kernel
        dist_h = dist / self.h
        if dist_h < .5:
            return sigma / self.h ** 3 * (6 * dist_h ** 3 - 6 * dist_h ** 2 + 1)
        elif dist_h <= 1:
            return sigma / self.h ** 3 * 2 * (1 - dist_h) ** 3
        else:
            return 0  # we need to check if dist/h is larger then 1 since we symmetrizing it.

    def viscosity(self, other):
        alpha, beta = 1, 2
        v_r = (self.v - other.v) * (self.r - other.r)
        u_ab = (self.h + other.h) / 2 * v_r
