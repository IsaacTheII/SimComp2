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
        self.mass = m
        self.radius = r

    def update_vel(self, acc_vec, dt):
        self.v += np.array(acc_vec) * dt

    def update_pos(self, dt):
        self.r += self.v * dt

    def get_force(self, paritcle_arr):
        force = np.zeros_like(self.v)
        for p in paritcle_arr:
            # if p == self:
            #     continue
            force += p.mass * (self.r - p.r) / (np.linalg.norm(self.r - p.r)**3 + np.finfo(float).eps)
        return force * self.mass
