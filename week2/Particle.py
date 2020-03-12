import numpy as np


class Particle:
    def __init__(self, x, y, z, vx, vy, vz, m, r):
        self.r = np.array([x, y, z])
        self.v = np.array([vx, vy, vz])
        self.mass = m
        self.radius = r

    def distance(self, pos_vec):
        dist_vec = self.r - np.array(pos_vec)
        return np.sqrt(np.dot(dist_vec, dist_vec))

    def update_vel(self, acc_vec, dt):
        self.v += np.array(acc_vec) * dt

    def update_pos(self, dt):
        self.r += self.v * dt

    def get_force(self, paritcle_arr):
        force = np.zeros_like(self.v)
        for p in paritcle_arr:
            force += p.mass / self.distance(p.r)**2
        return force * self.mass
