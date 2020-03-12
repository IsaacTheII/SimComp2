import numpy as np


class Particle:
    def __init__(self, x, y, vx, vy, m):
        self.r = np.array([x, y])
        self.v = np.array([vx, vy])
        self.mass = m

    def distance(self, pos_vec):
        dist_vec = self.r - np.array(pos_vec)
        return np.sqrt(np.dot(dist_vec, dist_vec))
