import numpy as np
import matplotlib.pyplot as plt
from week1.Particle import Particle


class Cell:
    def __init__(self, lowerleft, upperright, max_size, parent):
        self.ll = np.array(lowerleft)
        self.ur = np.array(upperright)
        self.mass = 0
        # self.length = 0             TODO: is it faster to have a dedicated variable or to implemnetn len()
        self.max_s = max_size
        self.pivot = (self.ll + self.ur) / 2  # TODO: weighted version
        self.max_radius = np.sqrt(np.dot(self.ur - self.ll, self.ur - self.ll)) / 2
        self.parent = parent
        self.particles = []
        self.isLeaf = True
        self.child = {}             # dic struc is not needed but for future nDim useful TODO: nDim implementation

    def __len__(self):
        return self.particles.__len__()

    def split(self):
        self.isLeaf = False
        leftdown, leftup, rightdown, rightup = [], [], [], []
        for p in self.particles:
            if p.r[0] < self.pivot[0]:
                if p.r[1] < self.pivot[1]:
                    leftdown.append(p)
                else:
                    leftup.append(p)
            else:
                if p.r[1] < self.pivot[1]:
                    rightdown.append(p)
                else:
                    rightup.append(p)
        if leftdown.__len__() > 0:
            self.child["00"] = Cell(self.ll, self.pivot, self.max_s, self).insert(leftdown)
        if leftup.__len__() > 0:
            self.child["01"] = Cell(np.array([self.ll[0], self.pivot[1]]),
                                    np.array([self.pivot[0], self.ur[1]]), self.max_s, self).insert(leftup)
        if rightdown.__len__() > 0:
            self.child["10"] = Cell(np.array([self.pivot[0], self.ll[1]]),
                                    np.array([self.ur[0], self.pivot[1]]), self.max_s, self).insert(rightdown)
        if rightup.__len__() > 0:
            self.child["11"] = Cell(self.pivot, self.ur, self.max_s, self).insert(rightup)

    def insert(self, particles_array):
        for p in particles_array:
            self.mass += p.mass
        self.particles = np.concatenate((self.particles, particles_array))
        if not self.isLeaf:
            leftdown, leftup, rightdown, rightup = [], [], [], []
            for p in self.particles:
                if p.r[0] < self.pivot[0]:
                    if p.r[1] < self.pivot[1]:
                        leftdown.append(p)
                    else:
                        leftup.append(p)
                else:
                    if p.r[1] < self.pivot[1]:
                        rightdown.append(p)
                    else:
                        rightup.append(p)
            if leftdown.__len__() > 0:
                self.child["00"].insert(leftdown)
            if leftup.__len__() > 0:
                self.child["01"].insert(leftup)
            if rightdown.__len__() > 0:
                self.child["10"].insert(rightdown)
            if rightup.__len__() > 0:
                self.child["11"].insert(rightup)

        if (len(self) > self.max_s) & self.isLeaf:
            self.split()
        return self

    def ballwalk(self, pos_vec: np.ndarray, max_dist: float) -> list:
        particles_inrange = []
        diag_dist_vec = self.ur - self.ll
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
        N_closest = [(particle, float('inf')) for _ in range(N+1)]  # N+1 since particle is part of tree
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

    def draw_Cells(self, ax):
        if self.isLeaf:
            x, y = [], []
            for p in self.particles:
                x.append(p.r[0])
                y.append(p.r[1])
            ax.scatter(x, y, c='blue', alpha=.5)
            with_height = self.ur - self.ll
            ax.add_patch(plt.Rectangle((self.ll[0], self.ll[1]), with_height[0], with_height[1], facecolor='none',
                                       edgecolor='lightgreen'))
        else:
            for cell in self.child.values():
                cell.draw_Cells(ax)
        return

    def leapfrog(self):
        # this tree structure is probably to slow anyway check the TreeCode_simp for a better/different approach
        return None

