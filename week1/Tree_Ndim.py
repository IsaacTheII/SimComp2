import numpy as np


class Particle:
    def __init__(self, x, y, z, vx=0, vy=0, vz=0, m=0):
        self.r = np.array((x, y, z))
        self.v = np.array((vx, vy, vz))
        self.m = m

    def dist(self, other):
        return np.dot(self.r - other.r)

    def get_m(self):
        return self.m


class QTree:
    def __init__(self,):



class Node:
    def __init__(self, lowbounds, highbounds, dim, particle_arr=[]):
        self.lowbounds = lowbounds
        self.highbounds = highbounds
        self.dim = dim
        self.container = np.array(particle_arr)
        self.isSplit = False
        self.isLeaf = True
        self.branches = {}


    def split(self):
        mid_points = 0.5 * (self.lowbounds + self.highbounds)
        np.random.permutation()
        for p in range(self.container):
            self.branches[p.r >= mid_points].append(p)






"""
def buildTree(node, A, dim):
    if node.isLeaf():
        return
    v = 0.5 (node.rLow[dim] +node.rHigh[dim])
    s = partition2(A, node.iStart, node.iEnd, v, dim)
    if s > node.iEnd:
        rLow = node.rHigh[:]
        rLow[dim] = v
        node.right = Node(None, None, node.rLow, rHigh, node.iStart, s)

    if s < node.iEnd:
        rLow = node.rLow[:]
        rLow[dim] = v
        node.right = Node(None, None, rLow, rHigh, node.istart, s)
        buildTree(node.left, A, nextDim)




def printTree(node, indent):
    strIndent = indent*" "

"""


