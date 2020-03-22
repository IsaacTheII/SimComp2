import numpy as np
import matplotlib.pyplot as plt
from time import time
from week1.Particle import Particle
from week1.TreeCode import Cell


def gen_particle(N, type="random"):
    if type == "random":
        return [Particle(np.random.rand(), np.random.rand(), 0, 0, 1) for _ in range(N)]

    if type == "pseudo_coherent":
        particles = []

        s = min(N, np.random.randint(20, 50))
        for _ in range(s):
            x, y = np.random.rand(), np.random.rand()
            for _ in range(N // s):
                particles.append(
                    Particle((x + np.random.normal(scale=1.6) / s) % 1, (y + np.random.normal(scale=1.6) / s) % 1, 0, 0,
                             1))
        return particles


def draw_N_clostest(root: Cell, N, ax):
    close_test, visited = root.N_closest(root.particles[0], N)
    t = time()
    x_close_test = []
    y_close_test = []
    for p in close_test:
        x_close_test.append(p[0].r[0])
        y_close_test.append(p[0].r[1])

    x_visited_test = []
    y_visited_test = []
    for p in visited:
        x_visited_test.append(p.r[0])
        y_visited_test.append(p.r[1])

    ax.scatter(x_visited_test, y_visited_test, c='orange', alpha=.5, linewidths=4)
    ax.scatter(x_close_test, y_close_test, c='green', alpha=.5, linewidths=4)
    ax.scatter(root.particles[0].r[0], root.particles[0].r[1], c='yellow', alpha=1, linewidths=6)
    return t


def draw_ball_walk(root: Cell, pos_vec, range, ax):
    range_test = root.ballwalk(np.array(pos_vec), range)
    t = time()
    x_range_test = []
    y_range_test = []
    for p in range_test:
        x_range_test.append(p[0].r[0])
        y_range_test.append(p[0].r[1])
    ax.scatter(x_range_test, y_range_test, c='red', alpha=1, linewidths=8)
    return t


def make_new_tree(size, rand_ver):
    particles = gen_particle(size, rand_ver)
    root = Cell([0, 0], [1, 1], 4, None)
    root.insert(particles)
    return root


def main():
    print("MAIN")
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot()
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)

    size, n_closest = 1000, 100

    t1 = time()
    root = make_new_tree(size, "pseudo_coherent")
    t1_end = time()
    print("QuadTree with ", len(root), " particles.", "Build in {:.4f} seconds".format(t1_end - t1))
    root.draw_Cells(ax1)

    t2 = time()
    t2_end = draw_ball_walk(root, [.75, .5], .10, ax1)
    print("Marked in red are particles .10 distance from position (.75, .5).",
          "Searched in {:.4f} seconds".format(t2_end - t2))

    t3 = time()
    t3_end = draw_N_clostest(root, n_closest, ax1)
    print("Marked in orange are the visited particles during the search of the ",
          n_closest, " closest particles in green from the yellow marked particle.",
          "Searched in {:.4f} seconds".format(t3_end - t3))

    plt.show()
    # plt.savefig("QTree_Hand-in_3.png")
    print("END")


if __name__ == "__main__":
    main()
