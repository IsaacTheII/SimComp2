import numpy as np
import matplotlib.pyplot as plt
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
    close_test = root.N_closest(root.particles[0], N)
    x_close_test = []
    y_close_test = []
    for p in close_test:
        x_close_test.append(p[0].r[0])
        y_close_test.append(p[0].r[1])
    ax.scatter(x_close_test, y_close_test, c='orange', alpha=.5, linewidths=5)
    ax.scatter(root.particles[0].r[0], root.particles[0].r[1], c='yellow', alpha=1, linewidths=5)

def draw_ball_walk(root: Cell, pos_vec, range, ax):
    range_test = root.ballwalk(np.array(pos_vec), range)
    x_range_test = []
    y_range_test = []
    for p in range_test:
        x_range_test.append(p[0].r[0])
        y_range_test.append(p[0].r[1])
    ax.scatter(x_range_test, y_range_test, c='red', alpha=.5)


def make_new_tree(size, rand_ver):
    particles = gen_particle(size, rand_ver)
    root = Cell([0, 0], [1, 1], 20, None)
    root.insert(particles)
    return root

def main():
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot()
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)

    size, n_closest = 10000, 1000

    root = make_new_tree(size, "pseudo_coherent")
    root.draw_Cells(ax1)
    draw_N_clostest(root, n_closest, ax1)
    draw_ball_walk(root, [.75, .5], .25, ax1)

    print("QuadTree with ", len(root), " particles.")
    print("Marked in red are point .25 distance from position (.75, .5).")
    print("Marked in orange are the 1000 clostest points from the yellow marked point.")

    plt.show()
    # plt.savefig("QTree_100000.png")
    print("END")


if __name__ == "__main__":
    main()

