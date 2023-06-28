import numpy as np
import matplotlib.pylab as plt, matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable


# set font sizes
plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=6)
matplotlib.rc("ytick", labelsize=6)
matplotlib.rc("axes", labelsize=7)
matplotlib.rc("axes", titlesize=7)
matplotlib.rc("legend", fontsize=6)


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def evolve_generation(p, q, D, r, s):
    # with underdominance (s > 0)
    p2 = p - s * p * (1 - p) * (1 - 2 * p)
    q2 = q - s * D * (1 - 2 * p)
    D2 = D - s * D * (1 - 2 * p) ** 2 - r * D
    return p2, q2, D2


def trajectories(p0, q0, D0, generations=1, r=0, s=0):
    p = [p0]
    q = [q0]
    D = [D0]
    for g in range(generations):
        p2, q2, D2 = evolve_generation(p[-1], q[-1], D[-1], r, s)
        p.append(p2)
        q.append(q2)
        D.append(D2)
    return p, q, D


def get_expected_frequency_dip(a, r_max, gens, p0, VS=1):
    rs = np.linspace(0, r_max, 101)
    rs_plot = 100 * np.concatenate((-rs[:0:-1], rs))
    qs = np.zeros(len(rs))
    s = a ** 2 / 2 / VS
    for i, r_dist in enumerate(rs):
        p, q, D = trajectories(p0, p0, p0 - p0 ** 2, generations=gens, r=r_dist, s=s)
        qs[i] = q[-1]
    qs = np.concatenate((qs[:0:-1], qs))
    return rs_plot, qs


if __name__ == "__main__":
    p0 = 0.05
    q0 = p0
    # initially fully linked
    D0 = p0 - p0 * q0

    VS = 1
    gens = 2000

    fig = plt.figure(6, figsize=(6.5, 2.5))
    fig.clf()

    ax1 = plt.subplot(1, 2, 1)
    a = 0.05

    rs, qs = get_expected_frequency_dip(a, 0.01, gens, p0)
    ax1.plot(rs, qs, lw=1.5, label="Deterministic")

    mus = [0.01, 0.0025, 0.001]
    for mu in mus:
        fin = np.load(f"data/introgressed_ancestry.means.a_{a}.mu_{mu}.npz")
        y = fin["arr_0"]
        ax1.plot(rs, y, "-", lw=1, label=f"$\mu={mu}$")

    ax1.legend(frameon=False)
    ax1.set_xlabel("cM")
    ax1.set_ylabel("Introgressed ancestry proportion")
    ax1.set_ylim(0, 0.055)

    ax2 = plt.subplot(1, 2, 2)
    a = 0.02
    rs, qs = get_expected_frequency_dip(a, 0.01, gens, p0)

    ax2.plot(rs, qs, lw=1.5, label="Deterministic")

    mus = [0.01, 0.0025, 0.001]
    for mu in mus:
        fin = np.load(f"data/introgressed_ancestry.means.a_{a}.mu_{mu}.npz")
        y = fin["arr_0"]
        ax2.plot(rs, y, "-", lw=1, label=f"$\mu={mu}$")

    #ax2.legend(frameon=False)
    ax2.set_xlabel("cM")
    ax2.set_ylabel("Introgressed ancestry proportion")
    ax2.set_ylim(0, 0.055)

    fig.tight_layout()
    fig.savefig("linkage_simulation.pdf")
