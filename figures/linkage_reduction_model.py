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


def plot_split_trajectories(a, r, gens, f, fig, ax, VS=1, legend=True):
    s = a ** 2 / 2 / VS
    # divide subplot to show top and bottom
    divider = make_axes_locatable(ax)
    axb = divider.new_vertical(size="100%", pad=0.1)
    fig.add_axes(axb)

    q0 = f
    # initially fully linked
    D0 = f * (1 - f)
    p, q, D = trajectories(f, q0, D0, generations=gens, r=r, s=a ** 2 / 2 / VS)

    ax.plot(p)
    ax.plot(q)
    ax.plot(D)
    ax.set_ylim(0, f * 1.5)
    ax.set_xlabel("Generations since admixture")

    # initially fully linked
    p, q, D = trajectories(1 - f, q0, -D0, generations=gens, r=r, s=a ** 2 / 2 / VS)
    axb.plot(p, label="Selected\nallele freq.")
    axb.plot(q, label="Introgressed\nallele freq.")
    axb.plot(D, label="LD $(|D|)$")
    axb.set_ylim(1 - f * 1.5, 1)
    if legend:
        axb.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    axb.tick_params(bottom=False, labelbottom=False)
    axb.spines["bottom"].set_visible(False)

    d = 0.015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=axb.transAxes, color="k", clip_on=False)
    axb.plot((-d, +d), (-d, +d), **kwargs, lw=1)  # top-left diagonal
    axb.plot((1 - d, 1 + d), (-d, +d), **kwargs, lw=1)  # top-right diagonal

    kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
    ax.plot((-d, +d), (1 - d, 1 + d), **kwargs, lw=1)  # bottom-left diagonal
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs, lw=1)  # bottom-right diagonal
    axb.set_title(f"$a={a}$, $r={r}$")


def frequency_dip(a_vals, r_max, gens, p0, ax_q, ax_D, VS=1, legend=True):
    rs = np.linspace(0, r_max, 101)
    rs_plot = 100 * np.concatenate((-rs[:0:-1], rs))
    for a in a_vals:
        qs = np.zeros(len(rs))
        Ds = np.zeros(len(rs))
        s = a ** 2 / 2 / VS
        for i, r_dist in enumerate(rs):
            p, q, D = trajectories(
                p0, p0, p0 - p0 ** 2, generations=gens, r=r_dist, s=s
            )
            qs[i] = q[-1]
            Ds[i] = D[-1]
        qs = np.concatenate((qs[:0:-1], qs))
        Ds = np.concatenate((Ds[:0:-1], Ds))
        ax_q.plot(rs_plot, qs, label=f"$a={a}$")
        ax_D.plot(rs_plot, Ds, label=f"$a={a}$")
    if legend:
        ax_q.legend(frameon=False)
        ax_D.legend(frameon=False)
    ax_q.set_ylim(bottom=0, top=p0 * 1.1)
    ax_D.set_ylim(bottom=0)
    ax_q.set_xlabel("Dist. from selected locus (cM)")
    ax_D.set_xlabel("Distance from selected locus (cM)")
    ax_q.set_ylabel("Introgressed ancestry frequency")
    ax_D.set_ylabel("Linkage disequilibrium ($|D|$)")


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


def plot_frequency_dip_sim(ax, a):
    p0 = 0.05
    q0 = p0
    # initially fully linked
    D0 = p0 - p0 * q0

    VS = 1
    gens = 2000

    rs, qs = get_expected_frequency_dip(a, 0.01, gens, p0)
    ax.plot(rs, qs, lw=1.5, label=f"Deterministic")

    mus = [0.001, 0.0025, 0.01]
    for mu in mus:
        fin = np.load(f"data/introgressed_ancestry.means.a_{a}.mu_{mu}.npz")
        y = fin["arr_0"]
        ax.plot(rs, y, "-", lw=1, label=f"$\\mu={mu}$")

    ax.legend(frameon=False, title=f"$a={a}$")
    ax.set_xlabel("Distance from selected locus (cM)")
    ax.set_ylabel("Introgressed ancestry frequency")
    ax.set_ylim(0, p0 * 1.1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])


if __name__ == "__main__":
    f = 0.05
    p0 = 1 - f
    q0 = f
    # initially fully linked
    D0 = f * (1 - f)

    VS = 1
    gens = 2000

    fig1 = plt.figure(5, figsize=(6.5, 4.5))
    fig1.clf()

    fig2 = plt.figure(12345, figsize=(4, 3))
    fig2.clf()

    ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=2, fig=fig1)
    a = 0.05
    r = 1e-3
    p0 = 0.05
    plot_split_trajectories(a, r, gens, p0, fig1, ax1, VS=VS)

    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2, fig=fig1)
    a = 0.02
    r = 1e-2
    p0 = 0.05
    plot_split_trajectories(a, r, gens, p0, fig1, ax2, VS=VS, legend=False)

    ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2, fig=fig1)
    ax4 = plt.subplot2grid((1, 1), (0, 0), fig=fig2)
    frequency_dip([0.01, 0.02, 0.05], 1e-2, gens, 0.05, ax3, ax4)


    ax5 = plt.subplot2grid((2, 6), (1, 0), colspan=3, fig=fig1)
    plot_frequency_dip_sim(ax5, 0.05)
    ax6 = plt.subplot2grid((2, 6), (1, 3), colspan=3, fig=fig1)
    plot_frequency_dip_sim(ax6, 0.02)

    fig1.tight_layout()
    fig2.tight_layout()

    fig1.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.95, wspace=1)
    fig1.text(0.02, 0.98, "A", fontsize=8, va="center", ha="center")
    fig1.text(0.35, 0.98, "B", fontsize=8, va="center", ha="center")
    fig1.text(0.67, 0.98, "C", fontsize=8, va="center", ha="center")
    fig1.text(0.02, 0.48, "D", fontsize=8, va="center", ha="center")
    fig1.text(0.51, 0.48, "E", fontsize=8, va="center", ha="center")

    fig1.savefig("ancestry_reduction.pdf")
    fig2.savefig("supp-LD.pdf")
