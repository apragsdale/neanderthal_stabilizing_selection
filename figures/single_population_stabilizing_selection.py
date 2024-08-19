import numpy as np
import matplotlib.pylab as plt, matplotlib
import moments
import demes, demesdraw
import scipy.stats
import pickle

# set font sizes
plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=6)
matplotlib.rc("ytick", labelsize=6)
matplotlib.rc("axes", labelsize=7)
matplotlib.rc("axes", titlesize=7)
matplotlib.rc("legend", fontsize=6)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def get_VG(fs, a):
    return fs.pi() * a ** 2


def get_gamma(a, VG, Ne, VS=1):
    s = a ** 2 / 2 / (VS + VG)
    gamma = -2 * Ne * s
    return gamma


def init_fs(n, gamma, theta):
    return moments.Spectrum(
        moments.LinearSystem_1D.steady_state_1D(n, overdominance=gamma, theta=theta)
    )


def get_time_series_VA(a, Ne, NB, T1, T2, VS=1, mu=0.025, n=200, update_sel=False):
    # initial sfs
    gamma = get_gamma(a, 0, Ne, VS=VS)
    theta = 4 * Ne * mu
    fs = init_fs(n, gamma, theta)
    VG = get_VG(fs, a)
    if update_sel:
        gamma = get_gamma(a, VG, Ne, VS=VS)
        fs = init_fs(n, gamma, theta)
        while abs(get_VG(fs, a) - VG) > 1e-8:
            VG = get_VG(fs, a)
            gamma = get_gamma(a, VG, Ne, VS=VS)
            fs = init_fs(n, gamma, theta)
    t = [0]
    VGs = [VG]
    num_slices = 100
    T = T1 / 2 / Ne
    Tslice = T / num_slices
    for i in range(num_slices):
        fs.integrate([NB / Ne], Tslice, overdominance=gamma, theta=theta)
        VGs.append(get_VG(fs, a))
        t.append(t[-1] + Tslice)
        if update_sel:
            gamma = get_gamma(a, VGs[-1], Ne, VS=VS)
    T = T2 / 2 / Ne
    Tslice = T / num_slices
    for i in range(num_slices):
        fs.integrate([1], Tslice, overdominance=gamma, theta=theta)
        VGs.append(get_VG(fs, a))
        t.append(t[-1] + Tslice)
        if update_sel:
            gamma = get_gamma(a, VGs[-1], Ne, VS=VS)

    final_t = t[-1]
    t = [2 * Ne * (final_t - _) for _ in t]
    t.insert(0, 2 * (T1 + T2))
    VGs.insert(0, VGs[0])
    return t, VGs


def gaussian_des(a_list, VG_list, SD):
    assert len(a_list) == len(VG_list)
    S = 0
    aa = np.concatenate(([0], a_list))
    dxs = (aa - np.concatenate(([aa[0]], aa))[:-1]) / 2 + (
        np.concatenate((aa, [aa[-1]]))[1:] - aa
    ) / 2
    weights = scipy.stats.norm.pdf(a_list, scale=SD)
    for v, dx, w in zip(VG_list, dxs[1:], weights):
        S += 2 * v * dx * w
    c = 2 * scipy.stats.norm.cdf(-a_list[-1], scale=SD)
    S += c * VG_list[-1]
    return S


def equilibrium_VG(SD):
    EVG = 4 * mu * VS
    gammas = [get_gamma(a, EVG, Ne, VS=VS) for a in a_list]

    n = 500
    fss = [init_fs(n, gamma, theta) for gamma in gammas]

    last_VG = EVG
    VG_list = [get_VG(fs, a) for a, fs in zip(a_list, fss)]
    VG = gaussian_des(a_list, VG_list, SD)
    while np.abs(last_VG - VG) / VG > 1e-4:
        last_VG = VG
        gammas = [get_gamma(a, VG, Ne, VS=VS) for a in a_list]
        fss = [init_fs(n, gamma, theta) for gamma in gammas]
        VG_list = [get_VG(fs, a) for a, fs in zip(a_list, fss)]
        VG = gaussian_des(a_list, VG_list, SD)
    print(SD, VG)
    return VG


def plot_stabilizing_selection_diagram(ax, VS=1, VG=0.1, optimum=0):
    z = np.linspace(-2, 2, 201)
    w = np.exp(-(z ** 2) / 2 / VS)
    f = np.exp(-(z ** 2) / 2 / VG)
    ax.plot(z, w, color=colors[0], label="Fitness\nfunction")
    ax.plot(z, f, color=colors[1], label="Phenotypic\ndistribution")
    ax.vlines(0, 0, 1, colors="k", linestyles="--", lw=1)
    ax.annotate(
        "",
        xy=(0.43, 0.45),
        xytext=(0, 0.45),
        arrowprops=dict(arrowstyle="->", lw=1),
    )
    ax.annotate(
        "",
        xy=(1.21, 0.5),
        xytext=(0, 0.5),
        arrowprops=dict(arrowstyle="->", lw=1),
    )
    ax.text(0.215, 0.40, "$V_G$", va="center", ha="center", color="k", fontsize=6)
    ax.text(0.605, 0.53, "$V_S$", va="center", ha="center", color="k", fontsize=6)
    ax.set_ylim(bottom=0)
    ax.set_xlim([-2, 2])
    ax.set_xlabel("Phenotype")
    ax.set_xticks([0])
    ax.set_xticklabels(["Optimum"])
    ax.set_yticks([])
    ax.annotate(
        "",
        xy=(-0.6, 0.25),
        xytext=(-1.5, 0.25),
        arrowprops=dict(arrowstyle="->", lw=1),
    )
    ax.annotate(
        "",
        xy=(0.6, 0.25),
        xytext=(1.5, 0.25),
        arrowprops=dict(arrowstyle="->", lw=1),
    )
    ax.text(-1.05, 0.27, "Stabilizing\nselection", ha="center", color="k", fontsize=5)
    ax.text(1.05, 0.27, "Stabilizing\nselection", ha="center", color="k", fontsize=5)
    ax.legend(frameon=False, fontsize=6, handlelength=1.5, loc="upper left")


def plot_bottleneck(ax):
    Ne = 1e4
    NB = 1e3
    T1 = 2 * Ne
    T2 = 2 * Ne
    b = demes.Builder(time_units="generations")
    b.add_deme(
        "Deme",
        epochs=[
            dict(end_time=T1 + T2, start_size=Ne),
            dict(end_time=T2, start_size=NB),
            dict(end_time=0, start_size=Ne),
        ],
    )
    g = b.resolve()
    demesdraw.size_history(g, ax=ax, invert_x=True)
    ax.set_xticks([0, T2, T1 + T2])
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.set_ylabel("Population size", rotation=270, labelpad=8)
    ax.set_xlabel("Time ago (generations)")


def plot_shoc(ax, VS=1):
    Ne = 1e4
    mu = 0.01
    theta = 4 * Ne * mu
    a_list = np.logspace(-6, -1, 101)
    SDs = np.concatenate(([1e-5], np.logspace(-4, -1, 37), [0.2]))
    fname = "data/single-pop-moments-prediction.pkl"
    try:
        VGs = pickle.load(open(fname, "rb"))
    except IOError:
        VGs = [equilibrium_VG(SD) for SD in SDs]
        with open(fname, "wb+") as fout:
            pickle.dump(VGs, fout)

    data = pickle.load(open("../unlinked/unlinked_VGs_array.pkl", "rb"))
    x = np.logspace(-5, 0, 101)
    ax.plot(x ** 2 * Ne, 4 * Ne * mu * x ** 2, ":", color="gray", lw=0.75, zorder=0)
    ax.plot(x ** 2 * Ne, 4 * mu * VS * np.ones(len(x)), ":", color="gray", lw=0.75, zorder=0)
    SHOC = 4 * mu * VS / (1 + VS / Ne / x ** 2)
    ax.plot(
        x ** 2 * Ne,
        SHOC,
        "--",
        color="black",
        lw=1,
        label="Stochastic\nhouse of cards",
        zorder=1,
    )
    ax.scatter(
        SDs ** 2 * Ne,
        VGs,
        s=14,
        marker="o",
        edgecolors=colors[0],
        facecolors="none",
        linewidths=1,
        label="moments",
        zorder=2,
    )
    ax.scatter(
        data["SD"] ** 2 * Ne,
        data["VG"],
        color=colors[1],
        s=3,
        marker="o",
        label="Simulations",
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(5e-5, 200)
    ax.set_ylim(1e-6, 1e-1)
    ax.set_xlabel("$N_e \\times V_M$")
    ax.set_ylabel("$V_G$")
    ax.legend(handlelength=1.5, frameon=False, loc="center right")
    ax.text(5e-4, 5e-5, "$4N_e\\mu V_M$", fontsize=6, color="k", rotation=45)
    ax.text(3e-2, 0.047, "$4\\mu V_S$", fontsize=6, color="k")


def plot_trajectories(ax, axb, VS=1):
    a_vals = [0.1, 0.04, 0.01]
    mu = 0.025
    Ne = 1e4
    NB = 1e3
    T1 = 2 * Ne
    T2 = 2 * Ne
    for a in a_vals:
        t, VGs = get_time_series_VA(a, Ne, NB, T1, T2, mu=mu, update_sel=True)
        VGs = [_ / (4 * mu * VS) for _ in VGs]
        axb.plot(t, VGs, "--", label=f"$a={a}$")
    plt.gca().set_prop_cycle(None)
    for a in a_vals:
        t, VGs = get_time_series_VA(a, Ne, NB, T1, T2, mu=mu, update_sel=False)
        VGs = [_ / (4 * mu * VS) for _ in VGs]
        ax.plot(t, VGs, "-", label=None)
        axb.plot(t, VGs, "-", label=None)
    ax.set_xlabel("Time ago (generations)")
    ax.set_ylim(bottom=0, top=1.3)
    ax.set_ylabel("$V_G$ (in units of $4\\mu V_S$)")
    ax.text(30000, 1.20, f"$a={a_vals[0]}$", ha="center", color=colors[0], fontsize=6)
    ax.text(30000, 0.86, f"$a={a_vals[1]}$", ha="center", color=colors[1], fontsize=6)
    ax.text(30000, 0.15, f"$a={a_vals[2]}$", ha="center", color=colors[2], fontsize=6)
    axb.set_xlabel("Time ago (generations)")
    axb.set_ylim(bottom=0, top=1.5)
    axb.set_ylabel("$V_G$ (in units of $4\\mu V_S$)")
    axb.text(30000, 1.25, f"$a={a_vals[0]}$", ha="center", color=colors[0], fontsize=6)
    axb.text(30000, 0.86, f"$a={a_vals[1]}$", ha="center", color=colors[1], fontsize=6)
    axb.text(30000, 0.15, f"$a={a_vals[2]}$", ha="center", color=colors[2], fontsize=6)


if __name__ == "__main__":
    # stabilizing selection model

    fig1 = plt.figure(1, figsize=(6.5, 2.5))
    fig1.clf()

    ax1a = plt.subplot2grid((1, 2), (0, 0), fig=fig1)
    plot_stabilizing_selection_diagram(ax1a)

    ax1b = plt.subplot2grid((1, 2), (0, 1), fig=fig1)
    plot_shoc(ax1b)

    fig2 = plt.figure(2, figsize=(3.25, 4))
    fig3 = plt.figure(3, figsize=(4, 3))
    fig2.clf()
    fig3.clf()

    ax2a = plt.subplot2grid((5, 1), (0, 0), rowspan=2, fig=fig2)
    plot_bottleneck(ax2a)

    ax2b = plt.subplot2grid((5, 1), (2, 0), rowspan=3, fig=fig2)
    # only plot the non-updated VGs in main fig, supp fig with both
    ax3a = plt.subplot2grid((1, 1), (0, 0), fig=fig3)
    plot_trajectories(ax2b, ax3a)

    ax2b.set_xlim(ax2a.get_xlim())
    ax3a.set_xlim(ax2a.get_xlim())
    ax2b.set_xticks(ax2a.get_xticks())
    ax3a.set_xticks(ax2a.get_xticks())

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig1.subplots_adjust(bottom=0.15, top=0.95)

    fig1.text(0.01, 0.97, "A", fontsize=8, va="center", ha="center")
    fig1.text(0.49, 0.97, "B", fontsize=8, va="center", ha="center")
    fig2.text(0.02, 0.97, "A", fontsize=8, va="center", ha="center")
    fig2.text(0.02, 0.59, "B", fontsize=8, va="center", ha="center")

    fig1.savefig("stab_sel.pdf")
    fig2.savefig("one_pop.pdf")
    fig3.savefig("one_pop_supp.pdf")
