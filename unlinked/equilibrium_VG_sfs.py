import moments
import numpy as np
import scipy.stats
import sys
import pickle

import matplotlib, matplotlib.pylab as plt

# set font sizes
plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=6)
matplotlib.rc("ytick", labelsize=6)
matplotlib.rc("axes", labelsize=7)
matplotlib.rc("axes", titlesize=7)
matplotlib.rc("legend", fontsize=6)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


a_list = np.logspace(-6, -1, 101)

optimum = 0
VS = 1
mu = 0.01
Ne = 1e4
theta = 4 * Ne * mu

# assumes symmetry +/- effect sizes, and population mean is at optimum of zero
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


def get_gamma(a, VG, Ne, VS=1):
    s = a ** 2 / 2 / (VS + VG)
    gamma = -2 * Ne * s
    return gamma


def init_fs(n, gamma, theta):
    return moments.Spectrum(
        moments.LinearSystem_1D.steady_state_1D(n, overdominance=gamma, theta=theta)
    )


def get_VG(fs, a):
    return fs.pi() * a ** 2


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


if __name__ == "__main__":
    SDs = np.concatenate(([1e-5], np.logspace(-4, -1, 37), [0.2]))
    VGs = [equilibrium_VG(SD) for SD in SDs]
    data = pickle.load(open("unlinked_VGs_array.pkl", "rb"))

    fig = plt.figure(987543, figsize=(4, 3))
    fig.clf()
    ax = plt.subplot(1, 1, 1)
    x = np.logspace(-5, 0, 101)
    ax.plot(
        x ** 2 * Ne, 4 * Ne * mu * x ** 2, "-",
        lw=1., c=colors[2], label="Small effect sizes",
    )
    ax.plot(
        x ** 2 * Ne, 4 * mu * VS * np.ones(len(x)), "-",
        lw=1., c=colors[3], label="House-of-cards approx.",
    )
    ax.plot(
        SDs ** 2 * Ne, VGs, "o",
        lw=1, ms=3.5, markerfacecolor="white", label="moments"
    )
    ax.plot(
        data["SD"] ** 2 * Ne, data["VG"], ".",
        color=colors[1], ms=2, label="Simulations",
    )
    ax.plot(
        x ** 2 * Ne, 4 * mu * VS / (1 + VS / (Ne * x ** 2)), "--",
        # See Walsh & Lynch, E&S of Quant Traits, page 1050, eq 28.30a
        color="k", lw=1, label="Stochastic HOC")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(5e-5, 200)
    ax.set_ylim(1e-6, 1e-1)
    ax.set_xlabel("$N_e \\times V_M$")
    ax.set_ylabel("$V_A$")
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2, 3, 0, 1, 4]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
        handlelength=2.4, frameon=False, title=f"With $\mu={mu}$, $V_S={VS}$")
    ax.text(0.5e-3, 5e-5, "$4N_e\mu V_M$", fontsize=8, color=colors[2], rotation=45)
    ax.text(4e-3, 0.02, "$4\mu V_S$", fontsize=8, color=colors[3])
    #ax.set_title("Lande $\longleftrightarrow$ Turelli")

    fig.tight_layout()
    plt.savefig("../figures/vary_SD.pdf")
    # plt.show()
