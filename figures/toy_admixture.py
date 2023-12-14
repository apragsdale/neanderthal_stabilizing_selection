import numpy as np
import matplotlib.pylab as plt, matplotlib
import moments
import demes, demesdraw
import scipy.stats
import os, pickle

from single_population_stabilizing_selection import *

# set font sizes
plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=6)
matplotlib.rc("ytick", labelsize=6)
matplotlib.rc("axes", labelsize=7)
matplotlib.rc("axes", titlesize=7)
matplotlib.rc("legend", fontsize=6)


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

Ne = 1e4
N1 = 1e4
N2 = 1e3
T1 = 500000 / 25
T2 = 100000 / 25
f = 0.05
#b = demes.Builder(time_units="years", generation_time=25)
b = demes.Builder(time_units="generations")
b.add_deme("Deme1", epochs=[dict(end_time=0, start_size=N1)])
b.add_deme(
    "Deme2",
    start_time=(T1 + T2),
    ancestors=["Deme1"],
    epochs=[dict(end_time=0, start_size=N2)],
)
b.add_pulse(sources=["Deme2"], dest="Deme1", time=T2, proportions=[f])
g1 = b.resolve()

b.data["pulses"][0]["sources"] = ["Deme1"]
b.data["pulses"][0]["dest"] = "Deme2"
g2 = b.resolve()


# assumes symmetry +/- effect sizes, and population mean is at optimum of zero
def gaussian_des(a_list, VG_list, SD):
    """
    a_list: vector of effect size points
    VG_list: computed VG for given theta from SFS
    SD: standard deviation of the distribution of effect sizes of new mutations
    """
    assert len(a_list) == len(VG_list)
    S = 0
    aa = np.concatenate(([0], a_list))
    dxs = (aa - np.concatenate(([aa[0]], aa))[:-1]) / 2 + (
        np.concatenate((aa, [aa[-1]]))[1:] - aa
    ) / 2
    weights = scipy.stats.norm.pdf(a_list, scale=SD)
    for v, dx, w in zip(VG_list, dxs[1:], weights):
        S += 2 * v * dx * w
    # add strong effects (which we assume contrubute as 4*mu*(VS+VG)
    c = 2 * scipy.stats.norm.cdf(-a_list[-1], scale=SD)
    S += c * 4 * mu * VS
    S /= 1 - c * 4 * mu
    return S


def get_VGs_two_pop(fss, a_list):
    VGs1 = [get_VG(fs.marginalize([1]), a) for fs, a in zip(fss, a_list)]
    VGs2 = [get_VG(fs.marginalize([0]), a) for fs, a in zip(fss, a_list)]
    return VGs1, VGs2


def additive_variance_one_way(
    a_list, VS=1, Ne=1e4, theta=1, SD=0.05, n=100, update_sel=True, EVG=0, admix_dest=1
):
    """
    If update sel is True, we update selection coefficients based on current VG.
    If update sel is False, we can specify an expected VG, such as 4 * mu * VS,
    from the house of cards limit.
    """
    fname = f"data/VG_traj.SD_{SD}.admix_dest_{admix_dest}.pkl"
    if os.path.exists(fname):
        with open(fname, "rb") as fin:
            data = pickle.load(fin)
        return data["t"], data["VG1"], data["VG2"]

    if admix_dest not in [1, 2]:
        raise ValueError("1 implies 2 into 1, 2 implies 1 into 2")

    gammas = [get_gamma(a, EVG, Ne, VS=VS) for a in a_list]
    ns = []
    for gamma in gammas:
        n_fs = 1 * n
        if gamma < -40:
            n_fs += n
        if gamma < -80:
            n_fs += n
        if gamma < -150:
            print("strong selection... will probably fail")
        ns.append(n_fs)

    fss = [init_fs(3 * n_fs, gamma, theta) for n_fs, gamma in zip(ns, gammas)]
    VGs = [get_VG(fs, a) for fs, a in zip(fss, a_list)]

    if update_sel:
        VG = 0
        VG_new = gaussian_des(a_list, VGs, SD)
        while np.abs(VG - VG_new) > 1e-6:
            VG = VG_new
            gammas = [get_gamma(a, VG, Ne, VS=VS) for a in a_list]
            fss = [init_fs(3 * n_fs, gamma, theta) for n_fs, gamma in zip(ns, gammas)]
            VGs = [get_VG(fs, a) for fs, a in zip(fss, a_list)]
            VG_new = gaussian_des(a_list, VGs, SD)

    t = [0]
    VG = [get_VG(fs, a) for fs, a in zip(fss, a_list)]
    VG1 = [gaussian_des(a_list, VG, SD)]
    VG2 = [gaussian_des(a_list, VG, SD)]
    # advance N generations
    for j, fs in enumerate(fss):
        fs.integrate([1], 0.5, theta=theta, overdominance=gammas[j])
    VG = [get_VG(fs, a) for fs, a in zip(fss, a_list)]
    t.append(0.5)
    VG1.append(gaussian_des(a_list, VG, SD))
    VG2.append(gaussian_des(a_list, VG, SD))

    # split population
    for i, fs in enumerate(fss):
        n_fs = fs.sample_sizes[0] // 3
        if admix_dest == 1:
            n1 = n_fs
            n2 = 2 * n_fs
        else:
            n1 = 2 * n_fs
            n2 = n_fs
        fss[i] = fs.split(0, n1, n2)

    num_slices = 100
    T = T1 / 2 / Ne
    T_slice = T / num_slices
    nu = [N1 / Ne, N2 / Ne]
    # pre-admixture
    for i in range(num_slices):
        for j in range(len(fss)):
            a = a_list[j]
            if update_sel:
                gamma = [
                    get_gamma(a, VG1[-1], Ne, VS=VS),
                    get_gamma(a, VG2[-1], Ne, VS=VS),
                ]
            else:
                gamma = [gammas[j], gammas[j]]
            fss[j].integrate(nu, T_slice, theta=theta, overdominance=gamma)
        VGs1, VGs2 = get_VGs_two_pop(fss, a_list)
        VG1.append(gaussian_des(a_list, VGs1, SD))
        VG2.append(gaussian_des(a_list, VGs2, SD))
        t.append(t[-1] + T_slice)
        print(t[-1], VG1[-1], VG2[-1])

    # admixture
    if admix_dest == 1:
        for i, fs in enumerate(fss):
            keep_from = fs.sample_sizes[1] // 2
            fss[i] = fs.pulse_migrate(1, 0, keep_from, f)
    if admix_dest == 2:
        for i, fs in enumerate(fss):
            keep_from = fs.sample_sizes[0] // 2
            fss[i] = fs.pulse_migrate(0, 1, keep_from, f)

    # spike in VGs
    t.append(t[-1])
    VGs1, VGs2 = get_VGs_two_pop(fss, a_list)
    VG1.append(gaussian_des(a_list, VGs1, SD))
    VG2.append(gaussian_des(a_list, VGs2, SD))
    print(t[-1], VG1[-1], VG2[-1])

    # remaining generations
    num_slices = 50
    T = T2 / 2 / Ne
    T_slice = T / num_slices
    for i in range(num_slices):
        for j in range(len(fss)):
            a = a_list[j]
            if update_sel:
                gamma = [
                    get_gamma(a, VG1[-1], Ne, VS=VS),
                    get_gamma(a, VG2[-1], Ne, VS=VS),
                ]
            else:
                gamma = [gammas[j], gammas[j]]
            fss[j].integrate(nu, T_slice, theta=theta, overdominance=gamma)
        VGs1, VGs2 = get_VGs_two_pop(fss, a_list)
        VG1.append(gaussian_des(a_list, VGs1, SD))
        VG2.append(gaussian_des(a_list, VGs2, SD))
        t.append(t[-1] + T_slice)
        print(t[-1], VG1[-1], VG2[-1])

    t, VG1, VG2 = np.array(t), np.array(VG1), np.array(VG2)
    with open(fname, "wb+") as fout:
        pickle.dump({"t": t, "VG1": VG1, "VG2": VG2}, fout)
    return t, VG1, VG2


def load_sim_data(admix_dest):
    data = pickle.load(
        open(f"../unlinked/one_way_simulations/dest_{admix_dest}.rep_0.data.pkl", "rb")
    )
    c = 1
    VGs = data["VGs"]
    VG_by_state = {}
    for k in data["VG_by_state"][0].keys():
        VG_by_state[k] = []
    for gen in sorted(data["VG_by_state"].keys()):
        for k in VG_by_state.keys():
            VG_by_state[k].append(data["VG_by_state"][gen][k])

    for k in VG_by_state.keys():
        VG_by_state[k] = np.array(VG_by_state[k]).T

    for i in range(1, 1000 + 1):
        try:
            data = pickle.load(
                open(
                    f"../unlinked/one_way_simulations/dest_{admix_dest}.rep_{i}.data.pkl",
                    "rb",
                )
            )
            c += 1
            VGs += data["VGs"]
            for gen in sorted(data["VG_by_state"].keys()):
                for k in VG_by_state.keys():
                    VG_by_state[k][:, gen] += data["VG_by_state"][gen][k]
        except IOError:
            break
    print("found", c, "data points")
    VGs /= c
    for k in VG_by_state.keys():
        VG_by_state[k] /= c
    return VGs, VG_by_state


if __name__ == "__main__":
    SD = 0.05
    optimum = 0
    VS = 1
    mu = 0.025
    theta = 4 * Ne * mu

    EVG = 4 * mu * VS

    a_list = np.logspace(-3, np.log10(2 * SD))

    grid = (2, 8)
    fig = plt.figure(3, figsize=(6.5, 3.5))
    fig.clf()

    ## from small to big
    admix_dest = 1

    # plot model
    ax1 = plt.subplot2grid(grid, (0, 0), colspan=2)
    demesdraw.tubes(g1, ax=ax1)
    ax1.set_yticks(np.arange(0, 32000, 4000))
    ax1.set_ylabel("Time ago (generations)")

    # plot simulation averages vs moments prediction
    ax2 = plt.subplot2grid(grid, (0, 2), colspan=3)
    # get moments predictions
    t, VG1, VG2 = additive_variance_one_way(
        a_list,
        VS=VS,
        theta=theta,
        admix_dest=admix_dest,
    )
    t_bp = 2 * Ne * (t - t[-1])
    # get simulation data
    VGs, VG_by_state = load_sim_data(admix_dest)
    t_sim = -np.arange(len(VGs[0]))[::-1]
    ax2.plot(t_bp, VG2, ":", lw=2, c=colors[1], label=None)
    ax2.plot(t_sim, VGs[1], c=colors[1], lw=0.5, label=None)
    ax2.plot(t_bp, VG1, ":", lw=2, c=colors[0], label=None)
    ax2.plot(t_sim, VGs[0], c=colors[0], lw=0.5, label=None)
    ax2.set_xlim(-ax1.get_ylim()[1], 400)
    ax2.set_xticks([-24000, -16000, -8000, 0])
    ax2.set_xticklabels([24000, 16000, 8000, 0])

    # for the legend
    xlim = ax2.get_xlim()
    ylim = ax2.get_ylim()
    ax2.plot((0, 1), (-1, -1), "k:", lw=2, label="moments")
    ax2.plot((0, 1), (-1, -1), "k", lw=0.5, label="simulation")
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.legend(fontsize=6, frameon=False)
    ax2.set_ylabel("$V_G$")
    ax2.text(-16000, 0.1, "Deme 1", fontsize=6, color=colors[0], va="center", ha="center")
    ax2.text(
        -10000, 0.08, "Deme 2", fontsize=6, color=colors[1], va="center", ha="center"
    )
    #ax2.set_xlabel("Time ago (gens.)")
    ax2.set_title("Total genetic variance")

    ax2b = plt.subplot2grid(grid, (0, 5), colspan=3)
    with open("data/VG_traj.part.SD_0.05.pkl", "rb") as fin:
        data = pickle.load(fin)
    ax2b.plot(data[1]["t"] / 25 * 1000, data[1]["tot"], label="Total")
    ax2b.plot(data[1]["t"] / 25 * 1000, data[1]["seg"], label="Segregating")
    ax2b.plot(data[1]["t"] / 25 * 1000, data[1]["lost"], label="Introgressed (der.)")
    ax2b.plot(data[1]["t"] / 25 * 1000, data[1]["fixed"], label="Introgressed (anc.)")
    ax2b.plot(data[1]["t"] / 25 * 1000, data[1]["new"], label="New mutations")
    #ax2b.set_xlabel("Time since admixture (gens.)")
    ax2b.set_title("Segregating, introgressed, and new variants")
    ax2b.set_ylabel("$V_G$")

    ## big to small
    admix_dest = 2
    ax3 = plt.subplot2grid(grid, (1, 0), colspan=2)
    demesdraw.tubes(g2, ax=ax3)
    ax3.set_yticks(np.arange(0, 32000, 4000))
    ax3.set_ylabel("Time ago (generations)")

    ax4 = plt.subplot2grid(grid, (1, 2), colspan=3)
    # get moments predictions
    t, VG1, VG2 = additive_variance_one_way(
        a_list,
        VS=VS,
        theta=theta,
        admix_dest=admix_dest,
    )
    t_bp = 2 * Ne * (t - t[-1])
    # get simulation data
    VGs, VG_by_state = load_sim_data(admix_dest)
    t_sim = -np.arange(len(VGs[0]))[::-1]
    ax4.plot(t_bp, VG2, ":", lw=2, c=colors[1])
    ax4.plot(t_sim, VGs[1], c=colors[1], lw=0.5)
    ax4.plot(t_bp, VG1, ":", lw=2, c=colors[0])
    ax4.plot(t_sim, VGs[0], c=colors[0], lw=0.5)
    ax4.set_xlim(-ax1.get_ylim()[1], 10)
    ax4.set_ylabel("$V_G$")
    ax4.set_xlabel("Time ago (gens.)")
    ax4.set_xticks([-24000, -16000, -8000, 0])
    ax4.set_xticklabels([24000, 16000, 8000, 0])
    ax4.set_ylim(ax2.get_ylim())

    ax4b = plt.subplot2grid(grid, (1, 5), colspan=3)
    with open("data/VG_traj.part.SD_0.05.pkl", "rb") as fin:
        data = pickle.load(fin)
    ax4b.plot(data[2]["t"] / 25 * 1000, data[2]["tot"], label="Total")
    ax4b.plot(data[2]["t"] / 25 * 1000, data[2]["seg"], label="Seg.")
    ax4b.plot(data[2]["t"] / 25 * 1000, data[2]["lost"], label="Intro. (der.)")
    ax4b.plot(data[2]["t"] / 25 * 1000, data[2]["fixed"], label="Intro. (anc.)")
    ax4b.plot(data[2]["t"] / 25 * 1000, data[2]["new"], label="New mut.")
    ax4b.set_xlabel("Time since admixture (gens.)")
    ax4b.legend(handlelength=1, loc=1, ncols=2, fontsize=5, frameon=False)
    ax4b.set_ylabel("$V_G$")
    ax4b.set_ylim(top=ax2b.get_ylim()[1])

    fig.text(0.01, 0.97, "A", fontsize=8, va="center", ha="center")
    fig.text(0.28, 0.97, "C", fontsize=8, va="center", ha="center")
    fig.text(0.63, 0.97, "E", fontsize=8, va="center", ha="center")
    fig.text(0.01, 0.49, "B", fontsize=8, va="center", ha="center")
    fig.text(0.28, 0.49, "D", fontsize=8, va="center", ha="center")
    fig.text(0.63, 0.49, "F", fontsize=8, va="center", ha="center")
    plt.tight_layout()
    plt.subplots_adjust(wspace=3, hspace=0.3, top=0.95, bottom=0.1, left=0.09, right=0.97)
    plt.savefig("reciprocal_admixture.pdf")
    # plt.show()
