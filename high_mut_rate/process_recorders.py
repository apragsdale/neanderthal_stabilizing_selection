import gzip
import pickle
import numpy as np
import os
import moments
import scipy.stats
import matplotlib.pylab as plt
import sys

import matplotlib

# set font sizes
plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=6)
matplotlib.rc("ytick", labelsize=6)
matplotlib.rc("axes", labelsize=7)
matplotlib.rc("axes", titlesize=7)
matplotlib.rc("legend", fontsize=6)


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# assumes symmetry +/- effect sizes, and population mean is at optimum of zero
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
    c = 2 * scipy.stats.norm.cdf(-a_list[-1], scale=SD)
    S += c * VG_list[-1]
    return S


def get_VGs_two_pop(fss, a_list):
    VGs1 = [get_VG(fs.marginalize([1]), a) for fs, a in zip(fss, a_list)]
    VGs2 = [get_VG(fs.marginalize([0]), a) for fs, a in zip(fss, a_list)]
    return VGs1, VGs2


def run_neanderthal_model(SD=0.05, mu=0.025, save=True):
    gen = 25
    Ne = 1e4
    N_N = 2e3
    T_split = 600000

    T_H_to_N = 250000
    T_N_to_H = 50000
    f_H_to_N = 0.05
    f_N_to_H = 0.02

    T_bottle = 60000
    N_bottle = 1000
    N_Eur = 20000
    T_N_extinct = 45000

    VS = 1
    # a_list = np.logspace(-3, np.log10(2 * SD))
    a_list = np.logspace(-3, -1)
    theta = 4 * Ne * mu
    optimum = 0
    EVG = 4 * mu * VS

    fname = f"VG_traj.SD_{SD}.mu_{mu}.human_neand.pkl"
    if os.path.exists(fname):
        with open(fname, "rb") as fin:
            data = pickle.load(fin)
        return data["t"], data["human"], data["neanderthal"]

    n = 120

    gammas = [get_gamma(a, EVG, Ne, VS=VS) for a in a_list]
    ns = []
    for gamma in gammas:
        n_fs = 1 * n
        if gamma < -30:
            n_fs += n
        if gamma < -60:
            n_fs += n
        if gamma < -80:
            n_fs += n
        if gamma < -120:
            n_fs += n
            print("strong selection... will probably fail")
        ns.append(n_fs)

    fss = [init_fs(4 * n_fs, gamma, theta) for n_fs, gamma in zip(ns, gammas)]
    VGs = [get_VG(fs, a) for fs, a in zip(fss, a_list)]

    VG = 0
    VG_new = gaussian_des(a_list, VGs, SD)
    while np.abs(VG - VG_new) > 1e-6:
        VG = VG_new
        gammas = [get_gamma(a, VG, Ne, VS=VS) for a in a_list]
        fss = [init_fs(4 * n_fs, gamma, theta) for n_fs, gamma in zip(ns, gammas)]
        VGs = [get_VG(fs, a) for fs, a in zip(fss, a_list)]
        VG_new = gaussian_des(a_list, VGs, SD)

    t = [0]
    VG = [get_VG(fs, a) for fs, a in zip(fss, a_list)]
    VG1 = [gaussian_des(a_list, VG, SD)]
    VG2 = [gaussian_des(a_list, VG, SD)]

    # split population
    for i, fs in enumerate(fss):
        n_fs = fs.sample_sizes[0]
        fss[i] = fs.split(0, n_fs // 2, n_fs // 2)

    # time between split and H->N admixture
    num_slices = 50
    T = (T_split - T_H_to_N) / gen / 2 / Ne
    T_slice = T / num_slices
    nu = [1, N_N / Ne]
    for i in range(num_slices):
        for j in range(len(fss)):
            a = a_list[j]
            gamma = [
                get_gamma(a, VG1[-1], Ne, VS=VS),
                get_gamma(a, VG2[-1], Ne, VS=VS),
            ]
            fss[j].integrate(nu, T_slice, theta=theta, overdominance=gamma)
        VGs1, VGs2 = get_VGs_two_pop(fss, a_list)
        VG1.append(gaussian_des(a_list, VGs1, SD))
        VG2.append(gaussian_des(a_list, VGs2, SD))
        t.append(t[-1] + T_slice)
        print(t[-1], VG1[-1], VG2[-1])

    # H -> N admixture
    for i, fs in enumerate(fss):
        keep_from = fs.sample_sizes[0] // 2
        fss[i] = fs.pulse_migrate(0, 1, keep_from, f_H_to_N)

    # record spike in VGs
    t.append(t[-1])
    VGs1, VGs2 = get_VGs_two_pop(fss, a_list)
    VG1.append(gaussian_des(a_list, VGs1, SD))
    VG2.append(gaussian_des(a_list, VGs2, SD))
    print(t[-1], VG1[-1], VG2[-1])

    # time between H -> N admixture and the Human bottleneck 60ka
    # remaining generations
    num_slices = 50
    T = (T_H_to_N - T_bottle) / gen / 2 / Ne
    T_slice = T / num_slices
    for i in range(num_slices):
        for j in range(len(fss)):
            a = a_list[j]
            gamma = [
                get_gamma(a, VG1[-1], Ne, VS=VS),
                get_gamma(a, VG2[-1], Ne, VS=VS),
            ]
            fss[j].integrate(nu, T_slice, theta=theta, overdominance=gamma)
        VGs1, VGs2 = get_VGs_two_pop(fss, a_list)
        VG1.append(gaussian_des(a_list, VGs1, SD))
        VG2.append(gaussian_des(a_list, VGs2, SD))
        t.append(t[-1] + T_slice)
        print(t[-1], VG1[-1], VG2[-1])

    # time between Human bottleneck and the N -> H admixture
    num_slices = 20
    T = (T_bottle - T_N_to_H) / gen / 2 / Ne
    T_slice = T / num_slices
    r = np.log(N_Eur / N_bottle) / (T_bottle / gen / 2 / Ne)
    N_curr = N_bottle / Ne
    for i in range(num_slices):
        nu_func = lambda t: [N_curr * np.exp(r * T_slice), N_N / Ne]
        for j in range(len(fss)):
            a = a_list[j]
            gamma = [
                get_gamma(a, VG1[-1], Ne, VS=VS),
                get_gamma(a, VG2[-1], Ne, VS=VS),
            ]
            fss[j].integrate(nu_func, T_slice, theta=theta, overdominance=gamma)
        N_curr = nu_func(T_slice)[0]
        VGs1, VGs2 = get_VGs_two_pop(fss, a_list)
        VG1.append(gaussian_des(a_list, VGs1, SD))
        VG2.append(gaussian_des(a_list, VGs2, SD))
        t.append(t[-1] + T_slice)
        print(t[-1], VG1[-1], VG2[-1])

    # admixture from N to H
    for i, fs in enumerate(fss):
        keep_from = fs.sample_sizes[1] // 2
        fss[i] = fs.pulse_migrate(1, 0, keep_from, f_N_to_H)

    # record spike in VGs
    t.append(t[-1])
    VGs1, VGs2 = get_VGs_two_pop(fss, a_list)
    VG1.append(gaussian_des(a_list, VGs1, SD))
    VG2.append(gaussian_des(a_list, VGs2, SD))
    print(t[-1], VG1[-1], VG2[-1])

    # to the neanderthal extinction 45ka
    num_slices = 10
    T = (T_N_to_H - 45000) / gen / 2 / Ne
    T_slice = T / num_slices
    for i in range(num_slices):
        nu_func = lambda t: [N_curr * np.exp(r * T_slice), N_N / Ne]
        for j in range(len(fss)):
            a = a_list[j]
            gamma = [
                get_gamma(a, VG1[-1], Ne, VS=VS),
                get_gamma(a, VG2[-1], Ne, VS=VS),
            ]
            fss[j].integrate(nu_func, T_slice, theta=theta, overdominance=gamma)
        N_curr = nu_func(T_slice)[0]
        VGs1, VGs2 = get_VGs_two_pop(fss, a_list)
        VG1.append(gaussian_des(a_list, VGs1, SD))
        VG2.append(gaussian_des(a_list, VGs2, SD))
        t.append(t[-1] + T_slice)
        print(t[-1], VG1[-1], VG2[-1])

    # neanderthal goes extinct
    for i, fs in enumerate(fss):
        fss[i] = fs.marginalize([1])

    num_slices = 20
    T = 45000 / gen / 2 / Ne
    T_slice = T / num_slices
    for i in range(num_slices):
        nu_func = lambda t: [N_curr * np.exp(r * T_slice)]
        for j in range(len(fss)):
            a = a_list[j]
            gamma = get_gamma(a, VG1[-1], Ne, VS=VS)
            fss[j].integrate(nu_func, T_slice, theta=theta, overdominance=gamma)
        N_curr = nu_func(T_slice)[0]
        VGs1 = [get_VG(fs, a) for fs, a in zip(fss, a_list)]
        VG1.append(gaussian_des(a_list, VGs1, SD))
        VG2.append(np.nan)
        t.append(t[-1] + T_slice)
        print(t[-1], VG1[-1], VG2[-1])

    t, VG1, VG2 = np.array(t), np.array(VG1), np.array(VG2)

    # put t in years in the past
    t -= t[-1]
    t *= -1
    t *= 2 * Ne * gen

    if save:
        with open(fname, "wb+") as fout:
            pickle.dump({"t": t, "human": VG1, "neanderthal": VG2}, fout)
    return t, VG1, VG2


def get_recorder_data(SD=0.05, mu=0.025):
    num_seeds = 1000
    num_found = 0
    for i in range(num_seeds):
        fname = f"recorders/recorder.SD_{SD}.mu_{mu}.seed_{i+1}.pkl.gz"
        try:
            data = pickle.load(gzip.open(fname, "rb"))
        except IOError:
            continue
        if num_found == 0:
            var_n = data["B"]["var_phenotype"]
            var_h = data["A"]["var_phenotype"]
            first_found = False
        else:
            var_n += data["B"]["var_phenotype"]
            var_h += data["A"]["var_phenotype"]
        num_found += 1
    print("found", num_found, "files")
    if num_found == 0:
        return None, None
    var_n /= num_found
    var_h /= num_found
    return var_n, var_h


if __name__ == "__main__":
    mu1 = 0.001
    mu2 = 0.025
    SD = float(sys.argv[1])
    print(f"processing SD={SD}, mus={mu1}, {mu2}")


    fig = plt.figure(1, figsize=(6.5, 3))
    fig.clf()

    ax1 = plt.subplot(1, 2, 1)

    t, VG1, VG2 = run_neanderthal_model(SD=SD, mu=mu1)
    var_n, var_h = get_recorder_data(SD=SD, mu=mu1)

    t_h = (len(var_h) - np.arange(len(var_h)) - 1) * 25
    t_n = (len(var_n) - np.arange(len(var_n)) - 1) * 25 + 45000

    ax1.plot(-t_h, var_h, c=colors[0], alpha=0.75)
    ax1.plot(-t_n, var_n, c=colors[1], alpha=0.75)
    ax1.plot(
        -np.insert(t, 0, 625000), np.insert(VG1, 0, VG1[0]), "-", lw=2, c=colors[0]
    )
    ax1.plot(-t, VG2, "-", c=colors[1], lw=2)
    ax1.set_xlabel("Time ago (years)")
    ax1.set_ylabel("$V_G$")
    ax1.legend(
        [
            "Simulation (human)",
            "Simulation (Neand.)",
            "Prediction (human)",
            "Prediction (Neand.)",
        ]
    )
    ax1.set_title(f"Low mutation rate ($\sigma_M={SD}$, $\mu={mu1}$)")

    ax2 = plt.subplot(1, 2, 2)
    
    t, VG1, VG2 = run_neanderthal_model(SD=SD, mu=mu2)
    var_n, var_h = get_recorder_data(SD=SD, mu=mu2)

    t_h = (len(var_h) - np.arange(len(var_h)) - 1) * 25
    t_n = (len(var_n) - np.arange(len(var_n)) - 1) * 25 + 45000

    ax2.plot(-t_h, var_h, c=colors[0], alpha=0.75)
    ax2.plot(-t_n, var_n, c=colors[1], alpha=0.75)
    ax2.plot(
        -np.insert(t, 0, 625000), np.insert(VG1, 0, VG1[0]), "-", lw=2, c=colors[0]
    )
    ax2.plot(-t, VG2, "-", c=colors[1], lw=2)
    ax2.set_xlabel("Time ago (years)")
    ax2.set_ylabel("$V_G$")
    ax2.set_title(f"High mutation rate ($\sigma_M={SD}$, $\mu={mu2}$)")

    fig.tight_layout()
    fig.savefig(f"model_comparison.SD_{SD}.mu_{mu1}_{mu2}.pdf")
    # plt.show()
