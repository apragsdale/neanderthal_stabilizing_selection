import numpy as np
import matplotlib.pylab as plt, matplotlib
import moments
import demes, demesdraw
import scipy.stats
import os, pickle
import sys


from single_population_stabilizing_selection import *

# set font sizes
plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=6)
matplotlib.rc("ytick", labelsize=6)
matplotlib.rc("axes", labelsize=7)
matplotlib.rc("axes", titlesize=7)
matplotlib.rc("legend", fontsize=6)


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

gen = 25
Ne = 1e4
N_N = 1e3
T_split = 600000

T_H_to_N = 250000
T_N_to_H = 50000
f_H_to_N = 0.05
f_N_to_H = 0.02

T_bottle = 60000
N_bottle = 1000
N_Eur = 20000

b = demes.Builder(time_units="years", generation_time=25)
b.add_deme(
    "Human",
    epochs=[
        dict(end_time=T_bottle, start_size=Ne),
        dict(end_time=0, start_size=N_bottle, end_size=N_Eur),
    ],
)
b.add_deme(
    "Neanderthal",
    start_time=T_split,
    ancestors=["Human"],
    epochs=[dict(end_time=45000, start_size=N_N)],
)
b.add_pulse(
    sources=["Human"], dest="Neanderthal", time=T_H_to_N, proportions=[f_H_to_N]
)
b.add_pulse(
    sources=["Neanderthal"], dest="Human", time=T_N_to_H, proportions=[f_N_to_H]
)
g = b.resolve()


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


def additive_variance_trajectories(
    a_list, VS=1, Ne=1e4, theta=1, SD=0.05, n=100, update_sel=True, EVG=0, save=True
):
    """
    If update sel is True, we update selection coefficients based on current VG.
    If update sel is False, we can specify an expected VG, such as 4 * mu * VS,
    from the house of cards limit.
    """
    fname = f"data/VG_traj.SD_{SD}.human_neand.pkl"
    if os.path.exists(fname):
        with open(fname, "rb") as fin:
            data = pickle.load(fin)
        return data["t"], data["human"], data["neanderthal"]

    gammas = [get_gamma(a, EVG, Ne, VS=VS) for a in a_list]
    ns = []
    for gamma in gammas:
        n_fs = 1 * n
        if gamma < -40:
            n_fs += n
        if gamma < -80:
            n_fs += n
        if gamma < -150:
            n_fs += n
        if gamma < -200:
            print("strong selection... will probably fail")
        ns.append(n_fs)

    fss = [init_fs(4 * n_fs, gamma, theta) for n_fs, gamma in zip(ns, gammas)]
    VGs = [get_VG(fs, a) for fs, a in zip(fss, a_list)]

    if update_sel:
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
    # advance N generations
    for j, fs in enumerate(fss):
        fs.integrate([1], 0.5, theta=theta, overdominance=gammas[j])
    VG = [get_VG(fs, a) for fs, a in zip(fss, a_list)]
    t.append(0.5)
    VG1.append(gaussian_des(a_list, VG, SD))
    VG2.append(gaussian_des(a_list, VG, SD))

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
            if update_sel:
                gamma = [
                    get_gamma(a, VG1[-1], Ne, VS=VS),
                    get_gamma(a, VG2[-1], Ne, VS=VS),
                ]
            else:
                gamma = [gammas[j], gammas[j]]
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
            if update_sel:
                gamma = [
                    get_gamma(a, VG1[-1], Ne, VS=VS),
                    get_gamma(a, VG2[-1], Ne, VS=VS),
                ]
            else:
                gamma = [gammas[j], gammas[j]]
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
            if update_sel:
                gamma = get_gamma(a, VG1[-1], Ne, VS=VS)
            else:
                gamma = gammas[j]
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
            pickle.dump({"t": t, "human": VG1, "neanderthal": VG2, "graph": g}, fout)
    return t, VG1, VG2


if __name__ == "__main__":
    SD = float(sys.argv[1])
    optimum = 0
    VS = 1
    mu = 0.025
    theta = 4 * Ne * mu

    EVG = 4 * mu * VS

    a_list = np.logspace(-3, np.log10(2 * SD))

    # get moments predictions
    t, VG1, VG2 = additive_variance_trajectories(a_list, VS=VS, theta=theta, SD=SD)
