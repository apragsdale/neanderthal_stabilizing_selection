import numpy as np
import matplotlib.pylab as plt, matplotlib
import moments
import demes, demesdraw
import scipy.stats

from single_population_stabilizing_selection import *

# set font sizes
plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=6)
matplotlib.rc("ytick", labelsize=6)
matplotlib.rc("axes", labelsize=8)
matplotlib.rc("axes", titlesize=8)
matplotlib.rc("legend", fontsize=7)


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

Ne = 1e4
N1 = 1e4
N2 = 1e3
T1 = 500000 / 25
T2 = 100000 / 25
f = 0.05
b = demes.Builder(time_units="generations")
b.add_deme(
    "Ancestral",
    epochs=[dict(end_time=T1 + T2, start_size=Ne)],
)
b.add_deme("Deme1", ancestors=["Ancestral"], epochs=[dict(end_time=0, start_size=N1)])
b.add_deme("Deme2", ancestors=["Ancestral"], epochs=[dict(end_time=0, start_size=N2)])
b.add_pulse(sources=["Deme1"], dest="Deme2", time=T2, proportions=[f])
g1 = b.resolve()

b.data["pulses"][0]["sources"] = ["Deme2"]
b.data["pulses"][0]["dest"] = "Deme1"
g2 = b.resolve()


# assumes symmetry +/- effect sizes, and population mean is at optimum of zero
def gaussian_des(a_list, VG_list, SD, hill=False):
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
    # or just take the point estimate from the last bin?
    c = 2 * scipy.stats.norm.cdf(-a_list[-1], scale=SD)
    if hill:
        S += c * 4 * mu * VS
        S /= 1 - c * 4 * mu
    else:
        S += c * VG_list[-1]
    return S


def get_VGs_two_pop(fss, a_list):
    VGs1 = [get_VG(fs.marginalize([1]), a) for fs, a in zip(fss, a_list)]
    VGs2 = [get_VG(fs.marginalize([0]), a) for fs, a in zip(fss, a_list)]
    return VGs1, VGs2


def additive_variance_one_way(
    a_list, VS=1, Ne=1e4, theta=1, n=100, update_sel=True, EVG=0
):
    """
    If update sel is True, we update selection coefficients based on current VG.
    If update sel is False, we can specify an expected VG, such as 4 * mu * VS,
    from the house of cards limit.
    """

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

    fss = [init_fs(2 * n_fs, gamma, theta) for n_fs, gamma in zip(ns, gammas)]
    VGs = [get_VG(fs, a) for fs, a in zip(fss, a_list)]

    if update_sel:
        VG = 0
        VG_new = gaussian_des(a_list, VGs, SD)
        while np.abs(VG - VG_new) > 1e-6:
            VG = VG_new
            gammas = [get_gamma(a, VG, Ne, VS=VS) for a in a_list]
            fss = [init_fs(2 * n_fs, gamma, theta) for n_fs, gamma in zip(ns, gammas)]
            VGs = [get_VG(fs, a) for fs, a in zip(fss, a_list)]
            VG_new = gaussian_des(a_list, VGs, SD)

    for i, fs in enumerate(fss):
        n_fs = fs.sample_sizes[0] // 2
        fss[i] = fs.split(0, n_fs, n_fs)

    t = [0]
    VGs1, VGs2 = get_VGs_two_pop(fss, a_list)
    VG1 = [gaussian_des(a_list, VGs1, SD)]
    VG2 = [gaussian_des(a_list, VGs2, SD)]
    print(t[-1], VG1[-1], VG2[-1])
    spectra = {t[-1]: fss}

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
        spectra[t[-1]] = fss
        print(t[-1], VG1[-1], VG2[-1])

    return np.array(t), np.array(VG1), np.array(VG2), spectra


def f2(fs):
    return fs.project([1, 1]).sum() - 0.5 * (
        fs.marginalize([0]).pi() + fs.marginalize([1]).pi()
    )


def get_VG_after_admixture(admix_dest, fss, a_list, VS=1, SD=0.05):
    if admix_dest == 1:
        source = 1
        dest = 0
        nu = 1
    elif admix_dest == 2:
        source = 0
        dest = 1
        nu = 0.1
    else:
        raise ValueError("bad admix dest")

    fss_admix = [fs.admix(source, dest, fs.sample_sizes[0], f) for fs in fss]

    fss_lost = []
    fss_seg = []
    fss_fixed = []
    fss_new = []
    for i, fs in enumerate(fss):
        fss_new.append(0 * fs.marginalize([1]))
        fac = np.ones(fs.shape)
        if admix_dest == 1:
            fac[1:] = 0
        else:
            fac[:, 1:] = 0
        fss_lost.append((fs * fac).admix(source, dest, fs.sample_sizes[0], f))
        fac = np.ones(fs.shape)
        if admix_dest == 1:
            fac[:-1] = 0
        else:
            fac[:, :-1] = 0
        fss_fixed.append((fs * fac).admix(source, dest, fs.sample_sizes[0], f))
        fac = np.ones(fs.shape)
        if admix_dest == 1:
            fac[0] = 0
            fac[-1] = 0
        else:
            fac[:, 0] = 0
            fac[:, -1] = 0
        fss_seg.append((fs * fac).admix(source, dest, fs.sample_sizes[0], f))
        assert np.allclose(fss_lost[i] + fss_seg[i] + fss_fixed[i], fss_admix[i])

    VG_lost = [
        gaussian_des(a_list, [get_VG(fs, a) for fs, a in zip(fss_lost, a_list)], SD)
    ]
    VG_seg = [
        gaussian_des(a_list, [get_VG(fs, a) for fs, a in zip(fss_seg, a_list)], SD)
    ]
    VG_fixed = [
        gaussian_des(a_list, [get_VG(fs, a) for fs, a in zip(fss_fixed, a_list)], SD)
    ]
    VG_tot = [
        gaussian_des(a_list, [get_VG(fs, a) for fs, a in zip(fss_admix, a_list)], SD)
    ]
    VG_new = [0]
    num_slices = 100
    T_int = T2 / 2 / Ne
    t = np.linspace(0, T_int, num_slices + 1)
    for i in range(num_slices):
        gammas = [get_gamma(a, VG_tot[-1], Ne, VS=VS) for a in a_list]
        for j, gamma in enumerate(gammas):
            fss_admix[j].integrate(
                [nu], T_int / num_slices, theta=theta, overdominance=gamma
            )
            fss_new[j].integrate(
                [nu], T_int / num_slices, theta=theta, overdominance=gamma
            )
            fss_fixed[j].integrate(
                [nu], T_int / num_slices, theta=0, overdominance=gamma
            )
            fss_seg[j].integrate([nu], T_int / num_slices, theta=0, overdominance=gamma)
            fss_lost[j].integrate(
                [nu], T_int / num_slices, theta=0, overdominance=gamma
            )
        VG_tot.append(
            gaussian_des(
                a_list, [get_VG(fs, a) for fs, a in zip(fss_admix, a_list)], SD
            )
        )
        VG_new.append(
            gaussian_des(a_list, [get_VG(fs, a) for fs, a in zip(fss_new, a_list)], SD)
        )
        VG_lost.append(
            gaussian_des(a_list, [get_VG(fs, a) for fs, a in zip(fss_lost, a_list)], SD)
        )
        VG_seg.append(
            gaussian_des(a_list, [get_VG(fs, a) for fs, a in zip(fss_seg, a_list)], SD)
        )
        VG_fixed.append(
            gaussian_des(
                a_list, [get_VG(fs, a) for fs, a in zip(fss_fixed, a_list)], SD
            )
        )
        print(i, VG_tot[-1], VG_new[-1] + VG_lost[-1] + VG_seg[-1] + VG_fixed[-1])
    return t, np.array(VG_tot), np.array(VG_new), np.array(VG_lost), np.array(VG_seg), np.array(VG_fixed)


if __name__ == "__main__":
    SD = 0.05
    optimum = 0
    VS = 1
    mu = 0.025
    theta = 4 * Ne * mu

    EVG = 4 * mu * VS

    a_list = np.logspace(-3, np.log10(2 * SD))

    tt, VG1, VG2, fss = additive_variance_one_way(
        a_list,
        VS=VS,
        theta=theta,
    )
    t_bp = 2 * Ne * (tt - tt[-1])
    fss0 = [fs.marginalize([1]) for fs in fss]
    fss1 = [fs.marginalize([0]) for fs in fss]
    VG0 = [get_VG(fs, a) for fs, a in zip(fss0, a_list)]
    VG1 = [get_VG(fs, a) for fs, a in zip(fss1, a_list)]

    t1, VG1_tot, VG1_new, VG1_lost, VG1_seg, VG1_fixed = get_VG_after_admixture(
        1, fss, a_list, VS=1, SD=0.05
    )
    t2, VG2_tot, VG2_new, VG2_lost, VG2_seg, VG2_fixed = get_VG_after_admixture(
        2, fss, a_list, VS=1, SD=0.05
    )

    t1 = np.insert(t1, 0, [-0.02, 0])
    t2 = np.insert(t2, 0, [-0.02, 0])
    t1 *= 2 * Ne * 25 / 1000
    t2 *= 2 * Ne * 25 / 1000
    
    VG1_tot = np.insert(VG1_tot, 0, [gaussian_des(a_list, VG0, SD)] * 2)
    VG2_tot = np.insert(VG2_tot, 0, [gaussian_des(a_list, VG1, SD)] * 2)
    VG1_new = np.insert(VG1_new, 0, [np.nan, np.nan])
    VG2_new = np.insert(VG2_new, 0, [np.nan, np.nan])
    VG1_lost = np.insert(VG1_lost, 0, [np.nan, np.nan])
    VG2_lost = np.insert(VG2_lost, 0, [np.nan, np.nan])
    VG1_seg = np.insert(VG1_seg, 0, [np.nan, np.nan])
    VG2_seg = np.insert(VG2_seg, 0, [np.nan, np.nan])
    VG1_fixed = np.insert(VG1_fixed, 0, [np.nan, np.nan])
    VG2_fixed = np.insert(VG2_fixed, 0, [np.nan, np.nan])

    fig = plt.figure(2, figsize=(6.5, 2.5))

    ax1 = plt.subplot(1, 2, 1)

    ax1.plot(t1, VG1_tot, label="$V_A$, total")
    ax1.plot(t1, VG1_seg, label="$V_A$, segregating")
    ax1.plot(t1, VG1_lost, label="$V_A$, introgressed (der)")
    ax1.plot(t1, VG1_fixed, label="$V_A$, introgressed (anc)")
    ax1.plot(t1, VG1_new, label="$V_A$, new mutations")
    ax1.set_ylabel("$V_A$")
    ax1.set_xlabel("Time since admixture (ky)")
    ax1.legend()
    ax1.set_title("Admixture into large deme")

    ax2 = plt.subplot(1, 2, 2)

    ax2.plot(t2, VG2_tot, label="$V_A$, total")
    ax2.plot(t2, VG2_seg, label="$V_A$, segregating")
    ax2.plot(t2, VG2_lost, label="$V_A$, introgressed (der)")
    ax2.plot(t2, VG2_fixed, label="$V_A$, introgressed (anc)")
    ax2.plot(t2, VG2_new, label="$V_A$, new mutations")
    ax2.set_xlabel("Time since admixture")
    ax2.legend()
    ax2.set_title("Admixture into small deme (ky)")

    plt.show()
