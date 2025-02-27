import moments
import numpy as np
import scipy.stats

SD = 0.05

a_list = np.logspace(-3, np.log10(2 * SD))
optimum = 0
VS = 1
mu = 0.025
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


def gaussian_des_bin(a_list, SFS_list, SD, j=1):
    """
    a_list: vector of effect size points
    VG_list: computed VG for given theta from SFS
    SD: standard deviation of the distribution of effect sizes of new mutations
    """
    assert len(a_list) == len(SFS_list)
    S = 0
    aa = np.concatenate(([0], a_list))
    dxs = (aa - np.concatenate(([aa[0]], aa))[:-1]) / 2 + (
        np.concatenate((aa, [aa[-1]]))[1:] - aa
    ) / 2
    weights = scipy.stats.norm.pdf(a_list, scale=SD)
    n = SFS_list[0].sample_sizes[0]
    for a, fs, dx, w in zip(a_list, SFS_list, dxs[1:], weights):
        assert fs.sample_sizes[0] == n
        v = 2 * j * (n - j) / n / (n - 1) * a ** 2 * fs[j]
        S += 2 * v * dx * w
    # add strong effects (which we assume contrubute as 4*mu*(VS+VG)
    c = 2 * scipy.stats.norm.cdf(-a_list[-1], scale=SD)
    v = 2 * j * (n - j) / n / (n - 1) * a ** 2 * SFS_list[-1][j]
    S += c * v
    return S


def gaussian_des_M_bin(a_list, SFS_list, SD, j=1):
    assert len(a_list) == len(SFS_list)
    S = 0
    aa = np.concatenate(([0], a_list))
    dxs = (aa - np.concatenate(([aa[0]], aa))[:-1]) / 2 + (
        np.concatenate((aa, [aa[-1]]))[1:] - aa
    ) / 2
    weights = scipy.stats.norm.pdf(a_list, scale=SD)
    n = SFS_list[0].sample_sizes[0]
    for a, fs, dx, w in zip(a_list, SFS_list, dxs[1:], weights):
        assert fs.sample_sizes[0] == n
        S += 2 * fs[j] * dx * w
    # add strong effects (which we assume contrubute as 4*mu*(VS+VG)
    c = 2 * scipy.stats.norm.cdf(-a_list[-1], scale=SD)
    S += c * SFS_list[-1][j]
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


def get_VGs_two_pop(fss, a_list):
    VGs1 = [get_VG(fs.marginalize([1]), a) for fs, a in zip(fss, a_list)]
    VGs2 = [get_VG(fs.marginalize([0]), a) for fs, a in zip(fss, a_list)]
    return VGs1, VGs2


EVG = 4 * mu * VS
gammas = [get_gamma(a, EVG, Ne, VS=VS) for a in a_list]

n = 300
fss = [init_fs(2 * n, gamma, theta) for gamma in gammas]

for i, fs in enumerate(fss):
    fss[i] = fs.split(0, n, n)


nus = [1, 0.1]
T = (600000 - 250000) / 25 / 2 / Ne

for a, fs in zip(a_list, fss):
    gamma = get_gamma(a, EVG, Ne, VS=VS)
    fs.integrate(nus, T, theta=theta, overdominance=gamma)

fss_seg = []
fss_intro = []
for i in range(len(fss)):
    fac_seg = np.zeros(fss[i].shape)
    fac_seg[:, 1:-1] = 1
    fac_intro = np.ones(fss[i].shape)
    fac_intro[:, 1:-1] = 0
    fs_seg = fss[i] * fac_seg
    fs_intro = fss[i] * fac_intro
    fss_seg.append(fs_seg.admix(0, 1, n, 0.05))
    fss_intro.append(fs_intro.admix(0, 1, n, 0.05))
    print("admixed", i)

fss_seg = copy.deepcopy(fss_seg_copy)
fss_intro = copy.deepcopy(fss_intro_copy)

VG_list = [get_VG(fs, a) for fs, a in zip(fss_seg, a_list)]
VG_seg = gaussian_des(a_list, VG_list, SD)

VG_list = [get_VG(fs, a) for fs, a in zip(fss_intro, a_list)]
VG_intro = gaussian_des(a_list, VG_list, SD)

VG_list = [get_VG(fs.marginalize([0]), a) for fs, a in zip(fss, a_list)]
VG_tot = gaussian_des(a_list, VG_list, SD)


def get_h2s(fss_intro, fss_seg, SD):
    n = fss_intro[0].sample_sizes[0]
    for fs in fss_intro:
        assert fs.Npop == 1 and fs.sample_sizes[0] == n
    for fs in fss_seg:
        assert fs.Npop == 1 and fs.sample_sizes[0] == n
    # by frequencies
    h2_intro_bin = [gaussian_des_bin(a_list, fss_intro, SD, j=j) for j in range(1, n)]
    h2_intro = sum(h2_intro_bin)
    M_intro_bin = [gaussian_des_M_bin(a_list, fss_intro, SD, j=j) for j in range(1, n)]
    M_intro = sum(M_intro_bin)

    h2_seg_bin = [gaussian_des_bin(a_list, fss_seg, SD, j=j) for j in range(1, n)]
    h2_seg = sum(h2_seg_bin)
    M_seg_bin = [gaussian_des_M_bin(a_list, fss_seg, SD, j=j) for j in range(1, n)]
    M_seg = sum(M_seg_bin)

    h2_intro_bin_snp = [h / m for h, m in zip(h2_intro_bin, M_intro_bin)]
    h2_seg_bin_snp = [h / m for h, m in zip(h2_seg_bin, M_seg_bin)]

    h2_intro_snp = h2_intro / M_intro
    h2_seg_snp = h2_seg / M_seg
    h2_seg_weighted_snp = (
        sum([h * m for h, m in zip(h2_seg_bin_snp, M_intro_bin)]) / M_intro
    )
    return h2_intro, h2_seg, h2_intro_snp, h2_seg_snp, h2_seg_weighted_snp


all_data = {}
for SD in [0.01, 0.02, 0.05]:
    T = 0.001

    fss_intro = copy.deepcopy(fss_intro_copy)
    fss_seg = copy.deepcopy(fss_seg_copy)

    VG_list = [get_VG(fs.marginalize([0]), a) for fs, a in zip(fss, a_list)]
    VG_tot = gaussian_des(a_list, VG_list, SD)
    print(VG_tot)

    data = {"VG_init": VG_tot}
    data["traj"] = {0: get_h2s(fss_intro, fss_seg, SD)}
    print(data["traj"][0])

    for i in range(100):
        for a, fs in zip(a_list, fss_seg):
            gamma = get_gamma(a, EVG, Ne, VS=VS)
            fs.integrate([0.1], T, theta=theta, overdominance=gamma)
        for a, fs in zip(a_list, fss_intro):
            gamma = get_gamma(a, EVG, Ne, VS=VS)
            fs.integrate([0.1], T, theta=0, overdominance=gamma)
        data["traj"][(i + 1) * T] = get_h2s(fss_intro, fss_seg, SD)
        print(data["traj"][(i + 1) * T])
    all_data[SD] = data
