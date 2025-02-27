import moments
import numpy as np
import matplotlib.pylab as plt
import scipy.stats


def get_VG(fs, a):
    return fs.pi() * a ** 2


def additive_variance_from_sfs(a=0, VS=1, Ne=1e4, theta=1, slices=50, n=50):
    s = a ** 2 / 2 / VS
    gamma = -2 * Ne * s

    fs = moments.Spectrum(
        moments.LinearSystem_1D.steady_state_1D(
            4 * n, overdominance=gamma, theta=theta
        ),
        pop_ids=["anc"],
    )

    t = [0]
    VG = [[get_VG(fs, a), get_VG(fs, a)]]
    fs = fs.split(0, 2 * n, 2 * n, new_ids=["H", "N"])
    T = 1
    Tslice = T / slices
    for i in range(slices):
        fs.integrate([1, 0.1], Tslice, overdominance=gamma, theta=theta)
        VG.append([get_VG(fs.marginalize([1]), a), get_VG(fs.marginalize([0]), a)])
        t.append(t[-1] + Tslice)

    fs = fs.admix(0, 1, n, 0.95, new_id="H")
    fs = fs.admix(0, 1, n, 0.05, new_id="N")
    t.append(t[-1])
    VG.append([get_VG(fs.marginalize([1]), a), get_VG(fs.marginalize([0]), a)])

    T = 0.2
    Tslice = T / slices
    for i in range(slices):
        fs.integrate([1, 0.1], Tslice, overdominance=gamma, theta=theta)
        VG.append([get_VG(fs.marginalize([1]), a), get_VG(fs.marginalize([0]), a)])
        t.append(t[-1] + Tslice)

    return t, VG


def additive_variance_one_way(
    a=0, VS=1, Ne=1e4, focal_pop="human", theta=1, n=100, update_sel=True
):
    def get_gamma(a, VS, v, Ne):
        s = a ** 2 / 2 / (VS + v)
        gamma = -2 * Ne * s
        return gamma

    gamma = get_gamma(a, VS, 0, Ne)
    if update_sel:
        for _ in range(8):
            fs = moments.Spectrum(
                moments.LinearSystem_1D.steady_state_1D(
                    2 * n, overdominance=gamma, theta=theta
                ),
            )
            v = get_VG(fs, a)
            gamma = get_gamma(a, VS, v, Ne)
    else:
        fs = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(
                2 * n, overdominance=gamma, theta=theta
            ),
        )

    t = [0]
    v = get_VG(fs, a)
    if update_sel:
        gamma = get_gamma(a, VS, v, Ne)
    VG = [v]
    fs = fs.split(0, n, n)
    T = 1
    num_slices = 100
    if focal_pop == "human":
        nu = [1, 0.1]
    else:
        nu = [0.1, 1]
    for i in range(num_slices):
        fs.integrate(nu, T / num_slices, overdominance=gamma, theta=theta)
        v0 = get_VG(fs.marginalize([1]), a)
        v1 = get_VG(fs.marginalize([0]), a)
        VG.append(v0)
        t.append(t[-1] + T / num_slices)
        if update_sel:
            gamma = [get_gamma(a, VS, v0, Ne), get_gamma(a, VS, v0, Ne)]

    fs = fs.admix(0, 1, n, 0.95)
    t.append(t[-1])
    v = get_VG(fs, a)
    if update_sel:
        gamma = get_gamma(a, VS, v, Ne)
    VG.append(v)
    T = 0.2
    num_slices = 50
    if focal_pop == "human":
        nu = [1]
    else:
        nu = [0.1]
    for i in range(num_slices):
        fs.integrate(nu, T / num_slices, overdominance=gamma, theta=theta)
        v = get_VG(fs, a)
        if update_sel:
            gamma = get_gamma(a, VS, v, Ne)
        VG.append(v)
        t.append(t[-1] + T / num_slices)

    return np.asarray(t), np.asarray(VG)


def gaussian_des(a_list, VG_list, SD):
    S = 0
    aa = np.concatenate(([0], a_list))
    dxs = (aa - np.concatenate(([aa[0]], aa))[:-1]) / 2 + (
        np.concatenate((aa, [aa[-1]]))[1:] - aa
    ) / 2
    weights = scipy.stats.norm.pdf(a_list, scale=SD)
    for v, dx, w in zip(VG_list, dxs[1:], weights):
        S += v * dx * w
    return 2 * S


if __name__ == "__main__":
    Ne = 1e4
    VS = 1
    theta = 1000

    a = 0.01

    t, VG = additive_variance_from_sfs(a=a, VS=VS, Ne=Ne, theta=theta)

    VG0 = np.array([v[0] for v in VG])
    VG1 = np.array([v[1] for v in VG])
    T = np.array(t) - t[-1]
    T *= 2 * Ne

    plt.plot(T, VG0, label="Human")
    plt.plot(T, VG1, label="Neand")
    plt.legend()
