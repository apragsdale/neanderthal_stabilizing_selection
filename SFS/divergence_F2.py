import moments
import numpy as np
import matplotlib, matplotlib.pylab as plt

# set font sizes
plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=7)
matplotlib.rc("ytick", labelsize=7)
matplotlib.rc("axes", labelsize=8)
matplotlib.rc("axes", titlesize=8)
matplotlib.rc("legend", fontsize=8)


def F2_prob(i, j, n, m):
    # F2 = (p - q) ** 2 = p ** 2 + q ** 2 - 2 * p * q
    # sampling probs by term:
    #  i * (i - 1) / n / (n - 1)
    #  j * (j - 1) / m / (m - 1)
    #  - 2 * i * j / n / m
    return i * (i - 1) / n / (n - 1) + j * (j - 1) / m / (m - 1) - 2 * i * j / n / m


def F2(fs):
    if fs.ndim != 2:
        raise ValueError("fs must have just two populations")

    val = 0
    n, m = fs.sample_sizes
    for i in range(n + 1):
        for j in range(m + 1):
            val += fs.data[i, j] * F2_prob(i, j, n, m)
    return val


def get_F2s(ns, s, theta, steps, T, nus):
    f2s = []

    fs = moments.Spectrum(
        moments.LinearSystem_1D.steady_state_1D(sum(ns), overdominance=s, theta=theta)
    )
    fs = fs.split(0, ns[0], ns[1])
    f2s.append(F2(fs))
    for _ in range(steps):
        fs.integrate(nus, T / steps, overdominance=s, theta=theta)
        f2s.append(F2(fs))
    return f2s


if __name__ == "__main__":
    ss = [-0.1, -1, -10]
    ns = [50, 50]
    Ne = 1e4
    u = 1e-8
    theta = 4 * Ne * u
    nus = [1, 1]

    T = 2
    steps = 100
    ts = np.linspace(0, T, steps + 1)

    fig = plt.figure(1234, figsize=(5, 4))
    fig.clf()
    ax = plt.subplot(1, 1, 1)

    neu = get_F2s(ns, 0, theta, steps, T, nus)
    ax.plot(ts, neu, label="Neutral")
    print("done with neutral sim")

    for s in ss:
        sel = get_F2s(ns, s, theta, steps, T, nus)
        ax.plot(ts, sel, label=f"Underdominance ($2N_es={s}$)")
        print("done with sel sim", s)

    ax.legend()
    ax.set_xlabel("Time ($2N_e$ generations)")
    ax.set_ylabel("$F_2$")
    fig.tight_layout()
    fig.savefig("../figures/underdominance_F2.pdf")
    # plt.show()
