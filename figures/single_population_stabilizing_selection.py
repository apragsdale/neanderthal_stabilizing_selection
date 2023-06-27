import numpy as np
import matplotlib.pylab as plt, matplotlib
import moments
import demes, demesdraw

# set font sizes
plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=6)
matplotlib.rc("ytick", labelsize=6)
matplotlib.rc("axes", labelsize=7)
matplotlib.rc("axes", titlesize=7)
matplotlib.rc("legend", fontsize=6)


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

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


def get_VG(fs, a):
    return fs.pi() * a ** 2

def get_gamma(a, VG, Ne, VS=1):
    s = a ** 2 / 2 / (VS + VG)
    gamma = -2 * Ne * s
    return gamma

def init_fs(n, gamma, theta):
    return moments.Spectrum(
        moments.LinearSystem_1D.steady_state_1D(
            n, overdominance=gamma, theta=theta
        )
    )

def get_time_series_VA(a, VS=1, mu=0.025, n=200, update_sel=False):
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

if __name__ == "__main__":
    grid = (2, 2)
    fig = plt.figure(1, figsize=(6.5, 3.5))

    ax1 = plt.subplot2grid(grid, (0, 0), rowspan=2)
    # stabilizing selection model

    VS = 1
    opt = 0
    VG = 0.1

    z = np.linspace(-2, 2, 201)
    w = np.exp(-(z ** 2) / 2 / VS)
    f = np.exp(-(z ** 2) / 2 / VG)

    ax1.plot(z, w, color=colors[0], label="$w(z)$")
    ax1.plot(z, f, color=colors[1], label="$f(z)$")
    ax1.vlines(0, 0, 1, colors="k", linestyles="--", lw=1)

    ax1.annotate(
        "",
        xy=(0.43, 0.45),
        xytext=(0, 0.45),
        arrowprops=dict(arrowstyle="->", lw=1),
    )
    ax1.annotate(
        "",
        xy=(1.21, 0.5),
        xytext=(0, 0.5),
        arrowprops=dict(arrowstyle="->", lw=1),
    )

    ax1.text(0.215, 0.41, "$V_P$", va="center", ha="center", color="k", fontsize=8)
    ax1.text(0.605, 0.53, "$V_S$", va="center", ha="center", color="k", fontsize=8)

    ax1.set_ylim(bottom=0)
    ax1.set_xlim([-2, 2])

    ax1.set_xlabel("Phenotype ($z$)")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(-1, 0.77, "$w(z)$", va="center", ha="center", color=colors[0], fontsize=8)
    ax1.text(-0.5, 0.65, "$f(z)$", va="center", ha="center", color=colors[1], fontsize=8)

    ax1.annotate(
        "",
        xy=(-0.6, 0.25),
        xytext=(-1.5, 0.25),
        arrowprops=dict(arrowstyle="->", lw=1),
    )
    ax1.annotate(
        "",
        xy=(0.6, 0.25),
        xytext=(1.5, 0.25),
        arrowprops=dict(arrowstyle="->", lw=1),
    )

    ax1.text(-1.05, 0.27, "Stabilizing\nselection", ha="center", color="k", fontsize=6)
    ax1.text(1.05, 0.27, "Stabilizing\nselection", ha="center", color="k", fontsize=6)
    # ax1.legend(frameon=False)

    ax2 = plt.subplot2grid(grid, (0, 1), rowspan=1)

    demesdraw.size_history(g, ax=ax2, invert_x=True)
    ax2.set_xticks([0, T2, T1+T2])
    ax2.set_ylabel(None)
    ax2.set_title("Size history")
    ax2.set_xlabel(None)

    ax3 = plt.subplot2grid(grid, (1, 1), rowspan=1)

    a_vals = [0.1, 0.04, 0.01]
    mu = 0.025

    for a in a_vals:
        t, VGs = get_time_series_VA(a, mu=mu, update_sel=True)
        VGs = [_ / (4 * mu * VS) for _ in VGs]
        ax3.plot(t, VGs, label=f"$a={a}$")

    plt.gca().set_prop_cycle(None)
    for a in a_vals:
        t, VGs = get_time_series_VA(a, mu=mu, update_sel=False)
        VGs = [_ / (4 * mu * VS) for _ in VGs]
        ax3.plot(t, VGs, "--", lw=1, label=None)

    #ax3.legend(frameon=False, loc=8)
    ax3.set_title("Additive genetic variance")
    ax3.set_xlim(ax2.get_xlim())
    ax3.set_xlabel("Time ago (generations)")
    ax3.set_xticks(ax2.get_xticks())
    ax3.set_ylim(bottom=0, top=1.5)
    ax3.set_ylabel("$V_{A}$ (in units of $4\mu V_S$)")

    ax3.text(30000, 1.2, f"$a={a_vals[0]}$", ha="center", color=colors[0], fontsize=6)
    ax3.text(30000, 0.8, f"$a={a_vals[1]}$", ha="center", color=colors[1], fontsize=6)
    ax3.text(30000, 0.2, f"$a={a_vals[2]}$", ha="center", color=colors[2], fontsize=6)

    plt.tight_layout()

    plt.savefig("one_pop.pdf")
    # plt.show()
