import numpy as np
import numpy.random as npr
import moments
import matplotlib.pylab as plt
import pickle

rng = npr.default_rng()


def mutation(X, Ne, u, L=1):
    num_mut = rng.poisson(2 * Ne * u * L)
    X = np.concatenate((X, np.ones(num_mut) / 2 / Ne))
    return X


def selection(X, s):
    X += s * X * (1 - X) * (1 - 2 * X)
    return X


def drift(X, Ne):
    X = rng.binomial(2 * Ne, X) / 2 / Ne
    return X


def cleanup(X):
    fixed = np.logical_or(X == 0, X == 1)
    X = X.compress(1 - fixed)
    return X


def evolve(X, Ne, u, s, L=1):
    X = selection(X, s)
    X = drift(X, Ne)
    X = mutation(X, Ne, u, L=L)
    X = cleanup(X)
    return X


def sample(X, n):
    F_sample = np.zeros(n + 1)
    idx, counts = np.unique(rng.binomial(n, X), return_counts=True)
    for i, c in zip(idx, counts):
        if i == 0 or i == n:
            continue
        F_sample[i] = c
    return F_sample


# params
Ne = 5000
u = 1e-8
s = 0.0005

print("gamma =", 2 * Ne * s)

X = np.empty(0)

L = 10000
n = 100
F = np.zeros(n + 1)

burnin = 20
for gen in range(burnin * 2 * Ne):
    X = evolve(X, Ne, u, s, L=L)
    if (gen + 1) % (2 * Ne) == 0:
        print((gen + 1) // 2 // Ne, "times 2Ne, of", burnin)


sample_units = 1000
for gen in range(sample_units * 2 * Ne):
    X = evolve(X, Ne, u, s, L=L)
    F += sample(X, n)
    if (gen + 1) % (2 * Ne) == 0:
        print((gen + 1) // 2 // Ne, "times 2Ne, of", sample_units)


fname = f"sim_data/Ne_{Ne}.n_{n}.L_{L}.u_{u}.s_{s}.sample_units_{sample_units}.pkl"
with open(fname, "wb+") as fout:
    pickle.dump(
        {
            "Ne": Ne,
            "n": n,
            "L": L,
            "u": u,
            "s": s,
            "sample_units": sample_units,
            "F": F,
        },
        fout,
    )

data = moments.Spectrum(F)
gamma = 2 * Ne * s
scaling = 4 * Ne * u * sample_units * 2 * Ne * L
fs = scaling * moments.Spectrum(
    moments.LinearSystem_1D.steady_state_1D(n, overdominance=gamma)
)

fig = plt.figure(1, figsize=(6.5, 5))
fig.clf()
moments.Plotting.plot_1d_comp_Poisson(
    fs, data, fig_num=1, show=False, labels=["moments", "Simulation"]
)
if s == -0.00005:
    fig.suptitle(f"$N_e={Ne}$, $s={s:.5f}$")
elif s == -0.001:
    fig.suptitle(f"$N_e={Ne}$, $s={s:.3f}$")
else:
    fig.suptitle(f"$N_e={Ne}$, $s={s:f}$")
fig.tight_layout()
fig.savefig(f"underdominance.Ne_{Ne}.s_{s}.pdf")
plt.show()
