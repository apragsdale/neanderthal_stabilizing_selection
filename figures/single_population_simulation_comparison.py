import moments
import copy
import sys
import numpy as np

from single_population_stabilizing_selection import *


def eq_sfs(Ne, gamma):
    # See Ewens, or Evans et al (2007), equation 31 for example
    x = np.arange(2 * Ne + 1) / 2 / Ne
    y = np.zeros(len(x))
    if gamma == 0:
        y[1:-1] = 1 / x[1:-1] / 2 / Ne
    else:
        y[1:-1] = (
            (
                np.exp(2 * gamma)
                * (1 - np.exp(-2 * gamma * (1 - x[1:-1])))
                / ((np.exp(2 * gamma) - 1) * x[1:-1] * (1 - x[1:-1]))
            )
            / 2
            / Ne
        )
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        y = np.zeros(len(x))
    return y


def initialize_population(a, VS, Ne, expectedVG):
    VG = 0
    EG = 0
    freqs = []
    es = []
    fs_neu = eq_sfs(Ne, 0)
    while VG < expectedVG:
        sign = (-1) ** np.random.randint(2)
        s = (sign * a + EG) ** 2 / 2 / VS
        gamma = -2 * Ne * s
        fs = eq_sfs(Ne, gamma=gamma)
        if np.random.rand() < fs.sum() / fs_neu.sum():
            i = np.random.choice(range(2 * Ne + 1), p=fs / fs.sum())
            p = i / 2 / Ne
            freqs.append(p)
            es.append(sign * a)
            VG += 2 * p * (1 - p) * a ** 2
            EG += 2 * p * sign * a
    # track segregating sites
    freqs = np.array(freqs)
    es = np.array(es)
    ss = np.ones(len(es), dtype=int)
    return freqs, es, ss


def get_fprime(freqs, es, optimum, VS):
    two_f_a = 2 * freqs * es
    VG_contributions = 2 * freqs * (1 - freqs) * es * es
    EG = two_f_a.sum()
    VG = VG_contributions.sum()
    VGprime = VG - VG_contributions
    EGprime = EG - two_f_a

    delta_p = (
        freqs
        * (1 - freqs)
        * np.sqrt((VG + VS) / (VGprime + VS))
        * np.exp(((EG - optimum) ** 2) / (2 * (VG + VS)))  # nearly 1
        * (  # also nearly 1
            freqs * np.exp(-((EGprime + 2 * es - optimum) ** 2) / (2 * (VGprime + VS)))
            - (1 - freqs) * np.exp(-((EGprime - optimum) ** 2) / (2 * (VGprime + VS)))
            + (1 - 2 * freqs)
            * np.exp(-((EGprime + es - optimum) ** 2) / (2 * (VGprime + VS)))
        )
    )
    return freqs + delta_p


def sample_generation(Ne, fprime):
    return np.random.binomial(2 * Ne, p=fprime) / 2 / Ne


def new_mutations(freqs, es, ss, mu, Ne, a):
    num_muts = np.random.poisson(2 * Ne * mu)
    extension = num_muts - (ss == 0).sum()
    if extension > 0:
        freqs = np.concatenate((freqs, np.zeros(extension)))
        es = np.concatenate((es, np.zeros(extension)))
        ss = np.concatenate((ss, np.zeros(extension, dtype=int)))
    free_sites = np.where(ss == 0)[0]
    for i in range(num_muts):
        site_idx = free_sites[i]
        # add to G
        freqs[site_idx] = 1 / 2 / Ne
        # add to seg sites
        ss[site_idx] = 1
        # add effect size
        es[site_idx] = (-1) ** np.random.randint(2) * a
    return freqs, es, ss


def evolve(freqs, es, ss, a, optimum, VS, mu, Ne):
    # get marginal fitnesses of alleles at all segregating sites
    fprime = get_fprime(freqs, es, optimum, VS)

    # binomial sampling of alleles to create offspring
    freqs = sample_generation(Ne, fprime)

    # update to remove lost mutations
    ss[freqs == 0] = 0

    # introduce new mutations
    freqs, es, ss = new_mutations(freqs, es, ss, mu, Ne, a)

    # clean up if many nonseg sites
    if np.sum(ss) < 0.9 * len(ss):
        freqs, es, ss = cleanup(freqs, es, ss)

    return freqs, es, ss


def cleanup(freqs, es, ss):
    es = es.compress(ss == 1)
    if freqs.ndim == 1:
        freqs = freqs.compress(ss == 1)
    else:
        freqs = freqs.compress(ss == 1, axis=1)
    ss = ss.compress(ss == 1)
    return freqs, es, ss


def report(freqs, es, ss, VGs=None, report=True):
    EP = 2 * np.sum(freqs * es)
    VP = 2 * np.sum(freqs * (1 - freqs) * es * es)
    if report:
        print(f" num sites: {ss.sum()}; VG={VP:0.3f}; mean phenotype={EP:0.5f}")
    if VGs is not None:
        VGs.append(VP)


if __name__ == "__main__":
    a = float(sys.argv[1])

    optimum = 0
    VS = 1
    mu = 0.025

    expectedVG = 4 * mu * VS

    VG_replicates = []

    # initialize population
    # freqs, es, ss = initialize_population(a, VS, Ne, expectedVG)
    freqs = np.array([])
    es = np.array([])
    ss = np.array([])

    report(freqs, es, ss)
    # burn in
    for i in range(int(20 * Ne)):
        # burn in from initial state
        freqs, es, ss = evolve(freqs, es, ss, a, optimum, VS, mu, Ne)
        # report(freqs, es, ss, VGs=VGs, report=False)
        # if i % 100 == 0:
        # report(freqs, es, ss)
    # burn in complete

    n_reps = 100

    for i in range(n_reps):
        print("Running replicate", i + 1)
        VGs = []
        report(freqs, es, ss)
        # advance Ne generations, pre bottleneck
        for i in range(int(Ne)):
            freqs, es, ss = evolve(freqs, es, ss, a, optimum, VS, mu, Ne)
            report(freqs, es, ss, VGs=VGs, report=False)
            #if i % 1000 == 0:
            #    report(freqs, es, ss)

        freqs_sim = copy.copy(freqs)
        es_sim = copy.copy(es)
        ss_sim = copy.copy(ss)
        # bottleneck
        for i in range(int(T1)):
            freqs_sim, es_sim, ss_sim = evolve(
                freqs_sim, es_sim, ss_sim, a, optimum, VS, mu, NB
            )
            report(freqs_sim, es_sim, ss_sim, VGs=VGs, report=False)
            #if i % 100 == 0:
            #    report(freqs_sim, es_sim, ss_sim)

        # recovery
        for i in range(int(T2)):
            freqs_sim, es_sim, ss_sim = evolve(
                freqs_sim, es_sim, ss_sim, a, optimum, VS, mu, Ne
            )
            report(freqs_sim, es_sim, ss_sim, VGs=VGs, report=False)
            #if i % 100 == 0:
            #    report(freqs_sim, es_sim, ss_sim)

        VG_replicates.append(VGs)

    ave_VG = np.zeros(5 * int(Ne))
    for i in range(n_reps):
        ave_VG += VG_replicates[i]

    ave_VG /= n_reps

    bs_VGs = []
    lower = []
    upper = []
    for i in range(n_reps):
        y = np.zeros(5 * int(Ne))
        for i in np.random.choice(n_reps, size=n_reps):
            y += VG_replicates[i]
        y /= n_reps
        bs_VGs.append(y)
    
    for i in range(len(bs_VGs[0])):
        vals = [_[i] for _ in bs_VGs]
        vals = np.sort(vals)
        lower.append(vals[1])
        upper.append(vals[-2])

    # SFS trajectory
    t, VG_moments = get_time_series_VA(a, mu=mu, update_sel=False)
    t[0] = 50000

    fig = plt.figure(figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)
    ax.plot(t, VG_moments, lw=3, label="moments")
    ax.plot(
        len(ave_VG) - 1 - np.arange(len(ave_VG)), ave_VG, "--", lw=2, label="simulation ave."
    )
    ax.fill_between(len(ave_VG) - 1 - np.arange(len(ave_VG)), lower, upper, alpha=0.5, color=colors[1])
    ax.legend()
    ax.set_xlabel("Generations ago")
    ax.set_ylabel("$V_A$")
    ax.set_title(f"$a={a}$")
    ax.set_xlim(ax.get_xlim()[1], ax.get_xlim()[0])
    fig.tight_layout()
    plt.savefig(f"one_pop.no_adjust.a_{a}.pdf")
    #plt.show()
