import numpy as np
import sys

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

SD = float(sys.argv[1])
out_idx = sys.argv[2]

print("running", out_idx)

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


def initialize_population(SD, VS, Ne, expectedVG):
    VG = 0
    EG = 0
    freqs = []
    es = []
    fs_neu = eq_sfs(Ne, 0)
    while VG < expectedVG:
        # draw from normal with given SD
        a = np.random.randn() * SD
        # determine allele frequency, if segregating (stronger effects, less chance of seg)
        # a + EG because there is stronger selection against effect sizes that point
        # in the same direction deviation from the optimum
        s = (a + EG) ** 2 / 2 / VS
        # assumes directional selection, despite this being the s w/underdominance
        fs = eq_sfs(Ne, gamma=-2 * Ne * s)
        if np.random.rand() < fs.sum() / fs_neu.sum():
            i = np.random.choice(range(2 * Ne + 1), p=fs / fs.sum())
            p = i / 2 / Ne
            freqs.append(p)
            es.append(a)
            VG += 2 * p * (1 - p) * a ** 2
            EG += 2 * p * a
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


def new_mutations(freqs, es, ss, mu, Ne):
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
        es[site_idx] = np.random.randn() * SD
    return freqs, es, ss


def evolve(freqs, es, ss, SD, optimum, VS, mu, Ne):
    # get marginal fitnesses of alleles at all segregating sites
    fprime = get_fprime(freqs, es, optimum, VS)

    # binomial sampling of alleles to create offspring
    freqs = sample_generation(Ne, fprime)

    # update to remove lost mutations
    ss[freqs == 0] = 0

    # introduce new mutations
    freqs, es, ss = new_mutations(freqs, es, ss, mu, Ne)

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

def split_1_to_2(freqs):
    assert len(freqs.shape) == 1
    return np.array([freqs, freqs])


def new_mutations_2(freqs2, es, ss, mu, Nes):
    num_muts1 = np.random.poisson(2 * Nes[0] * mu)
    num_muts2 = np.random.poisson(2 * Nes[1] * mu)
    num_muts = num_muts1 + num_muts2
    extension = num_muts - (ss == 0).sum()
    if extension > 0:
        freqs2 = np.concatenate((freqs2, np.zeros((2, extension))), axis=1)
        es = np.concatenate((es, np.zeros(extension)))
        ss = np.concatenate((ss, np.zeros(extension, dtype=int)))
    free_sites = np.where(ss == 0)[0]
    for i in range(num_muts1):
        site_idx = free_sites[i]
        # add to G
        freqs2[0, site_idx] = 1 / 2 / Nes[0]
        # add to seg sites
        ss[site_idx] = 1
        # add effect size
        es[site_idx] = np.random.randn() * SD
    free_sites = np.where(ss == 0)[0]
    for i in range(num_muts2):
        site_idx = free_sites[i]
        # add to G
        freqs2[1, site_idx] = 1 / 2 / Nes[1]
        # add to seg sites
        ss[site_idx] = 1
        # add effect size
        es[site_idx] = np.random.randn() * SD
    return freqs2, es, ss


def evolve_2(freqs2, es, ss, SD, optima, VSs, mu, Nes, allele_labels=None):
    assert len(Nes) == 2
    assert len(optima) == 2
    assert len(VSs) == 2
    assert len(freqs2.shape) == 2

    # get marginal fintesses of alleles
    fprime1 = get_fprime(freqs2[0], es, optima[0], VSs[0])
    fprime2 = get_fprime(freqs2[1], es, optima[1], VSs[1])

    # binomial sampling in each population
    freqs2[0] = sample_generation(Nes[0], fprime1)
    freqs2[1] = sample_generation(Nes[1], fprime2)

    # update to remove lost mutations
    if allele_labels is not None:
        if len(allele_labels) != len(ss):
            allele_labels = np.concatenate(
                (allele_labels, np.array([-1] * (len(ss) - len(allele_labels))))
            )
        allele_labels[np.sum(freqs2, axis=0) == 0] = -1
    ss[np.sum(freqs2, axis=0) == 0] = 0

    # introduce new mutations
    freqs2, es, ss = new_mutations_2(freqs2, es, ss, mu, Nes)

    # clean up if many nonseg sites
    if np.sum(ss) < 0.9 * len(ss):
        freqs2, es, ss = cleanup(freqs, es, ss)

    return freqs2, es, ss


def admix(freqs2, proportions):
    # proportions are [2 to 1, 1 to 2]
    freqs_new = np.zeros(freqs2.shape)
    freqs_new[0] = (1 - proportions[0]) * freqs2[0] + proportions[0] * freqs2[1]
    freqs_new[1] = (1 - proportions[1]) * freqs2[1] + proportions[1] * freqs2[0]
    return freqs_new


def report(freqs, es, ss, VGs=None, report=True):
    if len(freqs.shape) == 1:
        EP = 2 * np.sum(freqs * es)
        VP = 2 * np.sum(freqs * (1 - freqs) * es * es)
        if report:
            print(f" num sites: {ss.sum()}; VG={VP:0.3f}; mean phenotype={EP:0.5f}")
        VPs = [VP, VP]
    if len(freqs.shape) == 2:
        EPs = 2 * np.sum(freqs * es, axis=1)
        VPs = 2 * np.sum(freqs * (1 - freqs) * es * es, axis=1)
        if report:
            print(
                f" num_sites: {ss.sum()}; VGs={VPs[0]:0.3f}, {VPs[1]:0.3f}; mean phenos={EPs[0]:0.5f}, {EPs[1]:0.5f}"
            )
    if VGs is not None:
        VGs.append([V for V in VPs])


def F2(freqs2):
    return np.sum((freqs2[0] - freqs2[1]) ** 2)


def initialize_tracked_frequencies(freqs2, ss, es):
    tracked_frequencies = {}
    for i, (seg, e) in enumerate(zip(ss, es)):
        if seg == -1:
            continue
        else:
            k = (i, e)
        tracked_frequencies[(i, e)] = [list(freqs2[:, i])]
    return tracked_frequencies


def update_tracked_frequencies(tracked_frequencies, freqs2, es):
    for i, e in enumerate(es):
        k = (i, e)
        if k in tracked_frequencies:
            tracked_frequencies[k].append(list(freqs2[:, i]))


def sum_F2_a2(freqs2, es):
    return np.sum((freqs2[0] - freqs2[1]) ** 2 * es ** 2)


if __name__ == "__main__":
    # draw mutations from a normal
    optimum = 0
    VS = 1

    # per-generation per-individual mutation rate
    mu = 0.025
    Ne = 10000

    expectedVG = 4 * mu * VS

    VGs = []

    # initialize population
    freqs, es, ss = initialize_population(SD, VS, Ne, expectedVG)
    #report(freqs, es, ss)
    for i in range(10 * Ne):
        # burn in from initial state
        freqs, es, ss = evolve(freqs, es, ss, SD, optimum, VS, mu, Ne)
        report(freqs, es, ss, VGs=VGs, report=False)
        #if i % 1000 == 0:
        #    report(freqs, es, ss)

    #report(freqs, es, ss)
    # split population
    freqs2 = split_1_to_2(freqs)
    # evolve 500,000 years (20,000 generations)
    for i in range(20000):
        freqs2, es, ss = evolve_2(
            freqs2, es, ss, SD, [optimum, optimum], [VS, VS], mu, [Ne, Ne // 10]
        )
        report(freqs2, es, ss, VGs=VGs, report=False)
        #if i % 100 == 0:
        #    report(freqs2, es, ss)

    #report(freqs2, es, ss)
    # start tracking frequencies
    # only track frequencies of mutations segregating at this time, instead of
    # any new ones
    # tracked_frequencies = initialize_tracked_frequencies(freqs2, ss, es)

    # admixture
    #print(sum_F2_a2(freqs2, es))

    freqs2 = admix(freqs2, [0.05, 0.05])
    # continue 100,000 years (4,000 generations)
    for i in range(4000):
        freqs2, es, ss = evolve_2(
            freqs2, es, ss, SD, [optimum, optimum], [VS, VS], mu, [Ne, Ne // 10]
        )
        # update_tracked_frequencies(tracked_frequencies, freqs2, es)
        report(freqs2, es, ss, VGs=VGs, report=False)
        #if i % 100 == 0:
            #report(freqs2, es, ss)
    #report(freqs2, es, ss)

    VGs = np.array(VGs).T
    np.save(f"simulated_VGs/VGs.SD_{SD}.{out_idx}", VGs)

    """
    VG0_from0 = []
    VG1_from0 = []
    VG0_from1 = []
    VG1_from1 = []
    for i in range(1, 4001):
        x, y, z, w = 0, 0, 0, 0
        for k, v in tracked_frequencies.items():
            f00, f10 = v[0]
            if i >= len(v):
                continue
            f0, f1 = v[i]
            if f00 == 0 or f00 == 1:
                x += 2 * f0 * (1 - f0) * k[1] ** 2
            else:
                y += 2 * f0 * (1 - f0) * k[1] ** 2
            if f10 == 0 or f10 == 1:
                z += 2 * f1 * (1 - f1) * k[1] ** 2
            else:
                w += 2 * f1 * (1 - f1) * k[1] ** 2
        VG0_from1.append(x)
        VG0_from0.append(y)
        VG1_from0.append(z)
        VG1_from1.append(w)

    import matplotlib.pylab as plt
    plt.plot([_[0] for _ in VGs], label="VG0")
    plt.plot([_[1] for _ in VGs], label="VG1")
    plt.plot(VG0_from0, "--", label="VG0 (seg)")
    plt.plot(VG1_from1, "--", label="VG1 (seg)")
    plt.plot(VG0_from1, ":", label="VG0 (int)")
    plt.plot(VG1_from0, ":", label="VG1 (int)")
    plt.legend()
    plt.show()
    """
