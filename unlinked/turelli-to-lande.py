import numpy as np
import pickle
import sys


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


def report(freqs, es, ss, VGs=None, report=True):
    EP = 2 * np.sum(freqs * es)
    VP = 2 * np.sum(freqs * (1 - freqs) * es * es)
    if report:
        print(f" num sites: {ss.sum()}; VG={VP:0.6f}; mean phenotype={EP:0.5f}")
    if VGs is not None:
        VGs.append(VP)


if __name__ == "__main__":
    # draw mutations from a normal
    optimum = 0
    VS = 1
    mu = 0.01
    Ne = 10000

    expectedVG = 4 * mu * VS
    print("Expected VG (HoC):", expectedVG)

    #SDs = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    SDs = np.logspace(-4, -1, 37)
    mean_VGs = []
    for SD in SDs:
        VGs = []
        #print("Expected VG  (mu):", 4 * Ne * mu * SD ** 2)
        freqs, es, ss = np.array([]), np.array([]), np.array([])
        for i in range(40 * Ne):
            # burn in from initial state
            freqs, es, ss = evolve(freqs, es, ss, SD, optimum, VS, mu, Ne)
            report(freqs, es, ss, VGs=VGs, report=False)
        mean_VGs.append(np.mean(VGs[20 * Ne :]))
        print(SD, mean_VGs[-1])
