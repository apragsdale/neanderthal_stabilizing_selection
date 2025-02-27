import numpy as np
import time


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
    counts = []
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
            counts.append(i)
            es.append(a)
            VG += 2 * p * (1 - p) * a ** 2
            EG += 2 * p * a
    haps = np.zeros((2 * Ne, len(counts)), dtype=int)
    for i, c in enumerate(counts):
        idx = np.random.choice(2 * Ne, size=c, replace=False)
        haps[idx, i] = 1
    G = haps[:Ne, :] + haps[Ne:, :]
    # track segregating sites
    ss = np.ones(len(es), dtype=int)
    es = np.array(es)
    return G, es, ss


def get_fitnesses(G, es, optimum, VS):
    # get phenotypes
    P = (G * es).sum(axis=1)
    w = np.exp(-((P - optimum) ** 2) / 2 / VS)
    return w


def sample_generation(G, es, ss, w):
    # parental probabilities based on their fitnesses
    parent_probs = w / w.sum()
    Goff = np.zeros(G.shape, dtype=int)
    # get a list of 2 * Ne parents (with replacement), based off of their probabilities
    parents = np.random.choice(Ne, size=2 * Ne, p=parent_probs)
    hets = {}
    for i, p in enumerate(parents):
        child = G[p] // 2
        if p not in hets:
            hets[p] = np.where(G[p] == 1)[0]
        het = hets[p]
        # homozygotes always pass 0 or 1, while het sites are 50/50
        # child[het] = np.random.binomial(1, 0.5, size=len(het))
        child[het] = np.random.randint(2, size=len(het))
        Goff[i % len(G)] += child
    return Goff


def new_mutations(G, es, ss, mu, Ne):
    num_muts = np.random.poisson(2 * Ne * mu)
    # extend as needed
    extension = num_muts - (ss == 0).sum()
    if extension > 0:
        print(" extending", extension)
        G = np.concatenate((G, np.zeros((Ne, extension), dtype=int)), axis=1)
        es = np.concatenate((es, np.zeros(extension)))
        ss = np.concatenate((ss, np.zeros(extension, dtype=int)))
    free_sites = np.where(ss == 0)[0]
    for i in range(num_muts):
        site_idx = free_sites[i]
        # add to G
        G[np.random.randint(Ne), site_idx] = 1
        # add to seg sites
        ss[site_idx] = 1
        # add effect size
        es[site_idx] = np.random.randn() * SD
    return G, es, ss


def evolve(G, es, ss, SD, optimum, VS, mu, Ne):
    # get marginal fitnesses of alleles at all segregating sites
    w = get_fitnesses(G, es, optimum, VS)

    # binomial sampling of alleles to create offspring
    G = sample_generation(G, es, ss, w)

    # update to remove lost mutations
    counts = G.sum(axis=0)
    ss[counts == 0] = 0

    # introduce new mutations
    G, es, ss = new_mutations(G, es, ss, mu, Ne)
    return G, es, ss


def report(G, es, ss):
    P = np.sum(G * es, axis=1)
    print(
        f" num sites: {ss.sum()}; VG={np.var(P):0.3f}; mean phenotype={np.mean(P):0.5f}"
    )


if __name__ == "__main__":
    # draw mutations from a normal
    SD = 0.05
    optimum = 0
    VS = 1

    # per-generation per-individual mutation rate
    mu = 0.025
    Ne = 10000

    expectedVG = 4 * mu * VS

    # initialize population
    G, es, ss = initialize_population(SD, VS, Ne, expectedVG)
    report(G, es, ss)
    for i in range(2000):
        time1 = time.time()
        G, es, ss = evolve(G, es, ss, SD, optimum, VS, mu, Ne)
        time2 = time.time()
        print(i, time2 - time1)
        report(G, es, ss)
