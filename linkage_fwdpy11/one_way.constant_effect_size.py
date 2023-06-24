"""
Run the fwdpy11 simulation and get ancestry proportions around introgressed
alleles that were fixed differences between populations.

import matplotlib.pylab as plt
max_dist = 1e6  # 1 Mb = 1 cM at r = 1e-8
dists = np.linspace(-max_dist, max_dist, 201)
r = 1e-8

plt.plot(dists * 100 * r, np.mean(all_props, axis=0))
plt.plot(dists * 100 * r, 0.05 * np.ones(len(dists)), ":")
plt.ylim(bottom=0)
plt.xlabel("Distance (cM)")
plt.ylabel("Introgressed ancestry prop.")
plt.legend(["Ave. at focal alleles", "Admixture fraction"])
plt.tight_layout()
plt.show()
"""


import fwdpy11
import tskit
import numpy as np
from dataclasses import dataclass
from typing import List
from collections import defaultdict
import pickle

import time
import demes

import sys


def get_samples_dict(ts):
    samples_dict = defaultdict(list)
    for s in ts.samples():
        samples_dict[(ts.node(s).time, ts.node(s).population)].append(s)
    return samples_dict


def get_ancestry_counts(ts):
    # preprocess and set up storage arrays
    samples_dict = get_samples_dict(ts)

    max_t = max([k[0] for k in samples_dict.keys()])
    ts2 = ts.decapitate(max_t)

    num_samples = len(samples_dict[min(samples_dict.keys())])

    sample_counts = np.zeros(ts2.num_nodes, dtype=int)
    parents = -np.ones(ts2.num_nodes, dtype=int)

    for s in samples_dict[min(samples_dict.keys())]:
        sample_counts[s] += 1

    num_seg = np.zeros(ts2.num_trees)
    num_intro = np.zeros(ts2.num_trees)

    for i, ed in enumerate(ts2.edge_diffs()):
        interval, edges_out, edges_in = ed
        # take care of edges out
        for e_out in edges_out:
            c = e_out.child
            p = e_out.parent
            assert parents[c] == p
            parents[c] = -1
            sample_counts[p] -= sample_counts[c]
            while parents[p] != -1:
                c = p
                p = parents[c]
                sample_counts[p] -= sample_counts[e_out.child]

        # update storage with edges in
        for e_in in edges_in:
            c = e_in.child
            p = e_in.parent
            assert parents[c] == -1
            parents[c] = p
            sample_counts[p] += sample_counts[c]
            while parents[p] != -1:
                c = p
                p = parents[c]
                sample_counts[p] += sample_counts[e_in.child]
        # count up ancestry proportions in this segment
        n0 = sum(sample_counts[samples_dict[(max_t, 0)]])
        n1 = sum(sample_counts[samples_dict[(max_t, 1)]])
        assert n0 + n1 == num_samples
        num_seg[i] = n0
        num_intro[i] = n1

    x = np.array(list(ts2.breakpoints()))
    return x, num_seg, num_intro


def set_up_demographic_model(Ne=10000):
    ## Set up demographic model
    gens1 = 2 * Ne
    gens2 = int(0.1 * 2 * Ne)
    b = demes.Builder(time_units="generations")
    b.add_deme("A", epochs=[dict(start_size=Ne, end_time=gens2)])
    b.add_deme(
        "B",
        ancestors=["A"],
        start_time=gens1 + gens2,
        epochs=[dict(start_size=0.1 * Ne, end_time=gens2)],
    )
    b.add_deme(
        "C",
        start_time=gens2,
        ancestors=["A", "B"],
        proportions=[0.95, 0.05],
        epochs=[dict(start_size=Ne)],
    )
    g = b.resolve()
    return g


## set up recorders
@dataclass
class SimData:
    generation: int
    demes_ids: List[int]
    mean_phenotype: List[float]
    mean_fitness: List[float]
    var_phenotype: List[float]


@dataclass
class Recorder:
    data: list
    sample_times: list

    def __call__(self, pop, sampler):
        md = np.array(pop.diploid_metadata)
        # store lists of mean phenotypes, fitnesses, etc
        deme_ids = sorted(list(set(md["deme"])))
        mean_pheno = [md[md["deme"] == i]["g"].mean() for i in deme_ids]
        mean_fitness = [md[md["deme"] == i]["w"].mean() for i in deme_ids]
        var_pheno = [md[md["deme"] == i]["g"].var() for i in deme_ids]
        self.data.append(
            SimData(pop.generation, deme_ids, mean_pheno, mean_fitness, var_pheno)
        )
        if pop.generation % 5000 == 0:
            print(
                f"{time.ctime()}, at generation {pop.generation}"
            )
        if pop.generation in self.sample_times:
            sampler.assign(range(pop.N))


def run_sim(L, a, g, mu, optimum=0.0, VS=1, burnin=10):
    # constant effect sizes, +/- a
    sregions = [fwdpy11.ConstantS(0, L, 1, a), fwdpy11.ConstantS(0, L, 1, -a)]
    model = fwdpy11.discrete_demography.from_demes(g, burnin=burnin)
    simlen = model.metadata["total_simulation_length"]
    print("Total generations to run:", simlen)

    ## Set up pop
    initial_sizes = [
        model.metadata["initial_sizes"][i]
        for i in sorted(model.metadata["initial_sizes"].keys())
    ]
    N0 = Ne = initial_sizes[0]
    assert len(initial_sizes) == 1
    Nf = g.demes[-1].epochs[0].start_size

    GSSmo = fwdpy11.GSSmo(
        [
            fwdpy11.Optimum(when=0, optimum=optimum, VS=VS),
        ]
    )

    pdict = {
        "nregions": [],
        "sregions": sregions,
        "recregions": [fwdpy11.BinomialInterval(0, L, 1)],
        "rates": (0.0, mu, None),
        "gvalue": fwdpy11.Additive(scaling=2, gvalue_to_fitness=GSSmo),
        "simlen": simlen,
        "demography": model,
        "prune_selected": False,
    }
    params = fwdpy11.ModelParams(**pdict)

    sample_times = [model.metadata["burnin_time"] + 2 * Ne]

    ## Initialize and evolve population
    recorder = Recorder(data=[], sample_times=sample_times)
    pop = fwdpy11.DiploidPopulation(initial_sizes, L)
    rng = fwdpy11.GSLrng(seed)

    time1 = time.time()
    fwdpy11.evolvets(
        rng, pop, params, 50, recorder=recorder, suppress_table_indexing=True
    )
    time2 = time.time()
    print(f"simulation took {(time2 - time1)/60:0.2f} minutes")
    assert pop.generation == simlen

    return pop, recorder


def get_all_props(ts):
    # get mutations that were fixed between admixed populations
    samples_dict = get_samples_dict(ts)
    t_max = max([k[0] for k in samples_dict.keys()])

    G0 = ts.genotype_matrix(samples=samples_dict[(t_max, 0)])
    G1 = ts.genotype_matrix(samples=samples_dict[(t_max, 1)])
    G = ts.genotype_matrix(samples=samples_dict[(0, 2)])

    ns0 = G0.shape[1]
    ns1 = G1.shape[1]

    fixed_diffs = np.logical_or(
        np.logical_and(G0.sum(axis=1) == 0, G1.sum(axis=1) == ns1),
        np.logical_and(G0.sum(axis=1) == ns0, G1.sum(axis=1) == 0),
    )

    fixed_diff_positions = []
    for is_fixed, m in zip(fixed_diffs, ts.mutations()):
        if is_fixed:
            fixed_diff_positions.append(ts.site(m.site).position)

    print("There are", len(fixed_diff_positions), "fixed differences")

    # get ancestry proportions surrounding fixed differences
    max_dist = 1e6  # 1 Mb = 1 cM at r = 1e-8
    dists = np.linspace(-max_dist, max_dist, 201)

    all_props = []

    for pos in fixed_diff_positions:
        props = np.zeros(len(dists))
        if pos < max_dist or pos > L - max_dist:
            continue
        for j, d in enumerate(dists):
            i = np.argmin(pos + d > x) - 1
            n0 = num_seg[i]
            n1 = num_intro[i]
            props[j] = n1 / (n0 + n1)

        all_props.append(props)
    return all_props


# what else to save?
if __name__ == "__main__":
    # parameters for this simulation
    seed = int(sys.argv[1]) + 1 + 200
    mu = float(sys.argv[2])
    a = float(sys.argv[3])
    print(f"Running mu={mu}, a={a}, with seed={seed}")

    ## Set up parameters
    L = 1e8
    r = 1e-8
    assert L * r == 1

    optimum = 0.0
    VS = 1.0
    expectedVG = 4 * mu * VS
    print("Expected VG:", expectedVG)

    g = set_up_demographic_model()
    pop, recorder = run_sim(L, a, g, mu, optimum=optimum, VS=VS)

    ## Dump to tskit
    print(f"Dumping to tskit")
    ts = pop.dump_tables_to_tskit()
    print(f"Done!")

    ## Count up admixed segments
    print(f"Counting admixture ancestry segments")
    time1 = time.time()
    x, num_seg, num_intro = get_ancestry_counts(ts)
    time2 = time.time()
    print(f"Done, took {(time2 - time1)/60:0.2f} minutes")

    ## Get average admixture proportions
    S = 0
    for i, (n0, n1) in enumerate(zip(num_seg, num_intro)):
        dx = x[i + 1] - x[i]
        frac = n1 / (n0 + n1)
        S += dx * frac

    S /= ts.sequence_length
    print("Average admixture proportion:", S)

    all_props = get_all_props(ts)

    ## save data
    # ancestry proportions data
    fname = f"introgressed_ancestry.a_{a}.mu_{mu}.seed_{seed}.pkl"
    with open(fname, "wb+") as fout:
        pickle.dump(all_props, fout)

    # save recorder information
    # the recorder object itself is too large to store many copies of (15Mb),
    # so can we summarize the data more succinctly and efficiently
    #fname = f"recorder.a_{a}.mu_{mu}.seed_{seed}.pkl"
    #with open(fname, "wb+") as fout:
    #    pickle.dump(recorder, fout)
