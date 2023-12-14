"""
Run the fwdpy11 simulation and get ancestry proportions around introgressed
alleles that were fixed differences between populations.
"""


import fwdpy11
import tskit
import numpy as np
from dataclasses import dataclass
from typing import List
from collections import defaultdict
import pickle
import gzip
import time
import demes
import sys, os


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
        n0 = np.sum(sample_counts[samples_dict[(max_t, 0)]])
        n1 = np.sum(sample_counts[samples_dict[(max_t, 1)]])
        assert n0 + n1 == num_samples
        num_seg[i] = n0
        num_intro[i] = n1

    x = np.array(list(ts2.breakpoints()))
    return x, num_seg, num_intro


def set_up_demographic_model(Ne=10000):
    ## Set up demographic model
    gen = 25
    T_split = 600000
    T_H_to_N = 250000
    T_N_to_H = 50000
    b = demes.Builder(time_units="years", generation_time=25)
    b.add_deme(
        "Human",
        epochs=[
            dict(start_size=Ne, end_time=60000),
            dict(start_size=0.1 * Ne, end_size=2 * Ne),
        ],
    )
    b.add_deme(
        "Neand",
        start_time=T_split,
        ancestors=["Human"],
        epochs=[dict(start_size=0.2 * Ne, end_time=45000)],
    )
    b.add_pulse(sources=["Human"], dest="Neand", proportions=[0.05], time=T_H_to_N)
    b.add_pulse(sources=["Neand"], dest="Human", proportions=[0.02], time=T_N_to_H)
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
    burnin: int

    def __call__(self, pop, sampler):
        if pop.generation > self.burnin:
            md = np.array(pop.diploid_metadata)
            # store lists of mean phenotypes, fitnesses, etc
            deme_ids = sorted(list(set(md["deme"])))
            mean_pheno = [md[md["deme"] == i]["g"].mean() for i in deme_ids]
            mean_fitness = [md[md["deme"] == i]["w"].mean() for i in deme_ids]
            var_pheno = [md[md["deme"] == i]["g"].var() for i in deme_ids]
            self.data.append(
                SimData(pop.generation, deme_ids, mean_pheno, mean_fitness, var_pheno)
            )
        if pop.generation % 1000 == 0:
            print(f"{time.ctime()}, at generation {pop.generation}")
        if pop.generation in self.sample_times:
            sampler.assign(range(pop.N))


def run_sim(L, g, mu, spacing=2e6, win_size=100000, mean=-0.0002, shape=0.5, burnin=10):
    # mutation fitness effects (all deleterious) drawn from a gamma distribution,
    # with mutations falling in regions placed 2 Mb apart
    left = (spacing - win_size) / 2
    right = (spacing + win_size) / 2
    sregions = [
        fwdpy11.GammaS(i * spacing + left, i * spacing + right, 1, mean, shape)
        for i in range(int(L / spacing))
    ]

    model = fwdpy11.discrete_demography.from_demes(g, burnin=burnin)
    simlen = model.metadata["total_simulation_length"]
    print(f"{time.ctime()}, Total generations to run:", simlen)

    ## Set up pop
    initial_sizes = [
        model.metadata["initial_sizes"][i]
        for i in sorted(model.metadata["initial_sizes"].keys())
    ]
    N0 = Ne = initial_sizes[0]
    assert len(initial_sizes) == 1
    Nf = g.demes[-1].epochs[0].start_size

    pdict = {
        "nregions": [],
        "sregions": sregions,
        "recregions": [fwdpy11.BinomialInterval(0, L, 1)],
        "rates": (0.0, mu, None),
        "gvalue": fwdpy11.Multiplicative(scaling=2),
        "simlen": simlen,
        "demography": model,
        "prune_selected": True,
    }
    params = fwdpy11.ModelParams(**pdict)

    # sample just before admixture, both populations, then
    # from Neanderthals at time 120ka and 45ka
    sample_times = [
        simlen - 250000 // 25 - 1,  # sample generation _before_ introgression
        simlen - 120000 // 25,
        simlen - 45000 // 25,
    ]

    ## Initialize and evolve population
    recorder = Recorder(data=[], sample_times=sample_times, burnin=10 * Ne)
    pop = fwdpy11.DiploidPopulation(initial_sizes, L)
    rng = fwdpy11.GSLrng(seed)

    time1 = time.time()
    print(f"{time.ctime()}, starting simulation")
    fwdpy11.evolvets(
        rng, pop, params, 50, recorder=recorder, suppress_table_indexing=True
    )
    time2 = time.time()
    print(f"{time.ctime()}, simulation took {(time2 - time1)/60:0.2f} minutes")
    assert pop.generation == simlen

    return pop, recorder


def recorder_to_dict(recorder, g):
    gens0 = len(recorder.data)
    mp0 = np.zeros(gens0, dtype="float32")
    mf0 = np.zeros(gens0, dtype="float32")
    vp0 = np.zeros(gens0, dtype="float32")
    gens1 = g.demes[1].start_time - g.demes[1].epochs[0].end_time
    mp1 = np.zeros(gens1, dtype="float32")
    mf1 = np.zeros(gens1, dtype="float32")
    vp1 = np.zeros(gens1, dtype="float32")
    j = 0
    for i, d in enumerate(recorder.data):
        mp0[i] = d.mean_phenotype[0]
        mf0[i] = d.mean_fitness[0]
        vp0[i] = d.var_phenotype[0]
        if 1 in d.demes_ids:
            mp1[j] = d.mean_phenotype[1]
            mf1[j] = d.mean_fitness[1]
            vp1[j] = d.var_phenotype[1]
            j += 1
    data = {
        "model": g,
        "A": {"mean_phenotype": mp0, "mean_fitness": mf0, "var_phenotype": vp0},
        "B": {"mean_phenotype": mp1, "mean_fitness": mf1, "var_phenotype": vp1},
    }
    return data


def compress_ancestry_data(x, num_seg, num_intro):
    x2 = [x[0]]
    frac_intro = []
    for xx, ns, ni in zip(x[1:], num_seg, num_intro):
        frac = ni / (ni + ns)
        if len(frac_intro) == 0:
            frac_intro.append(frac)
            x2.append(xx)
        elif frac_intro[-1] == frac:
            x2[-1] = xx
        else:
            frac_intro.append(frac)
            x2.append(xx)
    return np.array(x2), np.array(frac_intro)


def get_average_ancestry(x, frac, spacing=2e6, win_size=1e5):
    # functional windows in middle of windows of size spacing
    assert (spacing / 2) % (win_size / 2) == 0
    assert x[-1] % spacing == 0

    left = (spacing - win_size) / 2
    right = (spacing + win_size) / 2

    func_regions = [
        [i * spacing + left, i * spacing + right] for i in range(int(L / 2e6))
    ]
    half_window = win_size / 2
    spacing_windows = np.arange(0, spacing, half_window)

    windows = np.concatenate(
        [spacing_windows + i * spacing for i in range(int(x[-1] / spacing))] + [[x[-1]]]
    )
    ave_ancestry = np.zeros(len(windows) - 1)

    curr_window = 0
    for x_l, x_r, f in zip(x[:-1], x[1:], frac):
        while x_r >= windows[curr_window + 1]:
            if x_l <= windows[curr_window]:
                ave_ancestry[curr_window] = f
            else:
                ave_ancestry[curr_window] += (
                    f * (windows[curr_window + 1] - x_l) / half_window
                )
            curr_window += 1
            if curr_window == len(ave_ancestry):
                break
        if x_r > windows[curr_window]:
            assert x_r < windows[curr_window + 1]
            ave_ancestry[curr_window] += (
                f * (x_r - max(x_l, windows[curr_window])) / half_window
            )
        if curr_window == len(ave_ancestry):
            break
    return windows, ave_ancestry


if __name__ == "__main__":
    # parameters for this simulation
    seed = int(sys.argv[1]) + 1
    mu = float(sys.argv[2])
    mean = float(sys.argv[3])
    shape = float(sys.argv[4])
    print(
        f"{time.ctime()}, Running mu={mu}, mean={mean}, shape={shape}, with seed={seed}"
    )

    if os.path.exists(
        f"ancestry_data.mean_{mean}.shape_{shape}.mu_{mu}.seed_{seed}.pkl.gz"
    ):
        print(f"{time.ctime()}, Already ran this simulation, exiting")
        exit()

    ## Set up parameters
    L = 1e8
    r = 1e-8
    assert L * r == 1

    Ne = 10000
    g = set_up_demographic_model(Ne=Ne)
    pop, recorder = run_sim(L, g, mu, mean=mean, shape=shape)

    ## Dump to tskit
    print(f"Dumping to tskit")
    ts = pop.dump_tables_to_tskit()
    print(f"Done!")

    ## Count up admixed segments for Neanderthals (45ka and 120ka) and humans (0ka)
    sample_sets = defaultdict(list)
    for s in ts.samples():
        t = ts.node(s).time
        p = ts.node(s).population
        sample_sets[(p, t)].append(s)

    ts_neand1 = ts.simplify(
        samples=sample_sets[(0, 10001.0)]
        + sample_sets[(1, 10001.0)]
        + sample_sets[(1, 4800.0)]
    )
    ts_neand2 = ts.simplify(
        samples=sample_sets[(0, 10001.0)]
        + sample_sets[(1, 10001.0)]
        + sample_sets[(1, 1800.0)]
    )
    ts_human = ts.simplify(
        samples=sample_sets[(0, 10001.0)]
        + sample_sets[(1, 10001.0)]
        + sample_sets[(0, 0.0)]
    )
    print(f"Counting admixture ancestry segments")
    time1 = time.time()
    x_neand1, num_seg_neand1, num_intro_neand1 = get_ancestry_counts(ts_neand1)
    time2 = time.time()
    print(f"Done with Neand 1, took {(time2 - time1)/60:0.2f} minutes")

    time1 = time.time()
    x_neand2, num_seg_neand2, num_intro_neand2 = get_ancestry_counts(ts_neand2)
    time2 = time.time()
    print(f"Done with Neand 2, took {(time2 - time1)/60:0.2f} minutes")

    time1 = time.time()
    x_human, num_seg_human, num_intro_human = get_ancestry_counts(ts_human)
    time2 = time.time()
    print(f"Done with Humans, took {(time2 - time1)/60:0.2f} minutes")

    x2_neand1, frac_intro_neand1 = compress_ancestry_data(
        x_neand1, num_intro_neand1, num_seg_neand1
    )
    x2_neand2, frac_intro_neand2 = compress_ancestry_data(
        x_neand2, num_intro_neand2, num_seg_neand2
    )
    x2_human, frac_intro_human = compress_ancestry_data(
        x_human, num_seg_human, num_intro_human
    )

    ancestry_data = {
        "Neand1": {
            "breakpoints": x2_neand1,
            "frac_intro": frac_intro_neand1,
        },
        "Neand2": {
            "breakpoints": x2_neand2,
            "frac_intro": frac_intro_neand2,
        },
        "Human": {
            "breakpoints": x2_human,
            "frac_intro": frac_intro_human,
        },
    }

    ## save data
    # ancestry proportions data
    fname = f"ancestry_data.mean_{mean}.shape_{shape}.mu_{mu}.seed_{seed}.pkl.gz"
    with gzip.open(fname, "wb+") as fout:
        pickle.dump(ancestry_data, fout)

    # Average ancestries in windows
    windows, ave_ancestry_h = get_average_ancestry(x2_human, frac_intro_human)
    windows, ave_ancestry_n1 = get_average_ancestry(x2_neand1, frac_intro_neand1)
    windows, ave_ancestry_n2 = get_average_ancestry(x2_neand2, frac_intro_neand2)
    window_data = {
        "windows": windows,
        "Human": ave_ancestry_h,
        "Neand1": ave_ancestry_n1,
        "Neand2": ave_ancestry_n2,
    }
    fname = f"window_ancestry_data.mean_{mean}.shape_{shape}.mu_{mu}.seed_{seed}.pkl.gz"
    with gzip.open(fname, "wb+") as fout:
        pickle.dump(window_data, fout)

    # save recorder information
    # the recorder object itself is too large to store many copies of (15Mb),
    # so can we summarize the data more succinctly and efficiently
    compressed_data = recorder_to_dict(recorder, g)
    fname = f"recorder.mean_{mean}.shape_{shape}.mu_{mu}.seed_{seed}.pkl.gz"
    with gzip.open(fname, "wb+") as fout:
        pickle.dump(compressed_data, fout)
