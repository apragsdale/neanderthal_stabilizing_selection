import fwdpy11
import numpy as np
from dataclasses import dataclass
from typing import List

import time
import demes

## Set up parameters
L = 1e8
r = 1e-8
assert L * r == 1

mu = 0.025
VS = 1.0
expectedVG = 4 * mu * VS
print("Expected VG:", expectedVG)


## Set up genomic architecture
optimum = 0.0
SD = 0.05
# sregions = [fwdpy11.GaussianS(0, L, 1, SD)]
sregions = [
    fwdpy11.GaussianS(i * 1e6 + 475000, i * 1e6 + 525000, 1, SD)
    for i in range(int(L / 1e6))
]

## Set up demographic model
g = demes.load("demog_reciprocal.yml")
burnin = 10
model = fwdpy11.discrete_demography.from_demes(g, burnin=burnin)
simlen = model.metadata["total_simulation_length"]


## Set up pop
initial_sizes = [
    model.metadata["initial_sizes"][i]
    for i in sorted(model.metadata["initial_sizes"].keys())
]
N0 = initial_sizes[0]
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

## set up recorders
@dataclass
class SimData:
    generation: int
    demes_ids: List[int]
    mean_phenotype: List[float]
    mean_fitness: List[float]
    var_phenotype: List[float]


sample_times = [model.metadata["burnin_time"] + 20000 + i for i in range(101)]
# and any generations directly following this time?


@dataclass
class Recorder:
    data: list

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
        if pop.generation % 1000 == 0:
            print(
                f"  at generation {pop.generation} of {simlen}, "
                f"{(time.time() - time1) / 60:0.2f} minutes"
            )
        if pop.generation in sample_times:
            sampler.assign(range(pop.N))


## Initialize and evolve population
recorder = Recorder(data=[])
pop = fwdpy11.DiploidPopulation(initial_sizes, L)
rng = fwdpy11.GSLrng(424242)

time1 = time.time()
fwdpy11.evolvets(rng, pop, params, 50, recorder=recorder, suppress_table_indexing=True)
time2 = time.time()
print(f"simulation took {(time2 - time1)/60:0.2f} minutes")
assert pop.generation == simlen
