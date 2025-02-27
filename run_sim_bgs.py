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
L_win = 1e6
assert L % L_win == 0

u = 5e-9

frac_sel = 0.05
mu = u * L * frac_sel

shape = 0.15
scale = 0.05
mean = shape * scale

sregions = [
    fwdpy11.GammaS(
        (i + 0.5 - frac_sel / 2) * 1e6, (i + 0.5 + frac_sel / 2) * 1e6, 1, -mean, shape
    )
    for i in range(int(L / 1e6))
]

#gamma = 5
#s = gamma / 2 / 1e4

#sregions = [
#    fwdpy11.ConstantS(
#        (i + 0.5 - frac_sel / 2) * 1e6, (i + 0.5 + frac_sel / 2) * 1e6, 1, -s
#    )
#    for i in range(int(L / 1e6))
#]

## Set up demographic model
g = demes.load("demog.yml")
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

## set up recorders
@dataclass
class SimData:
    generation: int
    demes_ids: List[int]
    mean_phenotype: List[float]
    mean_fitness: List[float]
    var_phenotype: List[float]


# We want to be able to trace lineages back through either the human
# or the neanderthal branches, so we record certain generations of the
# simulation, including
# - last generation of ancestral population
# - first generations of humans and neanderthals
# - generations before and after each admixture event
# - last generation of neanderthals
# or just preserve every ~ 1000 generations? would be 600000 / 1000 = 600 gens

# We also want to track allele frequencies after admixture - store
# generations following admixture

admix_times = [model.metadata["burnin_time"] + gens for gens in [14000, 22000]]
neanderthal_end = model.metadata["burnin_time"] + 22200

sample_times = [model.metadata["burnin_time"], model.metadata["burnin_time"] + 1]
for at in admix_times:
    # how many generations to keep?
    for i in range(101):
        sample_times.append(at + i)


@dataclass
class Recorder:
    data: list

    def __call__(self, pop, sampler):
        md = np.array(pop.diploid_metadata)
        # general properties of the population
        # store lists of mean phenotypes and fitness, and var(pheno)
        deme_ids = sorted(list(set(md["deme"])))
        mean_pheno = [md[md["deme"] == i]["g"].mean() for i in deme_ids]
        mean_fitness = [md[md["deme"] == i]["w"].mean() for i in deme_ids]
        var_pheno = [md[md["deme"] == i]["g"].var() for i in deme_ids]
        self.data.append(
            SimData(pop.generation, deme_ids, mean_pheno, mean_fitness, var_pheno)
        )

        if pop.generation in sample_times:
            sampler.assign(np.arange(0, pop.N))

        if pop.generation % 1000 == 0:
            print("  at generation", pop.generation, "of", simlen)


## Initialize and evolve full population
recorder = Recorder(data=[])
pop = fwdpy11.DiploidPopulation(initial_sizes, L)
rng = fwdpy11.GSLrng(424242)

time1 = time.time()
fwdpy11.evolvets(rng, pop, params, 100, recorder=recorder, suppress_table_indexing=True)
time2 = time.time()
print(f"simulation took {(time2 - time1)/60:0.2f} minutes")
assert pop.generation == simlen
