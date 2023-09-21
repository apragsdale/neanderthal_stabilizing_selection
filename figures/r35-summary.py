import numpy as np
import matplotlib.pylab as plt, matplotlib
import moments
import demes, demesdraw
import scipy.stats
import os, pickle
import sys


from single_population_stabilizing_selection import *


def evolve_generation(p, q, D, r, s):
    # with underdominance (s > 0)
    p2 = p - s * p * (1 - p) * (1 - 2 * p)
    q2 = q - s * D * (1 - 2 * p)
    D2 = D - s * D * (1 - 2 * p) ** 2 - r * D
    return p2, q2, D2


def trajectories(p0, q0, D0, generations=1, r=0, s=0):
    p = [p0]
    q = [q0]
    D = [D0]
    for g in range(generations):
        p2, q2, D2 = evolve_generation(p[-1], q[-1], D[-1], r, s)
        p.append(p2)
        q.append(q2)
        D.append(D2)
    return p, q, D


def frequency_dip(a_vals, r_max, gens, p0, ax_q, VS=1, legend=True):
    rs = np.linspace(0, r_max, 101)
    rs_plot = 100 * np.concatenate((-rs[:0:-1], rs))
    for a in a_vals:
        qs = np.zeros(len(rs))
        s = a**2 / 2 / VS
        for i, r_dist in enumerate(rs):
            p, q, D = trajectories(p0, p0, p0 - p0**2, generations=gens, r=r_dist, s=s)
            qs[i] = q[-1]
        qs = np.concatenate((qs[:0:-1], qs))
        ax_q.plot(rs_plot, qs, label=f"$a={a}$")
    if legend:
        ax_q.legend(frameon=False, title="Effect size")
    ax_q.set_ylim(bottom=0)
    ax_q.set_xlabel("Distance from selected locus (cM)")
    ax_q.set_ylabel("Introgressed ancestry")




# set font sizes
plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=6)
matplotlib.rc("ytick", labelsize=6)
matplotlib.rc("axes", labelsize=6)
matplotlib.rc("axes", titlesize=6)
matplotlib.rc("legend", fontsize=6)


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

gen = 25
Ne = 1e4
N_N = 1e3
T_split = 600000

T_H_to_N = 250000
T_N_to_H = 50000
f_H_to_N = 0.05
f_N_to_H = 0.02

T_bottle = 60000
N_bottle = 1000
N_Eur = 20000

b = demes.Builder(time_units="years", generation_time=25)
b.add_deme(
    "Human",
    epochs=[
        dict(end_time=T_bottle, start_size=Ne),
        dict(end_time=0, start_size=N_bottle, end_size=N_Eur),
    ],
)
b.add_deme(
    "Neanderthal",
    start_time=T_split,
    ancestors=["Human"],
    epochs=[dict(end_time=45000, start_size=N_N)],
)
b.add_pulse(
    sources=["Human"], dest="Neanderthal", time=T_H_to_N, proportions=[f_H_to_N]
)
b.add_pulse(
    sources=["Neanderthal"], dest="Human", time=T_N_to_H, proportions=[f_N_to_H]
)
g = b.resolve()

grid = (2, 6)

fig = plt.figure(3, figsize=(4.5, 3))
fig.clf()

# plot the model
ax1 = plt.subplot2grid(grid, (0, 0), colspan=2, rowspan=2)
demesdraw.tubes(g, ax=ax1, labels=None)
ax1.set_ylabel("Time ago (thousand years)")
ax1.set_xlabel("       Human        Neand")
ax1.text(2e3, 270000, "$f=0.05$", va="center", ha="center", fontsize=6)
ax1.text(2e3, 65000, "$f=0.02$", va="center", ha="center", fontsize=6)

ax1.set_yticks(np.arange(0, 800000, 100000))
ax1.set_yticklabels(["0", "100", "200", "300", "400", "500", "600", "700"])

ax2 = plt.subplot2grid(grid, (0, 2), colspan=4)

SD = 0.05
data = pickle.load(open(f"data/VG_traj.SD_{SD}.human_neand.pkl", "rb"))
x_05 = (50000 - data["t"][-52:]) / 1000
y_05 = data["human"][-52:]


ax2.plot(
    [_ / 1000 for _ in data["t"]],
    data["human"],
    c=colors[0],
    label=f"Human ($\sigma_M={SD}$)",
)
ax2.plot(
    [_ / 1000 for _ in data["t"]],
    data["neanderthal"],
    c=colors[1],
    label=f"Neand ($\sigma_M={SD}$)",
)

SD = 0.01
data = pickle.load(open(f"data/VG_traj.SD_{SD}.human_neand.pkl", "rb"))
x_01 = (50000 - data["t"][-52:]) / 1000
y_01 = data["human"][-52:]

ax2.plot(
    [_ / 1000 for _ in data["t"]],
    data["human"],
    "-",
    lw=1,
    label=f"Human ($\sigma_M={SD}$)",
)
ax2.plot(
    [_ / 1000 for _ in data["t"]],
    data["neanderthal"],
    "-",
    lw=1,
    label=f"Neand ($\sigma_M={SD}$)",
)

ax2.set_ylim(bottom=0)
ax2.invert_xaxis()
ax2.set_xlabel("Time ago (thousand years)")
ax2.set_ylabel("Genetic variance")

ax2.text(700, 0.095, "$\sigma_M=0.05$", va="center", ha="center", fontsize=6)
ax2.text(700, 0.041, "$\sigma_M=0.01$", va="center", ha="center", fontsize=6)

ax3 = plt.subplot2grid(grid, (1, 2), colspan=4)
frequency_dip([0.01, 0.02, 0.05], 1e-2, 2000, 0.05, ax3)

plt.tight_layout()
plt.subplots_adjust(hspace=0.4, top=0.98, bottom=0.12, left=0.1, right=0.99)

fig.text(0.02, 0.97, "A", fontsize=7, va="center", ha="center")
fig.text(0.33, 0.97, "B", fontsize=7, va="center", ha="center")
fig.text(0.33, 0.49, "C", fontsize=7, va="center", ha="center")
plt.savefig("r35-summary.pdf")

