import numpy as np
import matplotlib.pylab as plt, matplotlib
import moments
import demes, demesdraw
import scipy.stats
import os, pickle
import sys


from single_population_stabilizing_selection import *

# set font sizes
plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=6)
matplotlib.rc("ytick", labelsize=6)
matplotlib.rc("axes", labelsize=7)
matplotlib.rc("axes", titlesize=7)
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

grid = (3, 8)

fig = plt.figure(3, figsize=(6.5, 4.5))
fig.clf()

# plot the model
ax1 = plt.subplot2grid(grid, (0, 0), colspan=2, rowspan=2)
demesdraw.tubes(g, ax=ax1, labels=None)
ax1.set_ylabel("Time ago (ka)")
ax1.set_xlabel("       Human         Neand")
ax1.text(2e3, 270000, "$f=0.05$", va="center", ha="center", fontsize=6)
ax1.text(2e3, 65000, "$f=0.02$", va="center", ha="center", fontsize=6)

ax1.set_yticks(np.arange(0, 800000, 100000))
ax1.set_yticklabels(["0", "100", "200", "300", "400", "500", "600", "700"])
# plot VGs (for SD=0.05 and SD=0.01)
ax2 = plt.subplot2grid(grid, (0, 2), colspan=6)

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
ax2.set_xlabel("Time ago (ka)")
ax2.set_ylabel("$V_A$")

ax2.text(700, 0.095, "$\sigma_M=0.05$", va="center", ha="center", fontsize=6)
ax2.text(700, 0.041, "$\sigma_M=0.01$", va="center", ha="center", fontsize=6)
# ax2.legend(fontsize=5, loc="lower left", frameon=False)

ax3 = plt.subplot2grid(grid, (1, 2), colspan=3)
data = pickle.load(open("data/h2.neand_to_human.pkl", "rb"))  # change to neand to human

SD = 0.05
t = sorted(data[SD]["traj"].keys())
VG_intro = [data[SD]["traj"][k][0] for k in t]
VG_seg = [data[SD]["traj"][k][1] for k in t]
t = [2 * 1e4 * 25 * _ / 1000 for _ in t]

ax3.plot(x_05, y_05, color=colors[0], label="Total")
ax3.plot(t, VG_intro, color=colors[2], label="Introgressed")
ax3.plot(t, VG_seg, color=colors[3], label="Non-introgressed")
ax3.legend()

ax3.set_ylim(bottom=0)

ax3.set_ylabel("$V_A$")
ax3.set_xlabel("Time since admixture (ka)")
ax3.set_title("$\sigma_M=0.05$")

ax4 = plt.subplot2grid(grid, (1, 5), colspan=3)

SD = 0.01
t = sorted(data[SD]["traj"].keys())
VG_intro = [data[SD]["traj"][k][0] for k in t]
VG_seg = [data[SD]["traj"][k][1] for k in t]
t = [2 * 1e4 * 25 * _ / 1000 for _ in t]

ax4.plot(x_01, y_01, color=colors[0], label="Total")
ax4.plot(t, VG_intro, color=colors[2], label="Introgressed")
ax4.plot(t, VG_seg, color=colors[3], label="Non-introgressed")
# ax4.legend()
ax4.set_title("$\sigma_M=0.01$")

ax4.set_ylim(bottom=0)

ax4.set_ylabel("$V_A$")
ax4.set_xlabel("Time since admixture (ka)")
ax3.set_xlim(left=-4)
ax4.set_xlim(ax3.get_xlim())

ax5 = plt.subplot2grid(grid, (2, 0), colspan=4)
SD = 0.05
t = sorted(data[SD]["traj"].keys())
h2_intro = [data[SD]["traj"][k][2] for k in t]
h2_seg = [data[SD]["traj"][k][3] for k in t]
h2_seg_w = [data[SD]["traj"][k][4] for k in t]
t = [2 * 1e4 * 25 * _ / 1000 for _ in t]

ax5.plot(t, h2_intro, color=colors[2], label="Introgressed")
ax5.plot(t, h2_seg, color=colors[3], label="Non-introgressed")
ax5.plot(t, h2_seg_w, color=colors[4], label="Non-intro. (AF-weighted)")
ax5.set_ylim(bottom=0)
ax5.legend()

ax5.set_ylabel("$h^2$ per SNP")
ax5.set_xlabel("Time since admixture (ka)")
ax5.set_title("$\sigma_M=0.05$")

ax6 = plt.subplot2grid(grid, (2, 4), colspan=4)
SD = 0.01
t = sorted(data[SD]["traj"].keys())
h2_intro = [data[SD]["traj"][k][2] for k in t]
h2_seg = [data[SD]["traj"][k][3] for k in t]
h2_seg_w = [data[SD]["traj"][k][4] for k in t]
t = [2 * 1e4 * 25 * _ / 1000 for _ in t]

ax6.plot(t, h2_intro, color=colors[2], label="Introgressed")
ax6.plot(t, h2_seg, color=colors[3], label="Non-introgressed")
ax6.plot(t, h2_seg_w, color=colors[4], label="Non-intro. (AF-weighted)")
ax6.set_ylim(bottom=0)
# ax6.legend()
ax6.set_title("$\sigma_M=0.01$")

ax6.set_ylabel("$h^2$ per SNP")
ax6.set_xlabel("Time since admixture (ka)")

plt.tight_layout()
plt.subplots_adjust(hspace=0.6)
plt.savefig("neanderthal_admixture.pdf")


#####

SD = 0.01
data = pickle.load(open(f"data/VG_traj.SD_{SD}.human_neand.pkl", "rb"))
x_01 = (250000 - data["t"][49:66]) / 1000
y_01 = data["neanderthal"][49:66]

SD = 0.05
data = pickle.load(open(f"data/VG_traj.SD_{SD}.human_neand.pkl", "rb"))
x_05 = (250000 - data["t"][49:66]) / 1000
y_05 = data["neanderthal"][49:66]


fig2 = plt.figure(4, figsize=(5, 3.5))
fig2.clf()

ax1 = plt.subplot(2, 2, 1)
data = pickle.load(open("data/h2.human_to_neand.pkl", "rb"))  # change to neand to human

SD = 0.05
t = sorted(data[SD]["traj"].keys())
VG_intro = [data[SD]["traj"][k][0] for k in t]
VG_seg = [data[SD]["traj"][k][1] for k in t]
t = [2 * 1e4 * 25 * _ / 1000 for _ in t]

ax1.plot(x_05, y_05, color=colors[0], label="Total")
ax1.plot(t, VG_intro, color=colors[2], label="Introgressed")
ax1.plot(t, VG_seg, color=colors[3], label="Non-introgressed")
ax1.legend()

ax1.set_ylim(bottom=0)

ax1.set_ylabel("$V_A$")
ax1.set_xlabel("Time since admixture (ka)")
ax1.set_title("$\sigma_M=0.05$")

ax2 = plt.subplot(2, 2, 2)

SD = 0.01
t = sorted(data[SD]["traj"].keys())
VG_intro = [data[SD]["traj"][k][0] for k in t]
VG_seg = [data[SD]["traj"][k][1] for k in t]
t = [2 * 1e4 * 25 * _ / 1000 for _ in t]

ax2.plot(x_01, y_01, color=colors[0], label="Total")
ax2.plot(t, VG_intro, color=colors[2], label="Introgressed")
ax2.plot(t, VG_seg, color=colors[3], label="Non-introgressed")
# ax2.legend()
ax2.set_title("$\sigma_M=0.01$")

ax2.set_ylim(bottom=0)

ax2.set_ylabel("$V_A$")
ax2.set_xlabel("Time since admixture (ka)")
ax1.set_xlim(left=-4)
ax2.set_xlim(ax3.get_xlim())

ax3 = plt.subplot(2, 2, 3)
SD = 0.05
t = sorted(data[SD]["traj"].keys())
h2_intro = [data[SD]["traj"][k][2] for k in t]
h2_seg = [data[SD]["traj"][k][3] for k in t]
h2_seg_w = [data[SD]["traj"][k][4] for k in t]
t = [2 * 1e4 * 25 * _ / 1000 for _ in t]

ax3.plot(t, h2_intro, color=colors[2], label="Introgressed")
ax3.plot(t, h2_seg, color=colors[3], label="Non-introgressed")
ax3.plot(t, h2_seg_w, color=colors[4], label="Non-intro. (AF-weighted)")
ax3.set_ylim(bottom=0)
ax3.legend()

ax3.set_ylabel("$h^2$ per SNP")
ax3.set_xlabel("Time since admixture (ka)")
ax3.set_title("$\sigma_M=0.05$")

ax4 = plt.subplot(2, 2, 4)
SD = 0.01
t = sorted(data[SD]["traj"].keys())
h2_intro = [data[SD]["traj"][k][2] for k in t]
h2_seg = [data[SD]["traj"][k][3] for k in t]
h2_seg_w = [data[SD]["traj"][k][4] for k in t]
t = [2 * 1e4 * 25 * _ / 1000 for _ in t]

ax4.plot(t, h2_intro, color=colors[2], label="Introgressed")
ax4.plot(t, h2_seg, color=colors[3], label="Non-introgressed")
ax4.plot(t, h2_seg_w, color=colors[4], label="Non-intro. (AF-weighted)")
ax4.set_ylim(bottom=0)
# ax4.legend()
ax4.set_title("$\sigma_M=0.01$")

ax4.set_ylabel("$h^2$ per SNP")
ax4.set_xlabel("Time since admixture (ka)")

plt.tight_layout()
plt.subplots_adjust(hspace=0.7)
plt.savefig("human_admixture.pdf")


