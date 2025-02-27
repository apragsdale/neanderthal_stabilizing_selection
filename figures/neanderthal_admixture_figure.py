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

sFormatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
sFormatter.set_powerlimits((-3, 2))

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


def plot_model(ax):
    demesdraw.tubes(g, ax=ax, labels=None)
    ax.set_ylabel("Time ago (ka)")
    ax.set_xlabel("       Human         Neand")
    ax.text(1.5e4, 270000, "$f=0.05$", va="center", ha="center", fontsize=6)
    ax.text(1.5e4, 65000, "$f=0.02$", va="center", ha="center", fontsize=6)
    ax.set_yticks(np.arange(0, 800000, 100000))
    ax.set_yticklabels(["0", "100", "200", "300", "400", "500", "600", "700"])


def plot_full_VG(ax):
    SD = 0.05  # plot SD = 0.05
    data = pickle.load(open(f"data/VG_traj.SD_{SD}.human_neand.pkl", "rb"))
    x_05 = (50000 - data["t"][-52:]) / 1000
    y_05 = data["human"][-52:]
    ax.plot(
        [_ / 1000 for _ in data["t"]],
        data["human"],
        c=colors[0],
        label=f"Human ($V_M={SD**2:0.2}$)",
    )
    ax.plot(
        [_ / 1000 for _ in data["t"]],
        data["neanderthal"],
        c=colors[1],
        label=f"Neand ($V_M={SD**2:0.2}$)",
    )
    SD = 0.01  # plot SD = 0.01
    data = pickle.load(open(f"data/VG_traj.SD_{SD}.human_neand.pkl", "rb"))
    x_01 = (50000 - data["t"][-52:]) / 1000
    y_01 = data["human"][-52:]
    ax.plot(
        [_ / 1000 for _ in data["t"]],
        data["human"],
        "-",
        lw=1,
        label=f"Human ($V_M={SD**2:0.2}$)",
    )
    ax.plot(
        [_ / 1000 for _ in data["t"]],
        data["neanderthal"],
        "-",
        lw=1,
        label=f"Neand ($V_M={SD**2:0.2}$)",
    )
    ax.set_ylim(bottom=0)
    ax.invert_xaxis()
    ax.set_xlabel("Time ago (ka)")
    ax.set_ylabel("$V_G$")
    ax.text(700, 0.095, f"$V_M={0.05**2:0.2}$", va="center", ha="center", fontsize=6)
    ax.text(700, 0.041, f"$V_M={0.01**2:0.2}$", va="center", ha="center", fontsize=6)


def plot_VG_neand_to_human(ax, SD, legend=True):
    data_all = pickle.load(open(f"data/VG_traj.SD_{SD}.human_neand.pkl", "rb"))
    x = (50000 - data_all["t"][-52:]) / 1000
    y = data_all["human"][-52:]
    data = pickle.load(open("data/h2.neand_to_human.pkl", "rb"))
    t = sorted(data[SD]["traj"].keys())
    VG_intro = [data[SD]["traj"][k][0] for k in t]
    VG_seg = [data[SD]["traj"][k][1] for k in t]
    t = [2 * 1e4 * 25 * _ / 1000 for _ in t]
    ax.plot(t, VG_intro, color=colors[2], label="Introgressed")
    ax.plot(t, VG_seg, color=colors[3], label="Non-introgressed")
    ax.plot(x, y, color=colors[0], label="Total")
    if legend:
        ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("$V_G$")
    ax.set_xlabel("Time since admixture (ka)")
    ax.set_title(f"Neand$\\rightarrow$Human, $V_M={SD**2:0.2}$")
    ax.set_xlim(left=-4)
    ax.yaxis.set_major_formatter(sFormatter)


def plot_h2_neand_to_human(ax, SD, legend=True, xlabel=True, ylabel=True, title=True):
    data = pickle.load(open("data/h2.neand_to_human.pkl", "rb"))
    t = sorted(data[SD]["traj"].keys())
    h2_intro = [data[SD]["traj"][k][2] for k in t]
    h2_seg = [data[SD]["traj"][k][3] for k in t]
    h2_seg_w = [data[SD]["traj"][k][4] for k in t]
    t = [2 * 1e4 * 25 * _ / 1000 for _ in t]
    ax.plot(t, h2_intro, color=colors[2], label="Introgressed")
    ax.plot(t, h2_seg, color=colors[3], label="Non-introgressed")
    ax.plot(t, h2_seg_w, color=colors[4], label="Non-intro. (AF-weighted)")
    ax.set_ylim(bottom=0)
    if legend:
        ax.legend(frameon=False)
    if ylabel:
        ax.set_ylabel(f"$h^2$ per SNP\n($V_M={SD**2:0.2}$)")
    if xlabel:
        ax.set_xlabel("Time since admixture (ka)")
    if title:
        ax.set_title(f"Neand$\\rightarrow$Human")
    ax.yaxis.set_major_formatter(sFormatter)


def plot_VG_human_to_neand(ax, SD, legend=True):
    data_all = pickle.load(open(f"data/VG_traj.SD_{SD}.human_neand.pkl", "rb"))
    x = (250000 - data_all["t"][49:66]) / 1000
    y = data_all["neanderthal"][49:66]
    data = pickle.load(open("data/h2.human_to_neand.pkl", "rb"))
    t = sorted(data[SD]["traj"].keys())
    VG_intro = [data[SD]["traj"][k][0] for k in t]
    VG_seg = [data[SD]["traj"][k][1] for k in t]
    t = [2 * 1e4 * 25 * _ / 1000 for _ in t]
    ax.plot(t, VG_intro, color=colors[2], label="Introgressed")
    ax.plot(t, VG_seg, color=colors[3], label="Non-introgressed")
    ax.plot(x, y, color=colors[0], label="Total")
    if legend:
        ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("$V_G$")
    ax.set_xlabel("Time since admixture (ka)")
    ax.set_title(f"Human$\\rightarrow$Neand, $V_M={SD**2:0.2}$")
    ax.set_xlim(left=-4, right=54)
    ax.yaxis.set_major_formatter(sFormatter)


def plot_h2_human_to_neand(ax, SD, legend=True, xlabel=True, ylabel=True, title=True):
    data = pickle.load(open("data/h2.human_to_neand.pkl", "rb"))
    t = sorted(data[SD]["traj"].keys())
    h2_intro = [data[SD]["traj"][k][2] for k in t]
    h2_seg = [data[SD]["traj"][k][3] for k in t]
    h2_seg_w = [data[SD]["traj"][k][4] for k in t]
    t = [2 * 1e4 * 25 * _ / 1000 for _ in t]
    ax.plot(t, h2_intro, color=colors[2], label="Introgressed")
    ax.plot(t, h2_seg, color=colors[3], label="Non-introgressed")
    ax.plot(t, h2_seg_w, color=colors[4], label="Non-intro. (AF-weighted)")
    ax.set_ylim(bottom=0)
    if legend:
        ax.legend(frameon=False, fontsize=5)
    if ylabel:
        ax.set_ylabel(f"$h^2$ per SNP\n($V_M={SD**2:0.2}$)")
    if xlabel:
        ax.set_xlabel("Time since admixture (ka)")
    if title:
        ax.set_title(f"Human$\\rightarrow$Neand")
    ax.yaxis.set_major_formatter(sFormatter)


grid1 = (2, 8)
grid2 = (3, 2)

fig1 = plt.figure(1, figsize=(6.5, 3))
fig2 = plt.figure(2, figsize=(6.5, 5))
fig1.clf()
fig2.clf()

# plot the model
ax1 = plt.subplot2grid(grid1, (0, 0), colspan=2, rowspan=2, fig=fig1)
plot_model(ax1)

ax2 = plt.subplot2grid(grid1, (0, 2), colspan=3, fig=fig1)
plot_h2_neand_to_human(ax2, 0.05, xlabel=False)

ax3 = plt.subplot2grid(grid1, (1, 2), colspan=3, fig=fig1)
plot_h2_neand_to_human(ax3, 0.01, title=False, legend=False)

ax4 = plt.subplot2grid(grid1, (0, 5), colspan=3, fig=fig1)
plot_h2_human_to_neand(ax4, 0.05, xlabel=False, legend=False)

ax5 = plt.subplot2grid(grid1, (1, 5), colspan=3, fig=fig1)
plot_h2_human_to_neand(ax5, 0.01, title=False, legend=False)

# plot VGs (for SD=0.05 and SD=0.01)
ax6 = plt.subplot2grid(grid2, (0, 0), colspan=2, fig=fig2)
plot_full_VG(ax6)

ax7 = plt.subplot2grid(grid2, (1, 0), fig=fig2)
plot_VG_neand_to_human(ax7, 0.05)

ax8 = plt.subplot2grid(grid2, (1, 1), fig=fig2)
plot_VG_neand_to_human(ax8, 0.01, legend=False)

ax9 = plt.subplot2grid(grid2, (2, 0), fig=fig2)
plot_VG_human_to_neand(ax9, 0.05, legend=False)

ax10 = plt.subplot2grid(grid2, (2, 1), fig=fig2)
plot_VG_human_to_neand(ax10, 0.01, legend=False)

fig1.tight_layout()
fig1.subplots_adjust(
    hspace=0.4, wspace=3, left=0.08, top=0.93, bottom=0.13, right=0.98
)
fig1.text(0.02, 0.97, "A", fontsize=8, va="center", ha="center")
fig1.text(0.27, 0.97, "B", fontsize=8, va="center", ha="center")
fig1.text(0.27, 0.49, "C", fontsize=8, va="center", ha="center")
fig1.text(0.65, 0.97, "D", fontsize=8, va="center", ha="center")
fig1.text(0.65, 0.49, "E", fontsize=8, va="center", ha="center")
fig1.savefig("h2-per-SNP.pdf")

fig2.tight_layout()
fig2.savefig("human-neand-VG.pdf")
