import tarfile, gzip, pickle
import matplotlib.pylab as plt
import numpy as np

import matplotlib

# set font sizes
plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=6)
matplotlib.rc("ytick", labelsize=6)
matplotlib.rc("axes", labelsize=7)
matplotlib.rc("axes", titlesize=7)
matplotlib.rc("legend", fontsize=6)


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


tar = tarfile.open("window_ancestry_data.mean_-0.0002.shape_1000.mu_0.01.tar.gz")
n_files = len(tar.getmembers())
print("Total num files:", n_files)

n_regions = 50
n_windows = 40

for i, m in enumerate(tar):
    f = tar.extractfile(m)
    data = pickle.load(gzip.open(f, "rb"))
    if i == 0:
        windows = data["windows"]
        human = np.zeros((n_files, len(windows) - 1))
        neand1 = np.zeros((n_files, len(windows) - 1))
        neand2 = np.zeros((n_files, len(windows) - 1))
    human[i] = data["Human"]
    neand1[i] = data["Neand1"]
    neand2[i] = data["Neand2"]

mean_human = (
    np.sum(human, axis=0).reshape(n_regions, n_windows).sum(axis=0)
    / n_files
    / n_regions
)
mean_neand1 = (
    np.sum(neand1, axis=0).reshape(n_regions, n_windows).sum(axis=0)
    / n_files
    / n_regions
)
mean_neand2 = (
    np.sum(neand2, axis=0).reshape(n_regions, n_windows).sum(axis=0)
    / n_files
    / n_regions
)

dh = np.array([0] * n_windows)
dn1 = np.array([0] * n_windows)
dn2 = np.array([0] * n_windows)
tot = np.array([0] * n_windows)
dn1_shared = np.array([0] * n_windows)
dn2_shared = np.array([0] * n_windows)

for j in range(len(human)):
    for i in range(n_regions * n_windows):
        tot[i % 40] += 1
        human_desert = False
        if human[j][i] == 0:
            dh[i % 40] += 1
            human_desert = True
        if neand1[j][i] == 0:
            dn1[i % 40] += 1
            if human_desert:
                dn1_shared[i % 40] += 1
        if neand2[j][i] == 0:
            dn2[i % 40] += 1
            if human_desert:
                dn2_shared[i % 40] += 1


fig = plt.figure(1234, figsize=(6.5, 3))
fig.clf()
ax1 = plt.subplot(1, 2, 1)

ax1.fill_betweenx(
    (0, 0.1),
    0.95,
    1.05,
    color=colors[3],
    edgecolor=None,
    alpha=0.2,
    label="Functional region",
)
ax1.step(
    windows[:41] / 1e6, np.concatenate(([mean_human[0]], mean_human)), label="Human"
)
ax1.step(
    windows[:41] / 1e6,
    np.concatenate(([mean_neand1[0]], mean_neand1)),
    label="Neanderthal",
)

ax1.set_xticks([-0.05, 0.2, 0.45, 0.7, 0.95, 1.05, 1.3, 1.55, 1.8, 2.05])
ax1.set_xticklabels([-1.0, -0.75, -0.5, -0.25, 0, 0, 0.25, 0.5, 0.75, 1.0])
ax1.legend(frameon=False)
ax1.set_xlabel("Distance from functional region (cM)")
ax1.set_ylabel("Proportion introgressed ancestry")
ax1.set_ylim((0, 0.1))

ax2 = plt.subplot(1, 2, 2)

ax2.fill_betweenx(
    (0, 1),
    0.95,
    1.05,
    color=colors[3],
    edgecolor=None,
    alpha=0.2,
    label="Functional region",
)
fh = dh / tot
fn = dn2 / tot
fs = dn2_shared / tot
ax2.step(windows[:41] / 1e6, np.concatenate(([fh[0]], fh)), label="Human")
ax2.step(
    windows[:41] / 1e6,
    np.concatenate(([fn[0]], fn)),
    label="Neanderthal",
)
ax2.step(
    windows[:41] / 1e6,
    np.concatenate(([fs[0]], fs)),
    label="Shared",
)

ax2.set_xticks([-0.05, 0.2, 0.45, 0.7, 0.95, 1.05, 1.3, 1.55, 1.8, 2.05])
ax2.set_xticklabels([-1.0, -0.75, -0.5, -0.25, 0, 0, 0.25, 0.5, 0.75, 1.0])
ax2.legend(frameon=False)
ax2.set_xlabel("Distance from functional region (cM)")
ax2.set_ylabel("Probability ancestry desert")
ax2.set_ylim((0.65, 0.95))


fig.tight_layout()
plt.savefig("bgs_introgression_deserts.pdf")
