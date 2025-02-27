import matplotlib.pylab as plt
import moments
import numpy as np
import copy

gamma = -5
Ne = 1e4
theta = 1000

def mean_fitness(fs, gamma, Ne, theta):
    n = fs.sample_sizes[0]
    exponand = 2 * theta * np.sum(fs * np.arange(n + 1) / n)
    return (1 + gamma / 2 / Ne) ** exponand

fs = moments.Spectrum(np.zeros(401))

t = [0]
f = [1]

for i in range(100):
    fs.integrate([1], 5 / 100, gamma=gamma)
    t.append(t[-1] + 5 / 100)
    f.append(mean_fitness(fs, gamma, Ne, theta))

fs = fs.split(0, 200, 200)
f0 = copy.copy(f)
f1 = copy.copy(f)

for i in range(100):
    fs.integrate([1, 0.1], 0.7 / 100, gamma=gamma)
    t.append(t[-1] + 0.7 / 100)
    fs2 = fs.copy()
    fs2.unmask_all()
    fs2[-1, -1] = 0
    fs0 = fs2.marginalize([1])
    fs1 = fs2.marginalize([0])
    f0.append(mean_fitness(fs0, gamma, Ne, theta))
    f1.append(mean_fitness(fs1, gamma, Ne, theta))

fs = fs.pulse_migrate(0, 1, 150, 0.1)

for i in range(100):
    fs.integrate([1, 0.1], 0.4 / 100, gamma=gamma)
    t.append(t[-1] + 0.4 / 100)
    fs2 = fs.copy()
    fs2.unmask_all()
    fs2[-1, -1] = 0
    fs0 = fs2.marginalize([1])
    fs1 = fs2.marginalize([0])
    f0.append(mean_fitness(fs0, gamma, Ne, theta))
    f1.append(mean_fitness(fs1, gamma, Ne, theta))

fs = fs.pulse_migrate(1, 0, 150, 0.1)

for i in range(100):
    fs.integrate([1, 0.1], 0.1 / 100, gamma=gamma)
    t.append(t[-1] + 0.1 / 100)
    fs2 = fs.copy()
    fs2.unmask_all()
    fs2[-1, -1] = 0
    fs0 = fs2.marginalize([1])
    fs1 = fs2.marginalize([0])
    f0.append(mean_fitness(fs0, gamma, Ne, theta))
    f1.append(mean_fitness(fs1, gamma, Ne, theta))


t = np.array(t)
t *= 2 * Ne

plt.plot(t, f0, "--", lw=2, label="Moments (H)")
plt.plot(t, f1, "--", lw=2, label="Moments (N)")
plt.legend()

