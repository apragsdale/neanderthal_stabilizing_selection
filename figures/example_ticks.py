import matplotlib.pylab as plt
import matplotlib

sFormatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
sFormatter.set_powerlimits((-2, 2))

x = [0, 1]
y1 = [0, 8e-4]
y2 = [0, 8e-5]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 2))
axes[0].plot(x, y1)
axes[1].plot(x, y2)

axes[0].yaxis.set_major_formatter(sFormatter)

fig.savefig("test.png", dpi=300)
