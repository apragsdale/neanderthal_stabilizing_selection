import tarfile
import numpy as np
import pickle

import sys
a = float(sys.argv[1])
mu = float(sys.argv[2])

tar = tarfile.open(f"introgressed_ancestry.a_{a}.mu_{mu}.tar.gz")

n_files = len(tar.getmembers())
print("Total num files:", n_files)

data = []
for m in tar:
    f = tar.extractfile(m)
    data.append(pickle.load(f))

assert len(data) == n_files

all_data = []
for d in data:
    all_data += d


means = np.mean(all_data, axis=0)
np.savez(f"../figures/data/introgressed_ancestry.means.a_{a}.mu_{mu}.npz", means)
