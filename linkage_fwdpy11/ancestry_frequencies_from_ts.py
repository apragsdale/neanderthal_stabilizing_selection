"""
We assume we have a preserved generation, pre-admixture, as samples. We need to
store all samples from this generation.

We also assume that this generation is the earliest generation from which we have
saved ancient samples.

Strategy:
- Decapitate tree sequence to these samples. Those samples should now be roots
- Store the number of samples at the target time below each of these roots
- Use edge diffs to update these counts from tree to tree
"""

import tskit
from collections import defaultdict
import numpy as np

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
        n0 = sum(sample_counts[samples_dict[(max_t, 0)]])
        n1 = sum(sample_counts[samples_dict[(max_t, 1)]])
        assert n0 + n1 == num_samples
        num_seg[i] = n0
        num_intro[i] = n1


    x = np.array(list(ts2.breakpoints()))
    return x, num_seg, num_intro

if __name__ == "__main__":
    for i, (n0, n1) in enumerate(zip(num_seg, num_intro)):
        L = x[i + 1] - x[i]
        f = n1 / (n0 + n1)
        S += L * f

    S /= ts2.sequence_length
    print("Average admixture proportion:", S)
