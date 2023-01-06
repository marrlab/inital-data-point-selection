
import random
import numpy as np

def sample_n_per_class(xs, ys, labels, n=50):
    xs_subset, ys_subset = [], []

    for label_id in labels:
        xs_filtered = [x for j, x in enumerate(xs) if ys[j] == label_id]

        xs_sampled = random.sample(xs_filtered, min(n, len(xs_filtered)))

        xs_subset.extend(xs_sampled)
        ys_subset.extend([label_id]*len(xs_sampled))

    return np.array(xs_subset), np.array(ys_subset)
