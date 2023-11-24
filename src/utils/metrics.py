
import numpy as np
import scipy

def rankme(a, verbose=False):
    s = scipy.linalg.svdvals(a)
    s_normed = s / np.sum(s) + 1e-7
    e = scipy.stats.entropy(s_normed)
    r = np.exp(e)

    if verbose:
        return dict(a=a, s=s, s_normed=s_normed, e=e, r=r)

    return r
