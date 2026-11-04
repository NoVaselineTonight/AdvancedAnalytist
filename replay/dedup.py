"""One detection per sample and identity, chosen deterministically (tracking-fix plan, Task 1).

An identity can hold two detections of one sample (full box + partial box after tracklet merges / crumb absorption).
Keeping "whichever sorted first" made the foot point flip by 1-3 m between builds and between neighbouring samples.
"""

from __future__ import annotations

import numpy as np


def best_per_sample(ii: np.ndarray, conf: np.ndarray, R: np.ndarray | None = None) -> np.ndarray:
    """Indices into ``ii`` of one detection per distinct sample, sorted by sample: highest ``conf``; ties by the
    smallest covariance trace (``R`` (n, 2, 2), optional); remaining ties by the lowest original index."""
    ii = np.asarray(ii)
    n = len(ii)
    if not n:
        return np.zeros(0, int)
    conf = np.asarray(conf, float)
    tr = np.zeros(n) if R is None else np.trace(np.asarray(R, float).reshape(n, 2, 2), axis1=1, axis2=2)
    tr = np.nan_to_num(tr, nan=np.inf)
    order = np.lexsort((np.arange(n), tr, -np.nan_to_num(conf, nan=-np.inf), ii))
    s = ii[order]
    first = np.r_[True, s[1:] != s[:-1]]
    return order[first]
