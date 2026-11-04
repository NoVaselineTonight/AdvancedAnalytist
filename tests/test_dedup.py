import numpy as np

from replay.dedup import best_per_sample


def test_one_detection_per_sample_is_the_best_one():
    ii = np.array([5, 3, 5, 3, 4])
    conf = np.array([0.4, 0.9, 0.7, 0.9, 0.5])
    R = np.tile(np.eye(2), (5, 1, 1))
    R[1] *= 4.0  # sample 3: equal conf, index 3 has the smaller trace
    idx = best_per_sample(ii, conf, R)
    assert idx.tolist() == [3, 4, 2]
    assert best_per_sample(ii[::-1], conf[::-1], R[::-1]).tolist() == [1, 0, 2]  # order of input irrelevant
