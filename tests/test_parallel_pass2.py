import numpy as np

from replay import parallel_pass2 as pp


def test_segments_split_only_at_long_breaks(tmp_path):
    t = np.r_[np.arange(0, 300, 0.1), np.arange(330, 600, 0.1)]  # 30 s hole (skipped break)
    segs = pp.segments(tmp_path, t)
    assert len(segs) == 2 and segs[0] == (0, 3000) and segs[1][1] == len(t)
    t2 = np.r_[np.arange(0, 30, 0.1), np.arange(55, 600, 0.1)]  # break too close to the start: no tiny segment
    assert pp.segments(tmp_path, t2) == [(0, len(t2))]
    assert pp.segments(tmp_path, np.arange(0, 600, 0.1)) == [(0, 6000)]
