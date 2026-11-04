import math

import numpy as np

from replay import court


def test_dimensions_and_origin():
    assert court.LENGTH == 28.0 and court.WIDTH == 15.0
    kp = court.KEYPOINTS
    assert kp["L_corner_near"] == (0.0, 0.0)
    assert kp["R_corner_far"] == (28.0, 15.0)
    assert kp["mid_near"] == (14.0, 0.0)


def test_keypoints_mirror_symmetric():
    kp = court.KEYPOINTS
    for name, (x, y) in kp.items():
        if name.startswith("L_"):
            rx, ry = kp["R_" + name[2:]]
            assert math.isclose(rx, 28.0 - x) and math.isclose(ry, y)


def test_keypoints_inside_court():
    pts = np.array(list(court.KEYPOINTS.values()))
    assert court.in_court(pts).all()


def test_fiba_key_dimensions():
    kp = court.KEYPOINTS
    # paint is 4.9 m wide, free-throw line 5.8 m from the baseline
    assert math.isclose(kp["L_ft_far"][1] - kp["L_ft_near"][1], 4.9)
    assert math.isclose(kp["L_ft_near"][0], 5.8)
    # top of the 3-point arc is 6.75 m from the point below the basket
    assert math.isclose(kp["L_three_top"][0] - court.BASKET_X, 6.75)


def test_three_point_break_lies_on_arc():
    x, y = court.KEYPOINTS["L_three_break_near"]
    assert math.isclose(y, 0.9)
    assert math.isclose(math.hypot(x - court.BASKET_X, y - court.MID_Y), court.THREE_R, rel_tol=1e-9)


def test_in_court_margin():
    pts = np.array([[-0.5, 7.0], [28.5, 7.0], [14.0, 15.9], [-2.0, 7.0], [np.nan, 1.0]])
    assert court.in_court(pts, margin=0.0).tolist() == [False, False, False, False, False]
    assert court.in_court(pts, margin=1.0).tolist() == [True, True, True, False, False]


def test_polylines_within_bounds():
    for line in court.court_polylines():
        assert line.shape[1] == 2
        assert court.in_court(line, margin=1e-9).all()
