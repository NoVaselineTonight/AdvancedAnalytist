"""Confidence display in the 2D replay: seen / predicted dots, uncertainty halos, team-uncertain markers,
jersey numbers / lineup slots, and the tracking-quality lane."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QRect  # noqa: E402
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter  # noqa: E402

from replay.gui import motion  # noqa: E402
from replay.gui import widgets as w  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QGuiApplication.instance() or QGuiApplication([])


def _track(i, team, y, n=60, f=None, **extra):
    tr = {"id": i, "team": team, "type": "player", "i0": 0, "x": [500 + 10 * k for k in range(n)], "y": [y] * n,
          "f": f or [1] * n}
    tr.update(extra)
    return tr


def _data(tracks, n=60, **extra):
    d = {"fps": 10.0, "t": [k / 10 for k in range(n)], "calib": [1] * n, "tracks": tracks, "ball": None,
         "track_info": {}}
    d.update(extra)
    return d


def _render(d, t, **kw):
    img = QImage(560, 320, QImage.Format.Format_ARGB32)
    img.fill(QColor("#000000"))
    qp = QPainter(img)
    w.paint_court(qp, QRect(0, 0, 560, 320), d, t, colors={"A": "#ff0000", "B": "#0000ff"}, **kw)
    qp.end()
    return img


def _reddish(img):
    n = 0
    for y in range(0, img.height(), 2):
        for x in range(0, img.width(), 2):
            c = img.pixelColor(x, y)
            if c.red() > 150 and c.green() < 120 and c.blue() < 120:
                n += 1
    return n


def test_uncertainty_estimate_grows_in_gaps():
    n = 60
    f = [1] * 20 + [2] * 20 + [1] * 20  # filled in by the smoother in the middle
    tr = _track(1, "A", 500, n, f)
    p = motion.build_path(tr, [k / 10 for k in range(n)], 0.1)
    assert p.uncertainty(1.0) == pytest.approx(motion.SIG_SEEN)
    mid = p.uncertainty(3.0)
    assert mid > p.uncertainty(2.1) > motion.SIG_SEEN
    assert mid <= motion.SIG_MAX
    # coasting after the end of the data: grows with time
    assert p.uncertainty(6.5) > p.uncertainty(6.0) >= motion.SIG_SEEN


def test_uncertainty_from_data_field():
    n = 30
    tr = _track(1, "A", 500, n, u=[40] * n)  # 40 cm
    p = motion.build_path(tr, [k / 10 for k in range(n)], 0.1)
    assert p.uncertainty(1.05) == pytest.approx(0.4)


def test_meta_labels_numbers_slots_and_uncertain():
    d = _data([_track(1, "A", 300), _track(2, "B", 700), _track(3, "A", 1100), _track(4, "B", 900)],
              track_info={"1": {"team": "A", "team_confidence": 0.9, "number": "12"},
                          "2": {"team": "B", "team_confidence": 0.3},
                          "3": {"team": "A", "team_confidence": 1.0, "slot": "A4"},
                          "4": {"team": "B", "team_confidence": 1.0, "id_confidence": 0.2}})
    m = motion.track_meta(d)
    assert m["1"]["label"] == "12" and m["1"]["numbered"] and not m["1"]["uncertain"]
    assert m["2"]["label"] == "2" and m["2"]["uncertain"]
    assert m["3"]["label"] == "P4"
    assert m["4"]["uncertain"]


def test_meta_from_lineup_block():
    d = _data([_track(7, "A", 300)], lineup={"slots": [{"id": "A2", "team": "A", "number": "5", "tracks": [7]}]})
    assert motion.track_meta(d)["7"]["label"] == "5"


def test_quality_levels():
    n = 60
    calib = [1] * n
    calib[50:] = [0] * 10
    tracks = [_track(i, "A" if i % 2 else "B", 100 * i, n) for i in range(1, 11)]
    for tr in tracks[:6]:
        tr["f"] = [1] * 20 + [2] * 20 + [1] * 20  # only 4 seen in the middle, still located
    for tr in tracks[8:]:
        tr["f"] = [1] * 10 + [0] * 50  # 8 located after 1 s
        tr["x"] = tr["x"][:10] + [None] * 50
    q = motion.tracking_quality(_data(tracks, n, calib=calib))
    lev = q["level"]
    assert lev[5] == motion.Q_GOOD
    assert lev[15] == motion.Q_GOOD  # 8 located: fallback threshold
    assert lev[30] == motion.Q_FEW
    assert lev[55] == motion.Q_CALIB_NONE
    st = q["stretches"]
    assert st[0][0] == 0.0 and st[-1][2] == motion.Q_CALIB_NONE
    assert all(a[1] <= b[0] + 1e-9 for a, b in zip(st, st[1:]))
    # with a lineup: complete only with all 10 slots located
    info = {str(tr["id"]): {"slot": f"{tr['team']}{k}"} for k, tr in enumerate(tracks)}
    q2 = motion.tracking_quality(_data(tracks, n, calib=calib, track_info=info))
    assert q2["lineup"] and q2["level"][5] == motion.Q_GOOD and q2["level"][15] == motion.Q_LINEUP


def test_quality_calib_q_and_missing_fields():
    n = 20
    tracks = [_track(i, "A", 100 * i, n) for i in range(1, 11)]
    d = _data(tracks, n, calib_q=[0.9] * 10 + [0.2] * 10)
    lev = motion.tracking_quality(d)["level"]
    assert lev[2] == motion.Q_GOOD and lev[15] == motion.Q_CALIB_POOR
    assert motion.tracking_quality({"t": []})["stretches"] == []
    assert w.quality_stretches(None) == []


def _changed(img, ref):
    return sum(1 for y in range(0, img.height(), 2) for x in range(0, img.width(), 2)
               if img.pixel(x, y) != ref.pixel(x, y))


def test_predicted_player_draws_halo(app):
    n = 60
    f = [1] * 20 + [2] * 20 + [1] * 20
    d = _data([_track(1, "A", 700, n, f)], n)
    empty = _render(_data([], n), 3.0)
    assert _reddish(_render(d, 1.0, confidence=False)) > 5
    pred_off = _changed(_render(d, 3.0, confidence=False), empty)
    pred_on = _changed(_render(d, 3.0, confidence=True), empty)
    assert pred_on > 2 * pred_off  # the halo marks a larger area around a predicted dot
    # a seen player looks the same with or without confidence display (no halo)
    assert _changed(_render(d, 1.0, confidence=True), _render(d, 1.0, confidence=False)) == 0


def test_confidence_render_with_legend_and_markers(app):
    d = _data([_track(1, "A", 300), _track(2, "B", 700)],
              track_info={"1": {"team_confidence": 0.2}, "2": {"number": "23"}})
    img = _render(d, 2.0, confidence=True, legend=True)
    # legend panel (dark) in the top-left corner
    c = img.pixelColor(100, 3)
    assert c.red() < 60 and c.green() < 60
    # "?" badge: pale yellow pixels near the uncertain dot
    pale = sum(1 for y in range(img.height()) for x in range(0, img.width(), 1)
               if (lambda q: q.red() > 240 and q.green() > 230 and 190 < q.blue() < 225)(img.pixelColor(x, y)))
    assert pale > 3


def test_lineup_file_numbers_change_over_time():
    d = _data([_track(500, "A", 300), _track(9, "B", 700)])
    doc = {"slots": {"A1": {"id": 500, "team": "A", "numbers": [12, 7],
                            "pieces": [[40, 0.0, 2.0, 12], [41, 3.0, 5.9, 7], [42, 6.0, 6.0, -1]]},
                     "A2": {"id": 999, "team": "A", "numbers": [], "pieces": []}}}  # not in the data: ignored
    lu = motion.lineup_from_file(doc, d)
    assert [s["id"] for s in lu["slots"]] == ["A1"]
    d["lineup"] = lu
    mt = motion.track_meta(d)["500"]
    assert mt["slot"] == "A1" and mt["label"] == "P1"
    assert motion.label_at(mt, 1.0) == ("12", True)
    assert motion.label_at(mt, 4.0) == ("7", True)
    assert motion.label_at(mt, 2.4) == ("12", True)  # short hole: nearest piece
    assert motion.label_at(motion.track_meta(d)["9"], 1.0) == ("9", False)
    assert motion.lineup_from_file({}, d) is None


def test_state_and_sigma_fields_from_smoother():
    n = 30
    st = [1] * 10 + [4] * 10 + [1] * 10  # occluded in the middle
    tr = _track(1, "A", 500, n, s=[20] * 10 + [120] * 10 + [20] * 10, st=st)  # sigma in cm
    p = motion.build_path(tr, [k / 10 for k in range(n)], 0.1)
    assert p.f[15] == 2 and p.f[5] == 1
    assert p.uncertainty(1.5) == pytest.approx(1.2)
    tr2 = _track(2, "A", 500, n, sigma=[0.3] * n)  # metres
    assert motion.build_path(tr2, [k / 10 for k in range(n)], 0.1).uncertainty(0.5) == pytest.approx(0.3)
    q = motion.tracking_quality(_data([tr], n))
    assert q["seen"][15] == 0 and q["located"][15] == 1


def test_paths_are_built_lazily_with_a_time_index(app):
    n = 600
    tracks = []
    for k in range(40):  # 40 short tracks spread over 60 s
        tr = _track(k + 1, "A" if k % 2 else "B", 300 + 20 * k, n)
        lo = 15 * k
        tr["x"] = [None] * lo + tr["x"][lo:lo + 30] + [None] * (n - lo - 30)
        tracks.append(tr)
    d = _data(tracks, n)
    m = motion.model_for(d)
    paths = m.paths(0, 1.5)
    alive = m.alive(10.0, 0.25, 4.1)
    assert 0 < len(alive) < 15 and all(paths._p[i] is None for i in range(len(tracks)))
    _render(d, 10.0)
    built = [i for i, p in enumerate(paths._p) if p is not None]
    assert built and set(built) <= set(alive)  # only tracks alive at t were built
    # same picture as building everything up front
    d2 = _data([dict(tr) for tr in tracks], n)
    list(motion.model_for(d2).paths(0, 1.5))
    assert _render(d2, 10.0) == _render(d, 10.0)
    while not paths.build_some(0.001):
        pass
    assert all(p is not None for p in paths._p)


def test_dm_vectorised_matches_loop():
    import math
    f = [1, 2, 2, 1, 2, 2, 2, 1, 1, 2]
    t = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 2.0, 2.1, 2.2]  # a long gap: two runs
    p = motion.TrackPath(t, [1.0] * 10, [1.0] * 10, f, 0.1, bridge=0.5)
    exp = [0.0, 0.1, 0.1, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.1]
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(p.dm, exp))


def test_names_under_dots_and_dot_hits(app):
    d = _data([_track(1, "A", 300), _track(2, "B", 700)],
              track_info={"1": {"number": "23"}},
              roster_names={"tracks": {"2": "Matti Meikäläinen"}, "numbers": {"A": {"23": "Virtanen"}}})
    dots = []
    plain = _render(d, 2.0)
    img = QImage(560, 320, QImage.Format.Format_ARGB32)
    img.fill(QColor("#000000"))
    qp = QPainter(img)
    w.paint_court(qp, QRect(0, 0, 560, 320), d, 2.0, colors={"A": "#ff0000", "B": "#0000ff"}, names=True,
                  dots=dots)
    qp.end()
    assert img != plain
    assert {(str(tr["id"]), text, num) for tr, _x, _y, _r, text, num in dots} == {("1", "23", True),
                                                                                  ("2", "2", False)}
    assert w._short_name("Matti Meikäläinen") == "Meikäläinen"
    assert w._short_name("Al") == "Al"
