"""Second-look detector (replay/redetect.py) with a mock detector on synthetic frames."""

from types import SimpleNamespace

import cv2
import numpy as np

from replay import redetect
from replay.detection import PERSON, Detections

W, H = 1280, 720
ORANGE = (0, 110, 240)  # BGR
PURPLE = (150, 30, 110)


class ColourBlobDetector:
    """Finds saturated rectangles (our synthetic people) in the given image."""

    threshold = 0.3

    def __init__(self):
        self.calls = 0

    def detect(self, img):
        self.calls += 1
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = (hsv[..., 1] > 80).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(m)
        boxes = [[x, y, x + w, y + h] for x, y, w, h, a in st[1:] if a > 30]
        if not boxes:
            return Detections.empty()
        b = np.array(boxes, np.float32)
        return Detections(b, np.full(len(b), 0.5, np.float32), np.full(len(b), PERSON, np.int32))


def frame_with(boxes_colours):
    img = np.full((H, W, 3), 128, np.uint8)
    for (x1, y1, x2, y2), col in boxes_colours:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, -1)
    return img


def make_data(n, people):
    """people: list of (bt id, colour, box per sample or None)."""
    rows = []
    for i in range(n):
        for bt, col, boxes in people:
            b = boxes(i)
            if b is not None:
                rows.append([i, *b, 0.9, PERSON, bt, *([np.nan] * 6)])
    t = np.arange(n) / 10.0
    data = SimpleNamespace(k=np.arange(n), frame=np.arange(n) * 3, t=t, rows=np.array(rows, float),
                           motion=[None] + [np.eye(3)] * (n - 1),
                           signature={"process_fps": 10.0, "video": None})
    return data


def moving_box(x0, y=500, h=120, w=40, speed=4.0):
    return lambda i: (x0 + speed * i, y - h, x0 + speed * i + w, y)


def test_gap_probe_fills_gap_and_checks_colour():
    n = 30
    hidden = set(range(10, 16))
    orange = moving_box(300)
    people = [(1, ORANGE, lambda i: None if i in hidden else orange(i)),
              (2, PURPLE, moving_box(800))]
    data = make_data(n, people)
    rc = redetect.RedetectConfig(enabled=True, clusters=False)
    probes = redetect.gap_probes(data, None, rc, (W, H))
    assert probes and all(p.kind == "gap" and p.i in hidden and p.bt == 1 for p in probes)
    p0 = min(probes, key=lambda p: p.priority)
    assert abs(p0.pred[0] - orange(p0.i)[0]) < 3  # interpolated prediction

    def frames(colour_hidden):
        for i in range(n):
            boxes = [(orange(i), colour_hidden if i in hidden else ORANGE), (moving_box(800)(i), PURPLE)]
            yield i, frame_with(boxes)

    want = sorted({p.i for p in probes} | {9, 16})
    det = ColourBlobDetector()
    new, st = redetect.run_probes(data, None, probes, ((i, im) for i, im in frames(ORANGE) if i in want), det, rc)
    assert st["gap_added"] == len(probes) == len(new)
    assert set(new[:, 0].astype(int)) <= hidden and (new[:, 7] == 1).all()
    assert np.isfinite(new[:, 8:]).all()  # jersey features computed
    # a purple person where the orange one should be is rejected by the colour gate
    new2, st2 = redetect.run_probes(data, None, probes, ((i, im) for i, im in frames(PURPLE) if i in want),
                                    ColourBlobDetector(), rc)
    assert len(new2) == 0


def test_no_probe_when_already_detected_and_duplicates_rejected():
    n = 20
    orange = moving_box(300)
    # track 1 has a gap, but track 3 (ID switch) sits on the same spot: nothing to look for
    people = [(1, ORANGE, lambda i: None if 5 <= i <= 8 else orange(i)),
              (3, ORANGE, lambda i: orange(i) if 5 <= i <= 8 else None)]
    data = make_data(n, people)
    rc = redetect.RedetectConfig(enabled=True, clusters=False)
    assert redetect.gap_probes(data, None, rc, (W, H)) == []


def test_id_switch_gap_and_budget():
    n = 40
    orange = moving_box(300)
    people = [(1, ORANGE, lambda i: orange(i) if i < 10 else None),
              (2, ORANGE, lambda i: orange(i) if i >= 16 else None)]
    data = make_data(n, people)
    rc = redetect.RedetectConfig(enabled=True, clusters=False, probe_every_s=0.1)
    probes = redetect.gap_probes(data, None, rc, (W, H))
    assert len(probes) == 6 and all(p.bt == -1 for p in probes)
    chosen = redetect.select(probes, 2, rc.gap_share)
    assert len(chosen) == 2 and {p.i for p in chosen} <= {12, 13}  # middle of the gap first


class LinearH:
    """res with an image->court homography 0.02 m per pixel (key area of the left basket at x < 340 px)."""

    def __init__(self, n):
        self.H = [np.diag([0.02, 0.02, 1.0])] * n


def test_cluster_needs_confirmation_at_two_samples():
    from replay import court

    court.set_dimensions(28.0, 15.0)
    n = 10
    a, b = (200, 300, 240, 420), (225, 302, 265, 422)  # overlapping pair in the paint (court x 4-5 m, y 8.4 m)
    hidden = (275, 300, 305, 420)  # third person behind them, missed by pass 1
    data = make_data(n, [(1, ORANGE, lambda i: a), (2, PURPLE, lambda i: b)])
    res = LinearH(n)
    rc = redetect.RedetectConfig(enabled=True, cluster_every_s=10.0)
    probes = redetect.cluster_probes(data, res, rc, (W, H))
    assert len(probes) == 2 and probes[0].group == probes[1].group

    def img(i, with_hidden):
        im = frame_with([(b, PURPLE), (a, ORANGE)] + ([(hidden, ORANGE)] if with_hidden else []))
        im[:, 240] = 128  # keep a and b apart for the blob detector
        return im

    ii = [p.i for p in probes]
    new, st = redetect.run_probes(data, res, probes, [(i, img(i, True)) for i in ii], ColourBlobDetector(), rc)
    assert st["cluster_probes"] == 2 and len(new) == 2
    assert np.allclose(new[0, 1:5], hidden, atol=1.5) and (new[:, 7] == -1).all()
    # seen at only one of the two samples -> not confirmed
    new1, _ = redetect.run_probes(data, res, probes, [(ii[0], img(ii[0], True)), (ii[1], img(ii[1], False))],
                                  ColourBlobDetector(), rc)
    assert len(new1) == 0


def test_augment_disabled_is_identity(tmp_path, monkeypatch):
    monkeypatch.delenv("REPLAY_REDETECT", raising=False)
    data = make_data(5, [(1, ORANGE, moving_box(300))])
    assert redetect.augment(tmp_path, data, None, None, SimpleNamespace()) is data
    monkeypatch.setenv("REPLAY_REDETECT", "1")
    assert redetect.config_from(SimpleNamespace()).enabled
    # enabled but no video and no cache: unchanged
    assert redetect.augment(tmp_path, data, None, None, SimpleNamespace()) is data


def test_with_rows_sorted():
    data = make_data(5, [(1, ORANGE, moving_box(300))])
    data = SimpleNamespace(**vars(data))
    import dataclasses

    @dataclasses.dataclass
    class D:
        rows: np.ndarray

    d = D(data.rows)
    extra = np.array([[2, 1, 1, 5, 9, 0.4, 0, -1, *([0.0] * 6)]], float)
    out = redetect._with_rows(d, extra)
    assert len(out.rows) == 6 and (np.diff(out.rows[:, 0]) >= 0).all()


def test_slot_hints_from_previous_output(tmp_path):
    """A hidden player predicted by the previous output (interpolated rows) is probed at its court position."""
    n = 30
    orange = moving_box(300)
    hidden = set(range(10, 18))
    data = make_data(n, [(1, ORANGE, lambda i: None if i in hidden else orange(i))])
    data.frame = np.arange(n) * 3
    res = LinearH(n)
    out = tmp_path / "out"
    (out / "work").mkdir(parents=True)
    lines = ["frame,timestamp,track_id,object_type,team,x_m,y_m,confidence,interpolated,calibration,ball_state,"
             "ball_owner,img_x1,img_y1,img_x2,img_y2"]
    for i in range(n):
        b = orange(i)
        x, y = 0.02 * (b[0] + b[2]) / 2, 0.02 * b[3]
        box = "" if i in hidden else ",".join(f"{v:.1f}" for v in b)
        lines.append(f"{i * 3},{i / 10},7,player,A,{x:.3f},{y:.3f},0.9,{i in hidden},keyframe,,,"
                     + (box if box else ",,,"))
    (out / "tracking.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rc = redetect.RedetectConfig(enabled=True, clusters=False)
    hints = redetect.load_hints(out / "work", data, rc)
    assert len(hints) == len(hidden) and (out / "work" / "redetect_hints.json").exists()
    assert redetect.load_hints(out / "work", data, rc) == hints  # frozen copy
    probes = [p for p in redetect.hint_probes(data, res, rc, (W, H), hints) if p.kind == "slot"]
    assert probes and all(p.i in hidden for p in probes)
    p0 = min(probes, key=lambda p: p.priority)
    assert abs((p0.pred[0] + p0.pred[2]) / 2 - (orange(p0.i)[0] + 20)) < 3
    frames = [(i, frame_with([(orange(i), ORANGE)])) for i in range(n)]
    new, st = redetect.run_probes(data, res, probes, frames, ColourBlobDetector(), rc)
    assert st["slot_added"] == len(probes) == len(new)


def test_select_alternates_gap_and_slot_probes():
    mk = lambda kind, pr: redetect.Probe(0, (0, 0, 10, 10), kind, pr)  # noqa: E731
    probes = [mk("slot", 0.0) for _ in range(10)] + [mk("gap", 0.5) for _ in range(3)]
    out = redetect.select(probes, 6, 1.0)
    assert [p.kind for p in out] == ["gap", "slot", "gap", "slot", "gap", "slot"]


def test_fill_crumbs_keeps_only_gap_fillers(monkeypatch):
    """Task 9 retry: low-score rows survive only inside a gap of an existing identity; isolated ones are removed."""
    import dataclasses

    monkeypatch.delenv("REPLAY_CRUMBS", raising=False)
    n = 30
    orange = moving_box(300)
    data = make_data(n, [(1, ORANGE, lambda i: None if 10 <= i <= 14 else orange(i))])
    crumbs = [[i, *orange(i), 0.18, PERSON, -1, *([np.nan] * 6)] for i in range(10, 15)]  # where the player is
    crumbs.append([12, 900, 380, 940, 500, 0.2, PERSON, -1, *([np.nan] * 6)])  # isolated, far away
    crumbs.append([25, 600, 380, 640, 500, 0.2, PERSON, -1, *([np.nan] * 6)])  # isolated, no gap there
    data.rows = np.vstack([data.rows, np.array(crumbs, float)])
    data.signature = {**data.signature, "detector": {"threshold": 0.3, "store_threshold": 0.12}}

    @dataclasses.dataclass
    class D:
        k: np.ndarray
        frame: np.ndarray
        t: np.ndarray
        rows: np.ndarray
        motion: list
        signature: dict

    d = D(data.k, data.frame, data.t, data.rows, data.motion, data.signature)
    out = redetect.fill_crumbs(d, None, SimpleNamespace())
    low = out.rows[out.rows[:, 5] < 0.3]
    assert sorted(low[:, 0].astype(int).tolist()) == [10, 11, 12, 13, 14]
    assert (low[:, 7] == 1).all()  # joined to the gap's ByteTrack id
    monkeypatch.setenv("REPLAY_CRUMBS", "none")
    assert (redetect.fill_crumbs(d, None, SimpleNamespace()).rows[:, 5] >= 0.3).all()
    d.signature = {**data.signature, "detector": {"threshold": 0.3}}  # run without crumbs: unchanged
    assert redetect.fill_crumbs(d, None, SimpleNamespace()) is d
