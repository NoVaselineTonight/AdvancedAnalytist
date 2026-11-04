"""Clip playlists (replay/playlists.py + replay/gui/playlists.py): catalog, filtering, clip merging, storage,
export command building, a real (tiny) ffmpeg export, the .aareplay package and the GUI panel offscreen."""

import csv
import json
import os
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from replay import playlists as pl  # noqa: E402

TEAMS = {"A": {"name": "Team orange", "primary": "#d21c00"}, "B": {"name": "Team purple", "primary": "#b603d2"}}


def _replay():
    t = [round(k * 0.1, 2) for k in range(1200)]  # 120 s
    return {"game": {"teams": TEAMS}, "t": t, "fps": 10.0,
            "tracks": [{"id": 12, "team": "A", "i0": 0, "x": [100] * 600},  # 0 .. 59.9 s
                       {"id": 40, "team": "A", "i0": 600, "x": [100] * 600},  # 60 .. 119.9 s (same slot A1)
                       {"id": 7, "team": "B", "i0": 0, "x": [100] * 1200}],
            "track_info": {"12": {"team": "A", "number": "12"}, "40": {"team": "A"}, "7": {"team": "B",
                                                                                          "number": "5"}},
            "events": [{"id": "basket-300-L", "type": "basket", "t": 30.0, "team": "A", "confidence": 0.9,
                        "source": "auto", "details": {"shooter_track": 12}},
                       {"id": "basket-900-R", "type": "basket", "t": 90.0, "team": "B", "confidence": 0.4,
                        "source": "auto", "details": {}}]}


def _gs():
    ev = [
        {"id": "gs-shot-280", "type": "shot", "t": 28.0, "team": "A", "confidence": 0.8, "source": "game_state",
         "label": "Shot", "suffix": "", "details": {"shooter_track": "12", "distance_m": 7.1, "t_at_rim": 29.5,
                                                    "outcome": "made"}},
        {"id": "gs-shot-500", "type": "shot", "t": 50.0, "team": "B", "confidence": 0.7, "source": "game_state",
         "label": "Shot", "suffix": "", "details": {"shooter_track": "7", "distance_m": 3.0, "t_at_rim": 51.0}},
        {"id": "gs-turnover-100", "type": "turnover", "t": 10.0, "team": "A", "confidence": 0.6,
         "source": "game_state", "label": "Turnover", "suffix": "", "details": {}},
        {"id": "be-steal-700", "type": "steal", "t": 70.0, "team": "B", "confidence": 0.75, "source": "ball_events",
         "label": "Steal", "suffix": "", "details": {"stealer_track": "7", "lost_by_track": "40", "lost_by": "A"}},
        {"id": "be-lost_ball-699", "type": "lost_ball", "t": 69.8, "team": "A", "confidence": 0.7,
         "source": "ball_events", "label": "Lost ball", "suffix": "", "details": {"player_track": "40"}},
        {"id": "gs-fast_break-710", "type": "fast_break", "t": 71.0, "duration": 4.0, "team": "B", "confidence": 0.8,
         "source": "game_state", "label": "Fast break", "suffix": "", "details": {}},
        {"id": "gs-turnover-1100", "type": "turnover", "t": 110.0, "team": "A", "confidence": 0.2,
         "source": "game_state", "label": "Turnover", "suffix": "", "details": {}},
    ]
    return {"teams": {"A": "Team orange", "B": "Team purple"},
            "halves": [{"t0": 0.0, "t1": 60.0, "label": "1st half", "index": 0},
                       {"t0": 60.0, "t1": 120.0, "label": "2nd half", "index": 1}],
            "possessions": [{"id": "p0", "team": "A", "t0": 20.0, "t1": 30.0, "start": "start", "confidence": 0.9},
                            {"id": "p1", "team": "B", "t0": 40.0, "t1": 52.0, "start": "after shot",
                             "confidence": 0.8},
                            {"id": "p2", "team": "A", "t0": 62.0, "t1": 70.0, "start": "inbound", "confidence": 0.8}],
            "shots": [{"t_release": 50.0, "t_rim": 51.0, "team": "B", "made": False, "missed": True,
                       "distance_m": 3.0, "free_throw": False}],
            "free_throw_windows": [{"t0": 100.0, "t1": 106.0, "team": "B", "confidence": 0.7}],
            "lineup": {"slots": {"A1": [["12", 0.0, 59.9], ["40", 60.0, 119.9]], "B1": [["7", 0.0, 119.9]]},
                       "slot_of_track": {"12": "A1", "40": "A1", "7": "B1"}},
            "ball_events": {"states": [[21.0, 25.0, "held", "12"], [41.0, 45.0, "held", "7"],
                                       [63.0, 66.0, "held", "40"]]},
            "events": ev}


def _timeline():
    return {"format": "advanced-analytist/timeline@1",
            "tracks": [{"id": "auto1", "name": "Auto detection", "kind": "auto", "marks": []},
                       {"id": "notes", "name": "Notes", "kind": "user",
                        "marks": [{"id": "m1", "t": 15.0, "duration": 3.0, "title": "Bad spacing", "note": "x"}]},
                       {"id": "ai-analysis", "name": "AI analysis", "kind": "user",
                        "marks": [{"id": "ai1", "t": 80.0, "duration": 5.0, "title": "Late help"}]}],
            "auto_overrides": {"basket-900-R": {"dismissed": True}}}


@pytest.fixture()
def cat():
    return pl.collect(None, replay=_replay(), gs=_gs(), timeline=_timeline(), light=[], duration=120.0)


def _ids(items):
    return [it.id for it in items]


# ====================================================================================== catalog / filter
def test_catalog_categories(cat):
    by = {it.id: it for it in cat.items}
    assert "basket-900-R" not in by  # dismissed on the timeline by the coach
    assert {k: v for k, v in by["basket-300-L"].cats.items() if not k.startswith("track:")} ==         {"made": "A", "shot": "A"}
    assert by["gs-shot-280"].cats["made"] == "A" and by["gs-shot-280"].distance_m == 7.1
    assert by["gs-shot-280"].t_end == 29.5  # the clip runs until the ball reaches the rim
    assert by["gs-shot-500"].cats.get("missed") == "B"  # outcome from the game-state shot list
    st = by["be-steal-700"]
    assert st.cats["steal"] == "B" and st.cats["turnover"] == "A"  # a steal is the other team's turnover
    assert by["poss-p0"].players == ["12"]  # held the ball during the possession
    assert by["ftw-0"].cats == {"free_throw": "B"}  # windows used when there are no free-throw events
    assert by["mark-m1"].cats == {"track:notes": None, "note": None}
    assert by["mark-ai1"].cats == {"track:ai-analysis": None, "ai": None}
    assert "track:auto1" in by["gs-shot-280"].cats  # the Auto detection track = detected events ...
    assert "track:auto1" not in by["gs-turnover-1100"].cats  # ... above the timeline's confidence cut
    assert "track:auto1" not in by["poss-p0"].cats
    assert [k for k, _ in cat.player_choices()][:2] == ["num:A:12", "num:B:5"]


def test_filter_team_category_confidence(cat):
    ours = pl.apply_filter(cat, pl.Filter(["turnover"], team="A"))
    assert _ids(ours) == ["gs-turnover-100", "be-lost_ball-699", "be-steal-700", "gs-turnover-1100"]
    assert _ids(pl.apply_filter(cat, pl.Filter(["turnover"], team="A", min_confidence=0.5))) == \
        ["gs-turnover-100", "be-lost_ball-699", "be-steal-700"]
    assert _ids(pl.apply_filter(cat, pl.Filter(["fast_break"], team="B"))) == ["gs-fast_break-710"]
    assert _ids(pl.apply_filter(cat, pl.Filter(["made"], team="A"))) == ["gs-shot-280", "basket-300-L"]
    assert _ids(pl.apply_filter(cat, pl.Filter(["free_throw"]))) == ["ftw-0"]
    # teamless categories (my marks, AI issues) are not removed by a team filter
    assert _ids(pl.apply_filter(cat, pl.Filter(["note", "ai"], team="B", min_confidence=0.9))) == \
        ["mark-m1", "mark-ai1"]
    assert _ids(pl.apply_filter(cat, pl.Filter(["track:notes"]))) == ["mark-m1"]


def test_filter_distance_period_time_text(cat):
    assert _ids(pl.apply_filter(cat, pl.Filter(["shot"], min_distance_m=6.0))) == ["gs-shot-280"]
    assert _ids(pl.apply_filter(cat, pl.Filter(["shot"], max_distance_m=4.0))) == ["gs-shot-500"]
    assert _ids(pl.apply_filter(cat, pl.Filter(["possession"], period=1))) == ["poss-p2"]
    assert _ids(pl.apply_filter(cat, pl.Filter(["possession"], t_from=35.0, t_to=65.0))) == ["poss-p1", "poss-p2"]
    assert _ids(pl.apply_filter(cat, pl.Filter(["note"], text="SPACING"))) == ["mark-m1"]


def test_filter_players(cat):
    # "every possession of #12": had the ball in it
    f = pl.Filter(["possession"], players=["num:12"])
    assert _ids(pl.apply_filter(cat, f)) == ["poss-p0"]
    # the on-court slot follows the player across tracker ids 12 -> 40
    assert _ids(pl.apply_filter(cat, pl.Filter(["possession"], players=["slot:A1"]))) == ["poss-p0", "poss-p2"]
    f = pl.Filter(["possession"], players=["num:A:12"], player_mode="on_court")
    assert _ids(pl.apply_filter(cat, f)) == ["poss-p0", "poss-p1"]  # track 12 exists until 59.9 s
    assert _ids(pl.apply_filter(cat, pl.Filter(["turnover"], players=["trk:40"]))) == \
        ["be-lost_ball-699", "be-steal-700"]
    assert pl.parse_player("#12") == "num:12" and pl.parse_player("a3") == "slot:A3"
    assert pl.parse_player("T42") == "trk:42" and pl.parse_player("x") is None


# ====================================================================================== clips
def _item(i, t, end=None):
    return pl.Item(id=f"e{i}", t=t, t_end=t if end is None else end, title=f"Event {i}", source="t", cats={"x": "A"})


def test_overlapping_clips_are_merged():
    items = [_item(1, 10.0), _item(2, 13.0), _item(3, 30.0), _item(4, 31.0, 40.0), _item(5, 50.0)]
    clips = pl.build_clips(items, pre_s=4.0, post_s=2.0)
    assert [(c.t0, c.t1) for c in clips] == [(6.0, 15.0), (26.0, 42.0), (46.0, 52.0)]
    assert clips[0].item_ids == ["e1", "e2"] and clips[0].title == "Event 1  (+1 more)"
    # touching / close clips are joined with a merge gap
    assert len(pl.build_clips([_item(1, 10.0), _item(2, 17.5)], 1.0, 1.0)) == 2
    assert len(pl.build_clips([_item(1, 10.0), _item(2, 17.5)], 1.0, 1.0, merge_gap_s=6.0)) == 1
    # clamped to the video
    c = pl.build_clips([_item(1, 1.0), _item(2, 99.0)], 4.0, 4.0, duration=100.0)
    assert (c[0].t0, c[-1].t1) == (0.0, 100.0)
    # a duplicate (made shot + detected basket of the same play) is shown once
    assert len(pl.build_clips([_item(1, 28.0, 29.5), _item(2, 30.0)], 4.0, 2.0)) == 1


def test_playlist_rebuild_keeps_review_state_and_storage(cat, tmp_path):
    p = pl.make_playlist(cat, "Our turnovers", pl.Filter(["turnover"], team="A"), 5.0, 2.0)
    assert len(p.clips) == 3  # 10 s, 69.8+70 merged, 110 s
    p.clips[1].reviewed, p.clips[1].note = True, "Ball watching"
    p.filter.min_confidence = 0.5
    before, after = p.rebuild(cat)
    assert (before, after) == (3, 2)
    assert p.clips[1].reviewed and p.clips[1].note == "Ball watching"
    book = pl.PlaylistBook([p])
    book.save(tmp_path)
    again = pl.PlaylistBook.load(tmp_path)
    assert again.to_dict() == book.to_dict()
    (tmp_path / pl.FILENAME).write_text("{broken", encoding="utf-8")
    assert pl.PlaylistBook.load(tmp_path).playlists == []
    assert (tmp_path / (pl.FILENAME + ".bad")).exists()


def test_fixed_lists_and_stats_hand_off(cat, tmp_path):
    fx = pl.fixed_playlist("Stints", [{"t": 10.0, "t1": 20.0, "title": "Stint 1", "team": "A"},
                                      {"t": 21.0, "title": "Stint 2"}, {"t": 60.0, "title": "Stint 3"}],
                           pre_s=2.0, post_s=1.0, origin="stats")
    assert fx.fixed and [(c.t0, c.t1) for c in fx.clips] == [(8.0, 22.0), (58.0, 61.0)]
    assert fx.rebuild(cat) == (2, 2) and len(fx.clips) == 2  # never emptied by "Rebuild"
    book = pl.PlaylistBook([fx])
    book.save(tmp_path)
    assert pl.PlaylistBook.load(tmp_path).playlists[0].fixed
    # the Stats tab's hand-off (replay/gui/stats_tab.py): live filter, or a fixed list marked in the filter text
    from replay.gui import stats_tab
    assert stats_tab.playlist_module() is pl
    live = stats_tab.make_stats_playlist(pl, "Stats: turnovers", [], {"categories": ["turnover"], "team": "A",
                                                                      "t_from": None, "t_to": None,
                                                                      "min_confidence": 0.5}, lambda: cat)
    assert not live.fixed and len(live.clips) == 2
    old = stats_tab.make_stats_playlist(pl, "Stats: zone", [{"t": 5.0, "title": "Shot"}], None, lambda: cat)
    assert old.fixed and old.filter.text == "" and len(old.clips) == 1
    assert pl.Playlist.from_dict(old.to_dict()).fixed


def test_presets_use_coach_wording(cat):
    names = [n for n, *_ in pl.presets("B", cat.names)]
    assert "All our turnovers" in names and "All their fast breaks" in names
    assert "All made baskets by Team purple" in names and "All shots from > 6 m" in names
    f = dict((n, f) for n, f, *_ in pl.presets("B", cat.names))["All our turnovers"]
    assert f.team == "B"
    assert pl.team_label("B", "B", cat.names) == "Our team (Team purple)"
    assert pl.team_label("A", "B", cat.names, short=True) == "Opponent"
    assert pl.team_label("A", None, cat.names) == "Team orange"
    assert "Team orange turnovers" in " ".join(n for n, *_ in pl.presets(None, cat.names))


# ====================================================================================== export
def test_export_command_building(tmp_path):
    ff = "ffmpeg"
    seg = pl.segment_cmd(ff, "v.mp4", 12.5, 20.0, "o.mp4", fps=29.97, size=(1280, 720), audio=False)
    assert seg[seg.index("-ss") + 1] == "12.500" and seg.index("-ss") < seg.index("-i")  # fast, accurate seek
    assert seg[seg.index("-t") + 1] == "7.500" and "-an" in seg
    assert "fps=30000/1001" in seg[seg.index("-vf") + 1]
    card = pl.card_cmd(ff, "c.png", 2.0, "c.mp4", fps=29.97, size=(1280, 720), audio=True)
    assert "anullsrc=channel_layout=stereo:sample_rate=48000" in card
    # identical encoder settings for clips and cards: the concat can copy the streams
    tail = lambda c: c[c.index("-vf"):c.index("-video_track_timescale") + 2]  # noqa: E731
    assert tail(seg) == tail(card)
    seg_a = pl.segment_cmd(ff, "v.mp4", 0, 1, "o.mp4", fps=30, size=(1281, 721), audio=True)
    assert "-c:a" in seg_a and "scale=1280:720" in seg_a[seg_a.index("-vf") + 1]
    lst = pl.concat_list([tmp_path / "a b.mp4", tmp_path / "it's.mp4"])
    assert lst.splitlines()[0].startswith("file '") and "it'\\''s.mp4'" in lst
    cc = pl.concat_cmd(ff, "l.txt", "out.mp4")
    assert cc[cc.index("-f") + 1] == "concat" and cc[cc.index("-c") + 1] == "copy"
    assert pl.fps_arg(25) == "25" and pl.fps_arg(29.97) == "30000/1001"


def test_export_plan_and_csv(cat, tmp_path):
    p = pl.make_playlist(cat, "Our turnovers", pl.Filter(["turnover"], team="A", min_confidence=0.5), 5.0, 2.0)
    p.clips[0].reviewed, p.clips[0].note = True, "Kääntyi\nliian aikaisin"
    parts = pl.export_plan(p, pl.ExportOptions(title_cards=True), cat.names, "A")
    assert [x["kind"] for x in parts] == ["card", "card", "clip", "card", "clip"]
    assert "Our team (Team orange)" in parts[1]["lines"]
    parts = pl.export_plan(p, pl.ExportOptions(title_cards=False, only_unreviewed=True), cat.names, "A")
    assert [x["kind"] for x in parts] == ["clip"]
    out = pl.export_csv(p, tmp_path / "t.csv", cat.names, "A")
    rows = list(csv.reader(open(out, encoding="utf-8-sig")))
    assert rows[0][:4] == ["clip", "start_s", "end_s", "duration_s"]
    assert rows[1][1] == "5.00" and rows[1][4] == "0:05.0" and rows[1][9] == "yes"
    assert rows[1][10] == "Kääntyi liian aikaisin" and rows[2][8] == "Our team (Team orange)"


def test_batch_command_one_ffmpeg_run(tmp_path):
    parts = [{"kind": "card", "title": "x", "lines": [], "seconds": 1.5, "image": "c.png"},
             {"kind": "clip", "t0": 10.0, "t1": 14.5}, {"kind": "clip", "t0": 30.0, "t1": 31.0}]
    cmd = pl.batch_cmd("ffmpeg", "v.mp4", parts, "o.mp4", fps=29.97, size=(1280, 720), audio=True)
    assert cmd.count("-i") == 4  # card image + its silence + 2 clips of the same video
    i = cmd.index("10.000")
    assert cmd[i - 1] == "-ss" and cmd[i + 1:i + 3] == ["-t", "4.500"]  # input seek per clip
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "concat=n=3:v=1:a=1[v][a]" in fc and "[1:a:0]" in fc and "[3:a:0]" in fc
    seg = pl.segment_cmd("ffmpeg", "v.mp4", 0, 1, "o.mp4", fps=29.97, size=(1280, 720), audio=False)
    enc = seg[seg.index("-c:v"):seg.index("-video_track_timescale") + 2]
    assert cmd[cmd.index("-c:v"):cmd.index("-video_track_timescale") + 2] == enc  # batches can be concat-copied
    assert "-an" in pl.batch_cmd("ffmpeg", "v.mp4", parts, "o.mp4", fps=25, size=(320, 180), audio=False)


def test_export_mp4_real_ffmpeg(cat, tmp_path, monkeypatch):
    import subprocess
    import threading

    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    src = tmp_path / "src.mp4"
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=320x180:rate=25:duration=12", "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)], check=True)
    p = pl.Playlist("Test", clips=[pl.Clip(1.0, 3.0, "Turnover – Team orange"), pl.Clip(6.0, 9.0, "Steal")])
    seen = []
    out = pl.export_mp4(p, src, tmp_path / "out.mp4", fps=25, size=(320, 180), names=cat.names, coached="A",
                        has_audio=True, opts=pl.ExportOptions(title_cards=True, card_s=1.0), progress=seen.append)
    frames, secs = imageio_ffmpeg.count_frames_and_secs(str(out))
    assert abs(secs - (2.0 + 3.0 + 3 * 1.0)) < 0.3  # 2 clips + intro card + 2 clip cards
    assert seen[-1] == 1.0 and not list(tmp_path.glob("out.part*"))
    # several ffmpeg batches joined without re-encoding, no audio, no title cards
    monkeypatch.setattr(pl, "BATCH_PARTS", 1)
    out2 = pl.export_mp4(p, src, tmp_path / "out2.mp4", fps=25, size=(320, 180),
                         opts=pl.ExportOptions(title_cards=False))
    assert abs(imageio_ffmpeg.count_frames_and_secs(str(out2))[1] - 5.0) < 0.3
    ev = threading.Event()
    ev.set()
    with pytest.raises(pl.ExportCancelled):
        pl.export_mp4(p, src, tmp_path / "c.mp4", fps=25, size=(320, 180), cancel=ev)
    assert not (tmp_path / "c.mp4").exists()


# ====================================================================================== package
def test_playlists_travel_in_the_replay_package(tmp_path):
    from fake_project import make_project

    from replay.gui.projects import ProjectStore
    store = ProjectStore(tmp_path / "projects")
    root = make_project(store.root / "kouvot-honka-abc123")
    cat = pl.collect(root)
    assert {it.id for it in cat.items} >= {"basket-30-L", "basket-60-R", "mark-m1"}
    assert "basket-99-R" not in {it.id for it in cat.items}  # dismissed
    assert next(it for it in cat.items if it.id == "basket-60-R").title == "Honka layup"  # coach's own title
    book = pl.PlaylistBook([pl.make_playlist(cat, "Baskets", pl.Filter(["made"]), 2.0, 1.0)])
    book.save(root)
    dst = tmp_path / "x.aareplay"
    store.export("kouvot-honka-abc123", dst)
    with zipfile.ZipFile(dst) as z:
        assert pl.FILENAME in z.namelist()
    assert "Clip playlists" in store.export_plan("kouvot-honka-abc123").groups()  # shown in the export dialog
    p2 = store.import_package(dst)
    assert pl.PlaylistBook.load(p2.root).to_dict() == book.to_dict()


# ====================================================================================== GUI (offscreen)
@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_gui_panel_offscreen(tmp_path, app, monkeypatch):
    from fake_project import make_project
    from PySide6.QtCore import QSettings

    from replay.gui import app as A
    from replay.gui import playlists as G
    monkeypatch.setattr(A, "QSettings", lambda *a: QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    projects = tmp_path / "projects"
    projects.mkdir()
    make_project(projects / "kouvot-honka-abc123")
    win = A.MainWindow(projects)
    monkeypatch.setattr(win.jobs, "submit", lambda *a, **kw: None)  # no rebuild of the fake Replay
    try:
        panel = win.playlists_panel
        assert any(a.text() == "&Playlists" for a in win.menuBar().actions())
        p = win.store.get("kouvot-honka-abc123")
        win.show_project(p)
        panel._watch_project()
        assert panel.p is not None and panel.p.root == p.root
        cat = panel.catalog()
        dlg = G.FilterDialog(panel, cat, None)
        i = next(k for k, (n, *_r) in enumerate(dlg._presets) if n.startswith("All made baskets by Kouvot"))
        dlg.preset.setCurrentIndex(i + 1)
        dlg.we.setCurrentIndex(dlg.we.findData("A"))  # we coach Kouvot
        dlg._update_preview()
        assert "clips" in dlg.preview.text()
        new = panel._create_from_dialog(dlg)
        assert G.pl.coached_team(p.settings()) == "A"
        # 5 s pre-roll: the baskets at 1.0 s and 3.0 s overlap -> one clip (the 4-s video ends the clip)
        assert [(c.item_ids, c.t0, c.t1) for c in new.clips] == [(["basket-30-L", "basket-90-L"], 0.0, 4.0)]
        new.pre_s = new.post_s = 0.5
        panel.rebuild()
        assert [c.item_ids for c in new.clips] == [["basket-30-L"], ["basket-90-L"]]
        assert panel.tree.topLevelItemCount() == 2 and panel.tree.topLevelItem(0).text(G.COL_TEAM) == "Our team"
        assert panel.play(1, play_all=True) and panel.playing == 1
        assert win.tabs.currentWidget() is win.replay_tab
        panel.stop()
        panel.set_reviewed(0, True)
        panel.set_note(1, "Look at the screen")
        saved = pl.PlaylistBook.load(p.root).playlists[0]
        assert saved.clips[0].reviewed and saved.clips[1].note == "Look at the screen"
        out = panel.export_csv(str(tmp_path / "pl.csv"))
        assert out.exists() and "Look at the screen" in out.read_text(encoding="utf-8-sig")
        fx = panel.add_playlist(pl.fixed_playlist("Fixed", [{"t": 2.0, "title": "x"}], origin="stats"))
        assert panel.current() is fx and "Fixed clip list from the Stats tab" in panel.summary.text()
        panel.rebuild()  # no-op for a fixed list
        assert len(fx.clips) == 1
        panel.combo.setCurrentIndex(0)
        panel.rebuild()
        assert pl.PlaylistBook.load(p.root).playlists[0].clips[1].note == "Look at the screen"
        assert pl.PlaylistBook.load(p.root).playlists[0].clips[0].reviewed
    finally:
        win.timer.stop()
        win.close()
