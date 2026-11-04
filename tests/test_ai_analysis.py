"""AI tactical analysis: prompt building, response parsing, chunking/merging, errors, timeline track. The Gemini
API is always mocked - no network calls."""

import io
import json
import threading
import urllib.error

import pytest
from fake_project import make_project

from replay import ai_analysis as ai
from replay.gui import timeline_model as tm

KEY = "AIzaTESTKEY_0123456789abcdefghijklmnop"


def _good(issues=None, summary="Hyvä peli."):
    return json.dumps({"overview": {"went_well": ["Spacing"], "to_improve": ["Getting back"], "summary": summary},
                       "issues": issues if issues is not None else [
                           {"t_start": 1.0, "t_end": 2.5, "title": "Late transition D", "description": "d",
                            "category": "transition defense", "team_phase": "transition"}]})


class FakeClient:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def generate(self, model, system, user, schema):
        self.calls.append({"model": model, "system": system, "user": user, "schema": schema})
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a(user) if callable(a) else a


# ---------------------------------------------------------------------- digest / prompts
def test_digest_reads_court_size_teams_and_samples(tmp_path):
    root = make_project(tmp_path / "p")
    rd = json.loads((root / "results" / "replay_data.json").read_text(encoding="utf-8"))
    rd["court"] = {"length": 26, "width": 14, "lines": [], "baskets": [[1.575, 7.0], [24.425, 7.0]]}
    (root / "results" / "replay_data.json").write_text(json.dumps(rd), encoding="utf-8")
    tl = json.loads((root / "timeline.json").read_text(encoding="utf-8"))
    tl["tracks"].append({"id": ai.AI_TRACK_ID, "name": ai.AI_TRACK_NAME, "kind": "user",
                         "marks": [{"t": 1, "title": "OLD AI MARK"}]})
    (root / "timeline.json").write_text(json.dumps(tl), encoding="utf-8")

    d = ai.build_game_digest(root, {"A": "#123456"}, sample_hz=2.0)
    assert d.court_size == (26.0, 14.0)
    assert "26 m x 14 m" in d.header and "half-court line x = 13" in d.header and "R = (24.425, 7)" in d.header
    assert "Team A = Kouvot (orange jerseys, home; playback colour #123456)" in d.header
    assert "Team B = Tapiolan Honka (blue jerseys, away; playback colour #1144aa)" in d.header
    assert "[title withheld]" in d.header and "Fast break" not in d.header and "OLD AI MARK" not in d.header  # privacy: coach free text withheld; previous AI marks no
    assert "basket L by Kouvot" in d.header and "basket-99-R" not in d.header  # dismissed event left out
    # 10 fps over 4 s -> 2 Hz = 8 samples
    assert [round(t, 1) for t, _ in d.frames] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    first = d.frames[0][1]
    assert first.startswith("0.0|") and "A 1:5.0,7.0|B 2:20.0,3.0|R 3:14.0,1.0|ball 5.0,7.0 held A1|U1" in first
    assert any("1:6.5,7.0*" in ln for _, ln in d.frames)  # k=5 interpolated
    assert d.frames[-1][1].endswith("loose")  # ball estimated after k=25, no unknown person after k=20
    assert d.t_min == 0.0 and d.t_max == 3.9  # full analysed range


def test_system_prompt_has_team_level_language_and_scope():
    teams = {"A": {"name": "Kouvot", "color": "#ff8800"}, "B": {"name": "Honka", "color": "#1144aa"}}
    s = ai.build_system_prompt(teams, "B", "WU13 1D, tytöt")
    assert "team B = Honka (colour #1144aa)" in s and "team A = Kouvot" in s
    assert '"""WU13 1D, tytöt"""' in s and "Calibrate" in s and "NAISTEN 1D" in s
    assert "individual technique" in s and "language" in s.lower()
    for k in ("went_well", "to_improve", "summary", "t_start", "t_end", "team_phase", "category"):
        assert k in s
    assert "(not given)" in ai.build_system_prompt(teams, "A", "  ")


def test_chunks_split_and_merge_short_tail():
    frames = [(t * 0.5, f"{t * 0.5:.1f}|x") for t in range(0, 2 * 130)]  # 0..129.5 s
    d = ai.Digest("h", frames, 0.0, 129.5, {}, (28, 15), {})
    ch = d.chunks(60.0)
    assert [(a, b) for a, b, _ in ch] == [(0.0, 59.5), (60.0, 129.5)]  # 120..129.5 tail joined
    assert sum(len(ln) for _, _, ln in ch) == len(frames)
    u = ai.build_user_prompt(d, 0.0, 59.5, ch[0][2], 1, 2)
    assert "Part 1 of 2" in u and u.startswith("h")


# ---------------------------------------------------------------------- parsing
def test_parse_valid_fenced_and_normalised():
    text = "```json\n" + _good([
        {"t_start": 9, "t_end": 5, "title": "Swapped", "description": "x", "category": "", "team_phase": "Defence"},
        {"t_start": -3, "t_end": 1, "title": "Clamped", "description": "y", "category": "spacing",
         "team_phase": "offence"},
    ]) + "\n```"
    w = []
    r = ai.parse_response(text, 0.0, 20.0, w)
    assert r["overview"] == {"went_well": ["Spacing"], "to_improve": ["Getting back"], "summary": "Hyvä peli."}
    a, b = r["issues"]
    assert (a["t_start"], a["t_end"], a["team_phase"], a["title"]) == (0.0, 1.0, "offense", "Clamped")
    assert (b["t_start"], b["t_end"], b["team_phase"], b["category"]) == (5.0, 9.0, "defense", "other")
    assert w == []


def test_parse_drops_out_of_range_and_malformed():
    text = _good([
        {"t_start": 500, "t_end": 510, "title": "Far away", "description": "", "category": "x", "team_phase": "offense"},
        {"t_start": "soon", "title": "No time"},
        "not an object",
        {"t_start": 3, "title": "Only start", "team_phase": "weird"},
    ])
    w = []
    r = ai.parse_response(text, 0.0, 20.0, w)
    assert [(i["title"], i["t_end"], i["team_phase"]) for i in r["issues"]] == [("Only start", 3.0, "other")]
    assert any("outside the analysed range" in x for x in w) and any("2 malformed" in x for x in w)


@pytest.mark.parametrize("bad", ["", "not json at all", '{"overview": {"went_well": [', "[1, 2]", '{"foo": 1}'])
def test_parse_malformed_raises(bad):
    with pytest.raises(ai.AIError):
        ai.parse_response(bad, 0, 10)


def test_parse_json_with_surrounding_text():
    r = ai.parse_response("Here you go:\n" + _good() + "\nThanks", 0, 10)
    assert r["issues"][0]["title"] == "Late transition D"


# ---------------------------------------------------------------------- run (mocked API)
def test_run_analysis_single_request(tmp_path):
    root = make_project(tmp_path / "p")
    fc = FakeClient([_good()])
    seen = []
    res = ai.run_analysis(root, ai.Options(coached_team="A", level_context="NAISTEN 1D"), fc,
                          lambda f, m: seen.append(f))
    assert len(fc.calls) == 1
    c = fc.calls[0]
    assert c["model"] == ai.DEFAULT_MODEL and c["schema"] is ai.RESPONSE_SCHEMA
    assert "NAISTEN 1D" in c["system"] and "team A = Kouvot" in c["system"]
    assert "## TRACKING SAMPLES" in c["user"] and "A 1:5.0,7.0" in c["user"]
    assert res["format"] == ai.FORMAT and res["coached_team"]["name"] == "Kouvot"
    assert res["issues"][0]["title"] == "Late transition D" and res["overview"]["summary"] == "Hyvä peli."
    assert seen[-1] == 1.0
    assert KEY not in json.dumps(res)


def test_run_analysis_retries_malformed_answer_once(tmp_path):
    root = make_project(tmp_path / "p")
    fc = FakeClient(["{oops", _good()])
    res = ai.run_analysis(root, ai.Options(), fc)
    assert len(fc.calls) == 2 and len(res["issues"]) == 1
    fc = FakeClient(["{oops", "still bad"])
    with pytest.raises(ai.AIError):
        ai.run_analysis(root, ai.Options(), fc)


def _long_digest(monkeypatch, seconds=1500):
    frames = [(t * 0.5, f"{t * 0.5:.1f}|A 1:1.0,1.0") for t in range(int(seconds * 2))]
    d = ai.Digest("HEADER", frames, 0.0, frames[-1][0], {"A": {"name": "Kouvot", "color": "#f80"},
                                                           "B": {"name": "Honka", "color": "#14a"}}, (26, 14), {})
    monkeypatch.setattr(ai, "build_game_digest", lambda *a, **k: d)
    monkeypatch.setattr(ai.ai_light, "run_light", lambda *a, **k: ai.ai_light.LightResult(
        None, ai.ai_light.CleanReport(), [], False, [], [], {}))
    return d


def test_run_analysis_chunks_and_merges(monkeypatch, tmp_path):
    _long_digest(monkeypatch)

    def answer(user):
        t0 = float(user.split("video time ")[1].split("-")[0])
        return _good([{"t_start": t0 + 10, "t_end": t0 + 20, "title": f"at {t0:.0f}", "description": "",
                       "category": "spacing", "team_phase": "offense"},
                      {"t_start": t0 - 400, "t_end": t0 - 390, "title": "outside part", "description": "",
                       "category": "x", "team_phase": "offense"}])
    merged = json.dumps({"overview": {"went_well": ["W"], "to_improve": ["I"], "summary": "Whole game"}})
    fc = FakeClient([answer, answer, answer, merged])
    res = ai.run_analysis(tmp_path, ai.Options(chunk_s=600), fc)
    assert len(fc.calls) == 4 and fc.calls[-1]["schema"] is ai.MERGE_SCHEMA
    assert "Part 1 of 3" in fc.calls[0]["user"] and "Part 3 of 3" in fc.calls[2]["user"]
    assert [i["title"] for i in res["issues"]] == ["at 0", "at 600", "at 1200"]
    assert res["overview"]["summary"] == "Whole game" and len(res["parts"]) == 3


def test_merge_failure_falls_back_to_local_merge(monkeypatch, tmp_path):
    _long_digest(monkeypatch, 1300)
    fc = FakeClient([_good([], "one"), _good([], "two"), ai.AIError("Quota or rate limit reached (429).")])
    res = ai.run_analysis(tmp_path, ai.Options(chunk_s=600), fc)
    assert res["overview"]["summary"] == "one\n\ntwo" and res["overview"]["went_well"] == ["Spacing"]
    assert any("merged without the AI" in w for w in res["warnings"])


def test_transient_error_retried_and_cancel(monkeypatch, tmp_path):
    root = make_project(tmp_path / "p")
    monkeypatch.setattr(ai.time, "sleep", lambda s: None)
    fc = FakeClient([ai.AIError("The Gemini service had a problem (503)."), _good()])
    assert len(ai.run_analysis(root, ai.Options(), fc)["issues"]) == 1
    fc = FakeClient([ai.AIError("The API key was rejected (invalid key).")])
    with pytest.raises(ai.AIError, match="invalid key"):
        ai.run_analysis(root, ai.Options(), fc)
    ev = threading.Event()
    ev.set()
    with pytest.raises(ai.Cancelled):
        ai.run_analysis(root, ai.Options(), FakeClient([_good()]), cancel=ev)
    with pytest.raises(ai.AIError):
        ai.run_analysis(root, ai.Options(coached_team="C"), FakeClient([]))


# ---------------------------------------------------------------------- errors / REST
def test_friendly_errors_never_contain_the_key():
    from google.genai import errors
    e = errors.ClientError(400, {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.",
                                           "status": "INVALID_ARGUMENT"}})
    assert "invalid key" in ai._friendly_error(e, KEY, "m")
    e = errors.ClientError(429, {"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}})
    assert "Quota" in ai._friendly_error(e, KEY, "m")
    e = errors.ClientError(404, {"error": {"code": 404, "message": "models/x is not found", "status": "NOT_FOUND"}})
    assert "'gemini-x' was not found" in ai._friendly_error(e, KEY, "gemini-x")
    e = errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})
    assert "(503)" in ai._friendly_error(e, KEY, "m")
    msg = ai._friendly_error(urllib.error.URLError(f"cannot reach ?key={KEY}"), KEY, "m")
    assert msg.startswith("Network error") and KEY not in msg
    assert KEY not in ai._friendly_error(RuntimeError(f"boom {KEY}"), KEY, "m")


def test_rest_client_request_and_errors(monkeypatch):
    sent = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        sent.update(url=req.full_url, headers=dict(req.header_items()), body=json.loads(req.data))
        return Resp(json.dumps({"candidates": [{"content": {"parts": [{"text": _good()}]},
                                                "finishReason": "STOP"}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    c = ai.GeminiClient(KEY, use_sdk=False)
    assert KEY not in repr(c)
    out = c.generate("gemini-3.8-flash", "SYS", "USER", ai.RESPONSE_SCHEMA)
    assert json.loads(out)["issues"]
    assert sent["url"] == f"{ai.API_BASE}/models/gemini-3.8-flash:generateContent" and KEY not in sent["url"]
    assert sent["headers"]["X-goog-api-key"] == KEY
    gc = sent["body"]["generationConfig"]
    assert gc["responseMimeType"] == "application/json" and gc["responseSchema"] == ai.RESPONSE_SCHEMA
    assert sent["body"]["systemInstruction"]["parts"][0]["text"] == "SYS"

    def http_error(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad", {}, io.BytesIO(json.dumps(
            {"error": {"code": 400, "message": "API key not valid.", "status": "INVALID_ARGUMENT"}}).encode()))
    monkeypatch.setattr("urllib.request.urlopen", http_error)
    with pytest.raises(ai.AIError, match="invalid key"):
        c.generate("m", "s", "u", {})
    with pytest.raises(ai.AIError):
        ai.GeminiClient("  ")


def test_response_text_blocked_and_truncated():
    with pytest.raises(ai.AIError, match="blocked"):
        ai.response_text({"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(ai.AIError, match="cut off"):
        ai.response_text({"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]})
    assert ai.response_text({"candidates": [{"content": {"parts": [{"text": "a", "thought": True},
                                                                   {"text": "{}"}]}}]}) == "{}"


# ---------------------------------------------------------------------- storage / timeline / package
def test_save_load_results_and_timeline_track(tmp_path):
    root = make_project(tmp_path / "p")
    res = ai.run_analysis(root, ai.Options(), FakeClient([_good([
        {"t_start": 1, "t_end": 2, "title": "One", "description": "D1", "category": "spacing", "team_phase": "offense"},
        {"t_start": 3, "t_end": 3.5, "title": "Two", "description": "D2", "category": "rebounding",
         "team_phase": "defense"}])]))
    res["api_key"] = KEY  # must never be persisted even if something put it there
    path = ai.save_results(root, res)
    assert path == root / "results" / "ai_analysis.json"
    assert KEY not in path.read_text(encoding="utf-8")
    loaded = ai.load_results(root)
    assert [i["title"] for i in loaded["issues"]] == ["One", "Two"]

    tl = tm.load(root)
    n_tracks = len(tl.tracks)
    tr = ai.apply_to_timeline(tl, loaded)
    assert tr.id == ai.AI_TRACK_ID and tr.name == ai.AI_TRACK_NAME and not tr.is_auto
    assert len(tl.tracks) == n_tracks + 1
    m1, m2 = tr.marks
    assert (m1.t, m1.duration, m1.title, m1.color) == (1.0, 1.0, "One", ai.PHASE_COLORS["offense"])
    assert "D2" in m2.note and "rebounding" in m2.note and m2.color == ai.PHASE_COLORS["defense"]
    tm.save(tl, root)

    # re-run replaces the AI marks, keeps the user's tracks and does not add a second AI track
    tl = tm.load(root)
    notes = next(t for t in tl.tracks if t.name == "Notes")
    assert [m.title for m in notes.marks] == ["Fast break"]
    tr = ai.apply_to_timeline(tl, {"issues": [{"t_start": 2, "t_end": 1, "title": "New"}]})
    assert [m.title for m in tr.marks] == ["New"] and tr.marks[0].duration == 0.0
    assert sum(t.id == ai.AI_TRACK_ID for t in tl.tracks) == 1 and len(tl.tracks) == n_tracks + 1


def test_package_export_includes_results_but_no_key(tmp_path):
    from replay.gui.projects import ProjectStore
    store = ProjectStore(tmp_path / "projects")
    root = make_project(store.root / "kouvot-honka-abc123")
    ai.save_results(root, ai.run_analysis(root, ai.Options(), FakeClient([_good()])))
    plan = store.export_plan("kouvot-honka-abc123")
    rels = {rel for _, rel in plan.files}
    assert "results/ai_analysis.json" in rels
    for f, _ in plan.files:
        if f.suffix in (".json", ".md", ".csv", ".html"):
            assert KEY not in f.read_text(encoding="utf-8", errors="ignore")


# ---------------------------------------------------------------------- GUI helpers (no window shown)
def test_key_storage_is_not_plaintext_and_roundtrips(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from replay.gui import ai_dialog
    qs = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    ai_dialog.save_api_key(qs, KEY)
    qs.sync()
    assert KEY not in (tmp_path / "s.ini").read_text(encoding="utf-8", errors="ignore")
    assert ai_dialog.load_api_key(qs) == KEY
    ai_dialog.save_api_key(qs, "")
    assert ai_dialog.load_api_key(qs) == ""


def test_results_html_escapes_and_links_times():
    pytest.importorskip("PySide6")
    from replay.gui import ai_dialog
    h = ai_dialog.results_html({"model": "m", "coached_team": {"name": "<b>X</b>", "color": "#ff0000"},
                                "overview": {"went_well": ["a<script>"], "to_improve": [], "summary": "s"},
                                "issues": [{"t_start": 65.5, "t_end": 70, "title": "T", "description": "D",
                                            "category": "c", "team_phase": "defense"}]})
    assert "<script>" not in h and "&lt;b&gt;X" in h
    assert "href='t:65.5'" in h and "1:05–1:10" in h
