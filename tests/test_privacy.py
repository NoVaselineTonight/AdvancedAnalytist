"""Privacy: minors' names and licence numbers never reach Gemini prompts or exports; names in exports are opt-in."""

import json

from fake_project import make_project

from replay import ai_analysis as ai
from replay import roster_names as rn
from replay.data_export import export_simple

NAME, LIC = "Testi Pelaaja", "123456"


def _proj(tmp_path):
    root = make_project(tmp_path / "p")
    p = root / "results" / "replay_data.json"
    rd = json.loads(p.read_text(encoding="utf-8"))
    rd.setdefault("track_info", {})["1"] = {"number": "7", "team": "A"}
    p.write_text(json.dumps(rd), encoding="utf-8")
    rn.save_assignment(root, rn.Assignment(rn.Roster(team="Kouvot", players=[rn.Player("7", NAME)]), None))
    # scoresheet.json as written by the app: name kept, licence stripped (also write a licence defensively-absent)
    (root / "scoresheet.json").write_text(json.dumps(
        {"teams": [{"side": "A", "players": [{"number": "7", "name": NAME}]}]}), encoding="utf-8")
    tl = json.loads((root / "timeline.json").read_text(encoding="utf-8"))
    tl["tracks"][0]["marks"].append({"t": 1, "title": f"{NAME} missed", "note": f"{NAME} lic {LIC}"})
    (root / "timeline.json").write_text(json.dumps(tl), encoding="utf-8")
    return root


def _all_text(out):
    return "\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in out.rglob("*") if f.is_file())


def test_gemini_digest_has_no_names_or_licence(tmp_path):
    d = ai.build_game_digest(_proj(tmp_path), {"A": "#abcdef"}, sample_hz=2.0)
    text = d.header + chr(10) + chr(10).join(ln for _, ln in d.frames)
    for secret in (NAME, "Testi", "Pelaaja", LIC):
        assert secret not in text
    assert "[note withheld]" in text and "Fast break" not in text and "PLAYER NAMES" not in text


def test_export_names_opt_in_and_no_licence(tmp_path):
    root = _proj(tmp_path)
    off = export_simple(root, tmp_path / "off")
    assert NAME not in "\n".join((off / f).read_text(encoding="utf-8") for f in
                                 ("players.csv", "track_summary.csv", "events.csv")) and not (off / "roster.csv").exists()
    assert LIC not in "\n".join(f.read_text(encoding="utf-8", errors="ignore")
                                for f in off.rglob("*") if f.is_file() and f.name != "timeline_marks.csv"
                                and f.name != "timeline.json" and f.name != "replay_2d.html"
                                and f.name != "README_FOR_AI.md")
    on = export_simple(root, tmp_path / "on", include_names=True)
    assert NAME in (on / "roster.csv").read_text(encoding="utf-8")
    assert LIC not in (on / "roster.csv").read_text(encoding="utf-8")
    assert "player_name" in (on / "players.csv").read_text(encoding="utf-8").splitlines()[0]
