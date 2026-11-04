"""Every exported CSV column is documented in README_FOR_AI.md; game.json lists only files that exist."""
import csv
import json
import re

from fake_project import make_project

from replay import data_export
from replay.data_export import export_simple
from replay.export_docs import csv_column_docs


def _header(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def _documented(readme: str) -> set[str]:
    out = set()
    for tok in re.findall(r"`([^`\n]+)`", readme):
        for part in tok.split(","):
            out.add(part.strip())
    return out


def test_every_exported_column_is_documented(tmp_path):
    out = export_simple(make_project(tmp_path / "proj"), tmp_path / "out")
    readme = (out / "README_FOR_AI.md").read_text(encoding="utf-8")
    doc = _documented(readme)
    missing = []
    for f in sorted(out.glob("*.csv")):
        for c in _header(f):
            if c not in doc:
                missing.append(f"{f.name}:{c}")
    assert not missing, missing
    assert "(undocumented)" not in readme


def test_registry_covers_headers():
    for fname, (cols, docs) in csv_column_docs().items():
        assert [c for c in cols if c not in docs] == [], fname


def test_readme_documents_clock_events_and_zones(tmp_path):
    out = export_simple(make_project(tmp_path / "proj"), tmp_path / "out")
    readme = (out / "README_FOR_AI.md").read_text(encoding="utf-8")
    assert "There is no game clock" not in readme
    assert "Game-clock minutes" in readme
    assert "always 2" not in readme and "`source` | `auto`" not in readme
    assert "Shot zones and the 3-point margin" in readme and "zone_uncertain" in readme


def test_file_descriptions_not_mutated_and_only_existing_files(tmp_path):
    before = dict(data_export.FILE_DESCRIPTIONS)
    out = export_simple(make_project(tmp_path / "proj"), tmp_path / "out")
    assert data_export.FILE_DESCRIPTIONS == before
    files = json.loads((out / "game.json").read_text(encoding="utf-8"))["files"]
    for name in files:
        assert (out / name).exists(), name
