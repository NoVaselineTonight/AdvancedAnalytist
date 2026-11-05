"""Terms of Use and Privacy Policy shipped with the app (TERMS.md, PRIVACY.md next to this file).

Bump ``LEGAL_VERSION`` whenever either text changes in substance: the app then asks the user to accept again on the
next start (replay/gui/legal_dialog.py).
"""

from __future__ import annotations

from pathlib import Path

LEGAL_VERSION = "1.0"
_DIR = Path(__file__).resolve().parent


def terms_text() -> str:
    return (_DIR / "TERMS.md").read_text(encoding="utf-8")


def privacy_text() -> str:
    return (_DIR / "PRIVACY.md").read_text(encoding="utf-8")
