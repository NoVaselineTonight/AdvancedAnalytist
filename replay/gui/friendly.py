"""Friendly error texts: turn a technical error into a plain sentence plus one "what to do" line.

    explain("PermissionError: [Errno 13] ...")  -> Advice(headline, what_to_do)
    warn(parent, "Could not open the Replay file", exc)   # message box: headline, what to do, details hidden
    job_error_html(text)                                  # one-line HTML for the job status line

The original text is never lost: it is shown under "Show details" (message box) or as a tooltip/second line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from PySide6.QtWidgets import QMessageBox, QWidget

APP_NAME = "AdvancedAnalytist"


@dataclass(frozen=True)
class Advice:
    headline: str
    what_to_do: str


# (regex on the lower-cased error text, headline, what to do) - first match wins
_RULES: list[tuple[str, str, str]] = [
    (r"different settings",
     "The analysis was started with different settings.",
     "Press \"Re-analyze with new settings\", or put the settings back as they were."),
    (r"no space left|disk full|errno 28|not enough space|there is not enough space",
     "The disk is full.",
     "Free some disk space (delete old Replays or move videos to another drive), then press the button again."),
    (r"permission|access is denied|errno 13|being used by another process|winerror 32|\bin use\b",
     "A file is in use or the app is not allowed to write it.",
     "Close other programs that may have the video or Replay open (video player, Explorer preview), then try again."),
    (r"memoryerror|out of memory|cannot allocate|bad_alloc|not enough memory",
     "The computer ran out of memory.",
     "Close other programs and try again. In Overview choose a smaller \"Detector size\" or \"Processing device: CPU\"."),
    (r"directml|\bdml\b|d3d12|\bgpu\b|onnxruntime|execution provider",
     "The graphics card (GPU) could not be used.",
     "In Overview set \"Processing device\" to CPU and press the button again (slower, but works everywhere)."),
    (r"newer version",
     "This Replay was made with a newer version of the app.",
     "Update AdvancedAnalytist on this computer, then open the file again."),
    (r"not a zip|not an advancedanalytist|damaged|incomplete|bad ?zip|crc",
     "The Replay file is damaged or not a Replay file.",
     "Ask for the .aareplay file again (copy it once more; an interrupted download or copy breaks it)."),
    (r"can't be read as a video|could not open|invalid data found|moov atom|codec|ffmpeg|ffprobe|no video stream",
     "The video file could not be read.",
     "Check that the file plays in a normal video player. If it does not, export it again from the camera or "
     "the streaming site (MP4 works best)."),
    (r"filenotfound|no such file|errno 2|not found|cannot find|was not found|missing",
     "A file the Replay needs was not found.",
     "If you moved or renamed the original video or a drive is unplugged, put it back and try again."),
    (r"could not find the court|court not found|no court",
     "The court lines could not be found automatically.",
     "Open \"Court fix (advanced)\" and click a few court points on 2-3 frames; the replay then builds by itself."),
    (r"timeout|timed out|connection|network|urlopen|ssl|http",
     "The internet connection failed.",
     "Check the connection and try again. The analysis itself works without internet."),
    (r"scoresheet",
     "The scoresheet could not be used.",
     "Use the digital PDF from the federation system, or a CSV (a photo or scan cannot be read)."),
]

_GENERIC = Advice("Something went wrong.",
                  "Try again. If it keeps happening, close and reopen the app; your Replays are kept safely.")


def explain(err: object) -> Advice:
    """Plain-language headline and a "what to do" line for an exception or error text."""
    text = f"{type(err).__name__}: {err}" if isinstance(err, BaseException) else str(err or "")
    low = text.lower()
    for pat, head, todo in _RULES:
        if re.search(pat, low):
            return Advice(head, todo)
    return _GENERIC


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def job_error_html(err: str) -> str:
    """Status-line HTML: friendly headline, what to do, and the original text in grey (short)."""
    a = explain(err)
    detail = str(err or "")
    detail = detail if len(detail) <= 160 else detail[:157] + "…"
    return (f"<b style='color:#b91c1c'>{_esc(a.headline)}</b> <b>What to do:</b> {_esc(a.what_to_do)}"
            f"<br><span style='color:gray; font-size:small'>Details: {_esc(detail)}</span>")


def message_html(context: str, err: object) -> str:
    a = explain(err)
    return (f"<b>{_esc(context)}</b><br><br>{_esc(a.headline)}<br><br>"
            f"<b>What to do:</b> {_esc(a.what_to_do)}")


def warn(parent: QWidget | None, context: str, err: object, title: str = APP_NAME) -> None:
    """Warning box: context + friendly headline + "What to do"; the technical text under "Show Details…"."""
    box = QMessageBox(QMessageBox.Icon.Warning, title, message_html(context, err), QMessageBox.StandardButton.Ok,
                      parent)
    box.setTextFormat(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.TextFormat.RichText)
    detail = f"{type(err).__name__}: {err}" if isinstance(err, BaseException) else str(err or "")
    if detail:
        box.setDetailedText(detail)
    box.exec()
