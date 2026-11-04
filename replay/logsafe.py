"""Log hygiene: API keys never reach a log file, and log files never grow without limit.

- ``redact(text)``: hides API keys (Google ``AIza...``, ``key=...``, ``x-goog-api-key``, ``Bearer ...``,
  ``sk-...``, ``api_key: ...``) and every secret registered with ``register_secret`` (e.g. the key the user
  typed into the AI dialog).
- ``rotate(path)``: ``app.log`` -> ``app.log.1`` -> ... -> ``app.log.5`` (oldest dropped) when it is too big.
- ``RotatingStream``: a text stream for ``sys.stdout``/``sys.stderr`` (the app runs under pythonw, without a
  console) that redacts every write and rotates at 5 MB, keeping the last 5 files.

Standard library only; importing this module is instant.
"""

from __future__ import annotations

import io
import os
import re
import threading
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024
BACKUPS = 5

_SECRETS: set[str] = set()
_PATTERNS = [
    (re.compile(r"AIza[0-9A-Za-z_\-]{30,}"), "AIza***"),  # Google API keys (AI Studio / Gemini)
    (re.compile(r"(?i)(x-goog-api-key[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]+"), r"\1***"),
    (re.compile(r"(?i)(\bapi[_-]?key[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]{6,}"), r"\1***"),
    (re.compile(r"(?i)([?&;]key=)[^\s&\"'#]{6,}"), r"\1***"),
    (re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/\-]{8,}=*"), r"\1***"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), "sk-***"),
]


def register_secret(secret: str | None) -> None:
    """Remember a secret (e.g. the user's API key) so it is masked wherever it would appear in a log."""
    s = (secret or "").strip()
    if len(s) >= 6:
        _SECRETS.add(s)


def redact(text: str) -> str:
    text = str(text)
    for s in _SECRETS:
        if s in text:
            text = text.replace(s, "***")
    for rx, rep in _PATTERNS:
        text = rx.sub(rep, text)
    return text


def rotate(path: Path | str, max_bytes: int = MAX_BYTES, backups: int = BACKUPS, force: bool = False) -> bool:
    """Shift ``path`` to ``path.1`` (and .1 -> .2 ...) if it is bigger than ``max_bytes``. Never raises: a file
    that is open in another program (Windows) stays where it is. True when rotated."""
    path = Path(path)
    try:
        if not path.exists() or (not force and path.stat().st_size <= max_bytes):
            return False
        oldest = path.with_name(f"{path.name}.{backups}")
        if oldest.exists():
            oldest.unlink()
        for i in range(backups - 1, 0, -1):
            src = path.with_name(f"{path.name}.{i}")
            if src.exists():
                os.replace(src, path.with_name(f"{path.name}.{i + 1}"))
        os.replace(path, path.with_name(f"{path.name}.1"))
        return True
    except OSError:
        return False


class RotatingStream(io.TextIOBase):
    """Append-only, line-buffered, redacting, size-rotating text stream (thread safe)."""

    def __init__(self, path: Path | str, max_bytes: int = MAX_BYTES, backups: int = BACKUPS):
        super().__init__()
        self.path = Path(path)
        self.max_bytes, self.backups = max_bytes, backups
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rotate(self.path, max_bytes, backups)
        self._f = self._open()

    def _open(self):
        return open(self.path, "a", encoding="utf-8", errors="replace", buffering=1)

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return "utf-8"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return self._f.fileno()

    def write(self, s: str) -> int:
        if not s:
            return 0
        with self._lock:
            try:
                self._f.write(redact(s))
                if self._f.tell() > self.max_bytes:
                    self._rollover()
            except (OSError, ValueError):
                pass  # a full disk or a closed file must never crash the app through a print()
        return len(s)

    def _rollover(self) -> None:
        self._f.close()
        rotate(self.path, self.max_bytes, self.backups, force=True)
        self._f = self._open()  # if the rename failed (file open elsewhere) keep appending to the same file

    def flush(self) -> None:
        with self._lock:
            try:
                self._f.flush()
            except (OSError, ValueError):
                pass

    def close(self) -> None:
        with self._lock:
            try:
                self._f.close()
            finally:
                super().close()
