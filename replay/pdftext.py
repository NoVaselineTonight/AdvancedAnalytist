"""Small built-in PDF text + vector-mark extractor (no third-party PDF library needed).

Used by replay/scoresheet.py to read generated (not scanned) scoresheet PDFs. It understands what PDF
generators such as FPDF / tFPDF / TCPDF / Acrobat write for simple documents:

* objects anywhere in the file (scanned with a regex, so a broken xref table does not matter), object streams
  (``/Type /ObjStm``), FlateDecode / ASCIIHex / ASCII85 / uncompressed streams;
* the page tree (inherited Resources / MediaBox), content streams (single or arrays) and form XObjects (``Do``,
  recursively, with their /Matrix);
* text operators (Tf Tm Td TD T* TL Tc Tw Tz Ts Tj TJ ' ") with the current transformation matrix, so every
  text run gets its page position (PDF points, origin bottom-left) and effective font size;
* font encodings: WinAnsi / Standard / MacRoman base encodings, /Differences arrays (Adobe glyph names),
  /ToUnicode CMaps (bfchar / bfrange) and two-byte Identity-H Type0 fonts;
* vector marks: straight line segments and circles / ellipses (four Bezier curves), with stroke colour, line
  width and whether they are filled - scoresheets mark points (slashes, dots, circles) with these.

Public interface (kept small so that pypdf / pdfminer could replace the backend later)::

    pages = extract(path_or_bytes)            # -> list[Page]
    page.spans  -> [Span(x, y, size, text, font, color)]
    page.marks  -> [Mark(kind="line"|"circle"|"rect", ...)]
    page.lines() -> text lines (spans grouped by baseline)

Raises :class:`PdfError` for files that are not PDFs or cannot be decoded. A PDF without any text (scanned or
photographed page) returns pages with empty ``spans``; the caller decides what to do (scoresheet.py raises a
clear "scanned sheet: OCR not supported yet" error).
"""

from __future__ import annotations

import base64
import math
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

__all__ = ["PdfError", "Span", "Mark", "Page", "extract", "BACKEND"]

BACKEND = "builtin"


class PdfError(ValueError):
    """The file is not a readable PDF (or uses features this small extractor does not support)."""


# ====================================================================================== results
@dataclass
class Span:
    x: float
    y: float
    size: float
    text: str
    font: str = ""
    color: tuple[float, float, float] = (0.0, 0.0, 0.0)  # fill colour (RGB 0..1) = text colour
    width: float = 0.0  # approximate advance width in points
    seq: int = 0  # drawing order on the page (spans and marks share one counter)
    depth: int = 0  # 0 = page content, 1+ = inside a form XObject (templates / backgrounds)

    @property
    def x1(self) -> float:
        return self.x + self.width

    @property
    def cx(self) -> float:
        return self.x + self.width / 2.0


@dataclass
class Mark:
    kind: str  # "line" | "circle" | "rect"
    x0: float
    y0: float
    x1: float
    y1: float
    color: tuple[float, float, float] = (0.0, 0.0, 0.0)  # stroke colour (or fill colour when only filled)
    width: float = 1.0
    filled: bool = False
    seq: int = 0
    depth: int = 0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def r(self) -> float:
        return (abs(self.x1 - self.x0) + abs(self.y1 - self.y0)) / 4.0


@dataclass
class Page:
    index: int
    width: float
    height: float
    spans: list[Span] = field(default_factory=list)
    marks: list[Mark] = field(default_factory=list)

    def lines(self, tol: float = 1.5) -> list[list[Span]]:
        """Spans grouped into lines (same baseline within ``tol`` points), top to bottom, left to right."""
        out: list[list[Span]] = []
        for s in sorted(self.spans, key=lambda s: (-s.y, s.x)):
            if out and abs(out[-1][0].y - s.y) <= tol:
                out[-1].append(s)
            else:
                out.append([s])
        for ln in out:
            ln.sort(key=lambda s: s.x)
        return out

    def text(self) -> str:
        return "\n".join(" ".join(s.text for s in ln if s.text.strip()) for ln in self.lines())


# ====================================================================================== object parser
class _Ref:
    __slots__ = ("num",)

    def __init__(self, num: int):
        self.num = num

    def __repr__(self) -> str:
        return f"{self.num} R"


class _Name(str):
    pass


class _Op(str):
    pass


_WS = b" \t\r\n\f\x00"
_DELIM = b"()<>[]{}/%"
_NUM_RE = re.compile(rb"[+-]?(?:\d+\.?\d*|\.\d+)")


def _lit_string(data: bytes, i: int) -> tuple[bytes, int]:
    """Parse a literal string starting after '('. Returns (bytes, index after ')')."""
    out = bytearray()
    depth = 1
    n = len(data)
    while i < n:
        c = data[i]
        if c == 0x5C:  # backslash
            i += 1
            if i >= n:
                break
            e = data[i]
            esc = {0x6E: 10, 0x72: 13, 0x74: 9, 0x62: 8, 0x66: 12, 0x28: 0x28, 0x29: 0x29, 0x5C: 0x5C}
            if e in esc:
                out.append(esc[e])
                i += 1
            elif 0x30 <= e <= 0x37:
                j = i
                while j < n and j < i + 3 and 0x30 <= data[j] <= 0x37:
                    j += 1
                out.append(int(data[i:j], 8) & 0xFF)
                i = j
            elif e in (0x0D, 0x0A):  # line continuation
                i += 1
                if e == 0x0D and i < n and data[i] == 0x0A:
                    i += 1
            else:
                out.append(e)
                i += 1
            continue
        if c == 0x28:
            depth += 1
        elif c == 0x29:
            depth -= 1
            if depth == 0:
                return bytes(out), i + 1
        out.append(c)
        i += 1
    return bytes(out), i


def _tokens(data: bytes, i: int = 0) -> Iterator[tuple[Any, int]]:
    """Yield (token, next index). Tokens: float/int, bytes (strings), _Name, _Op (keywords/operators),
    '[' ']' '<<' '>>' markers as _Op."""
    n = len(data)
    while i < n:
        c = data[i]
        if c in _WS:
            i += 1
            continue
        if c == 0x25:  # % comment
            while i < n and data[i] not in b"\r\n":
                i += 1
            continue
        if c == 0x28:  # (
            s, i = _lit_string(data, i + 1)
            yield s, i
            continue
        if c == 0x3C:  # <
            if i + 1 < n and data[i + 1] == 0x3C:
                yield _Op("<<"), i + 2
                i += 2
                continue
            j = data.find(b">", i)
            if j < 0:
                j = n
            hx = re.sub(rb"[^0-9A-Fa-f]", b"", data[i + 1:j])
            if len(hx) % 2:
                hx += b"0"
            yield bytes.fromhex(hx.decode()), j + 1
            i = j + 1
            continue
        if c == 0x3E:  # >
            if i + 1 < n and data[i + 1] == 0x3E:
                yield _Op(">>"), i + 2
                i += 2
                continue
            i += 1
            continue
        if c in b"[]{}":
            yield _Op(chr(c)), i + 1
            i += 1
            continue
        if c == 0x2F:  # /Name
            j = i + 1
            while j < n and data[j] not in _WS and data[j] not in _DELIM:
                j += 1
            raw = data[i + 1:j]
            raw = re.sub(rb"#([0-9A-Fa-f]{2})", lambda m: bytes([int(m.group(1), 16)]), raw)
            yield _Name(raw.decode("latin-1")), j
            i = j
            continue
        m = _NUM_RE.match(data, i)
        if m and (m.end() >= n or data[m.end()] in _WS or data[m.end()] in _DELIM):
            t = m.group(0)
            yield (float(t) if b"." in t else int(t)), m.end()
            i = m.end()
            continue
        j = i
        while j < n and data[j] not in _WS and data[j] not in _DELIM:
            j += 1
        if j == i:
            j = i + 1
        yield _Op(data[i:j].decode("latin-1")), j
        i = j


def _parse_value(it: Iterator[tuple[Any, int]], first: Any = None) -> Any:
    """Parse one PDF value (dict, array, ref, scalar) from a token iterator."""
    tok = first if first is not None else next(it)[0]
    if isinstance(tok, _Op):
        if tok == "<<":
            d: dict[str, Any] = {}
            while True:
                k = next(it)[0]
                if k == ">>":
                    return d
                v = _parse_any(it)
                if isinstance(k, _Name):
                    d[str(k)] = v
        if tok == "[":
            arr: list[Any] = []
            while True:
                t = next(it)[0]
                if t == "]":
                    return _fold_refs(arr)
                arr.append(_parse_value(it, t) if t in ("<<", "[") else t)
        if tok == "true":
            return True
        if tok == "false":
            return False
        if tok == "null":
            return None
    return tok


def _fold_refs(arr: list[Any]) -> list[Any]:
    out: list[Any] = []
    i = 0
    while i < len(arr):
        if (i + 2 < len(arr) and isinstance(arr[i], int) and isinstance(arr[i + 1], int)
                and isinstance(arr[i + 2], _Op) and arr[i + 2] == "R"):
            out.append(_Ref(arr[i]))
            i += 3
        else:
            out.append(arr[i])
            i += 1
    return out


class _Peek:
    def __init__(self, it: Iterator[tuple[Any, int]]):
        self.it = it
        self.buf: list[tuple[Any, int]] = []

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Any, int]:
        return self.buf.pop() if self.buf else next(self.it)

    def push(self, x: tuple[Any, int]) -> None:
        self.buf.append(x)


def _parse_any(it: Any) -> Any:
    t = next(it)
    tok = t[0]
    if tok in ("<<", "["):
        return _parse_value(it, tok)
    if isinstance(tok, int) and isinstance(it, _Peek):
        try:
            t2 = next(it)
        except StopIteration:
            return _parse_value(it, tok)
        if isinstance(t2[0], int):
            try:
                t3 = next(it)
            except StopIteration:
                it.push(t2)
                return tok
            if t3[0] == "R":
                return _Ref(tok)
            it.push(t3)
            it.push(t2)
            return tok
        it.push(t2)
        return tok
    return _parse_value(it, tok)


def _parse_obj_bytes(data: bytes, i: int = 0) -> Any:
    it = _Peek(_tokens(data, i))
    return _parse_any(it)


# ====================================================================================== document
_OBJ_RE = re.compile(rb"(?<![0-9])(\d+)\s+(\d+)\s+obj\b")


class _Doc:
    def __init__(self, data: bytes):
        if not data.lstrip()[:5].startswith(b"%PDF"):
            raise PdfError("not a PDF file (no %PDF header)")
        self.data = data
        self.objs: dict[int, Any] = {}
        self.streams: dict[int, bytes] = {}  # raw (undecoded) stream bytes
        self._scan()

    def _scan(self) -> None:
        d = self.data
        for m in _OBJ_RE.finditer(d):
            num = int(m.group(1))
            try:
                it = _Peek(_tokens(d, m.end()))
                val = _parse_any(it)
                nxt, pos = next(it)
            except (StopIteration, ValueError, IndexError):
                continue
            if nxt == "stream" and isinstance(val, dict):
                start = pos
                if d[start:start + 2] == b"\r\n":
                    start += 2
                elif d[start:start + 1] in (b"\n", b"\r"):
                    start += 1
                length = val.get("Length")
                raw = None
                if isinstance(length, int) and 0 <= length <= len(d) - start:
                    end = start + length
                    if d[end:end + 30].lstrip().startswith(b"endstream"):
                        raw = d[start:end]
                if raw is None:  # indirect / wrong length: search for endstream
                    end = d.find(b"endstream", start)
                    if end < 0:
                        continue
                    raw = d[start:end].rstrip(b"\r\n")
                self.streams[num] = raw
            self.objs[num] = val  # later definitions (incremental updates) win
        for num, val in list(self.objs.items()):
            if isinstance(val, dict) and val.get("Type") == "ObjStm" and num in self.streams:
                try:
                    self._objstm(num, val)
                except Exception:  # noqa: BLE001 - a broken object stream just loses its objects
                    continue
        if not self.objs:
            raise PdfError("no PDF objects found (damaged file?)")

    def _objstm(self, num: int, d: dict) -> None:
        data = self.stream(num)
        n, first = int(self.resolve(d.get("N")) or 0), int(self.resolve(d.get("First")) or 0)
        head = [t for t, _ in zip(_tokens(data[:first]), range(2 * n))]
        nums = [int(head[k][0]) for k in range(0, len(head), 2)]
        offs = [int(head[k][0]) for k in range(1, len(head), 2)]
        for on, off in zip(nums, offs):
            if on not in self.objs:
                self.objs[on] = _parse_obj_bytes(data, first + off)

    def resolve(self, v: Any, depth: int = 0) -> Any:
        while isinstance(v, _Ref) and depth < 32:
            v = self.objs.get(v.num)
            depth += 1
        return v

    def stream(self, num_or_ref: Any) -> bytes:
        num = num_or_ref.num if isinstance(num_or_ref, _Ref) else int(num_or_ref)
        raw = self.streams.get(num)
        if raw is None:
            return b""
        d = self.objs.get(num) or {}
        filt = self.resolve(d.get("Filter"))
        filters = filt if isinstance(filt, list) else [filt] if filt else []
        data = raw
        for f in filters:
            f = str(self.resolve(f))
            if f in ("FlateDecode", "Fl"):
                try:
                    data = zlib.decompress(data)
                except zlib.error:
                    try:
                        data = zlib.decompressobj().decompress(data)
                    except zlib.error as e:
                        raise PdfError(f"cannot inflate stream {num}: {e}") from e
            elif f in ("ASCIIHexDecode", "AHx"):
                hx = re.sub(rb"[^0-9A-Fa-f]", b"", data.split(b">")[0])
                data = bytes.fromhex((hx + b"0" * (len(hx) % 2)).decode())
            elif f in ("ASCII85Decode", "A85"):
                body = data.strip()
                if body.startswith(b"<~"):
                    body = body[2:]
                data = base64.a85decode(body.split(b"~>")[0], adobe=False)
            else:  # DCT / JBIG2 / CCITT ...: images, not text
                return b""
        return data

    # ---- page tree
    def pages(self) -> list[dict]:
        root = None
        for v in self.objs.values():
            if isinstance(v, dict) and v.get("Type") == "Catalog":
                root = self.resolve(v.get("Pages"))
        out: list[dict] = []

        def walk(node: Any, inherited: dict, seen: set) -> None:
            node = self.resolve(node)
            if not isinstance(node, dict) or id(node) in seen:
                return
            seen.add(id(node))
            inh = dict(inherited)
            for k in ("Resources", "MediaBox", "Rotate"):
                if k in node:
                    inh[k] = node[k]
            if node.get("Type") == "Pages" or "Kids" in node:
                for k in self.resolve(node.get("Kids")) or []:
                    walk(k, inh, seen)
            else:
                pg = dict(inh)
                pg.update(node)
                out.append(pg)

        if root is not None:
            walk(root, {}, set())
        if not out:  # no catalog: every /Type /Page object in file order
            out = [v for _, v in sorted(self.objs.items()) if isinstance(v, dict) and v.get("Type") == "Page"]
        return out


# ====================================================================================== fonts
_GLYPHS = {
    "space": " ", "exclam": "!", "quotedbl": '"', "numbersign": "#", "dollar": "$", "percent": "%",
    "ampersand": "&", "quotesingle": "'", "quoteright": "’", "quoteleft": "‘", "parenleft": "(",
    "parenright": ")", "asterisk": "*", "plus": "+", "comma": ",", "hyphen": "-", "minus": "-", "period": ".",
    "slash": "/", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "colon": ":", "semicolon": ";", "less": "<", "equal": "=",
    "greater": ">", "question": "?", "at": "@", "bracketleft": "[", "backslash": "\\", "bracketright": "]",
    "underscore": "_", "braceleft": "{", "bar": "|", "braceright": "}", "endash": "–", "emdash": "—",
    "bullet": "•", "degree": "°", "periodcentered": "·", "quotedblleft": "“",
    "quotedblright": "”", "adieresis": "ä", "Adieresis": "Ä", "odieresis": "ö", "Odieresis": "Ö",
    "aring": "å", "Aring": "Å", "udieresis": "ü", "Udieresis": "Ü", "eacute": "é", "Eacute": "É",
    "egrave": "è", "aacute": "á", "oacute": "ó", "scaron": "š", "Scaron": "Š", "zcaron": "ž", "Zcaron": "Ž",
    "otilde": "õ", "ntilde": "ñ", "ccedilla": "ç", "oslash": "ø", "Oslash": "Ø", "ae": "æ", "AE": "Æ",
    "germandbls": "ß", "nbspace": " ", "uni00A0": " ",
}
for _c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
    _GLYPHS[_c] = _c


def _glyph_char(name: str) -> str:
    if name in _GLYPHS:
        return _GLYPHS[name]
    m = re.fullmatch(r"uni([0-9A-Fa-f]{4})", name) or re.fullmatch(r"u([0-9A-Fa-f]{4,6})", name)
    if m:
        try:
            return chr(int(m.group(1), 16))
        except ValueError:
            return ""
    base = name.split(".")[0].split("_")[0]
    return _GLYPHS.get(base, "")


def _base_table(name: str) -> dict[int, str]:
    codec = {"WinAnsiEncoding": "cp1252", "MacRomanEncoding": "mac_roman"}.get(name, "cp1252")
    out = {}
    for b in range(256):
        try:
            out[b] = bytes([b]).decode(codec)
        except UnicodeDecodeError:
            out[b] = bytes([b]).decode("latin-1")
    return out


def _parse_cmap(data: bytes) -> dict[int, str]:
    out: dict[int, str] = {}
    txt = data

    def hexval(h: bytes) -> int:
        return int(h, 16) if h else 0

    def hexstr(h: bytes) -> str:
        h = re.sub(rb"\s", b"", h)
        if len(h) % 4:
            try:
                return bytes.fromhex(h.decode()).decode("latin-1")
            except ValueError:
                return ""
        try:
            return bytes.fromhex(h.decode()).decode("utf-16-be", errors="ignore")
        except ValueError:
            return ""

    for blk in re.finditer(rb"beginbfchar(.*?)endbfchar", txt, re.S):
        for a, b in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f\s]*)>", blk.group(1)):
            out[hexval(a)] = hexstr(b)
    for blk in re.finditer(rb"beginbfrange(.*?)endbfrange", txt, re.S):
        body = blk.group(1)
        for m in re.finditer(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(<[0-9A-Fa-f\s]*>|\[[^\]]*\])", body):
            lo, hi = hexval(m.group(1)), hexval(m.group(2))
            if hi - lo > 0x10000:
                continue
            dst = m.group(3)
            if dst.startswith(b"["):
                items = re.findall(rb"<([0-9A-Fa-f\s]*)>", dst)
                for k, h in enumerate(items):
                    out[lo + k] = hexstr(h)
            else:
                h = re.sub(rb"[<>\s]", b"", dst)
                start = hexval(h)
                width = len(h)
                for code in range(lo, hi + 1):
                    v = start + (code - lo)
                    out[code] = hexstr(f"{v:0{width}X}".encode())
    return out


class _Font:
    def __init__(self, doc: _Doc, fd: Any, name: str):
        fd = doc.resolve(fd) if fd is not None else {}
        fd = fd if isinstance(fd, dict) else {}
        self.name = str(doc.resolve(fd.get("BaseFont")) or name)
        self.two_byte = doc.resolve(fd.get("Subtype")) == "Type0"
        self.tounicode: dict[int, str] = {}
        tu = fd.get("ToUnicode")
        if isinstance(tu, _Ref):
            try:
                self.tounicode = _parse_cmap(doc.stream(tu))
            except Exception:  # noqa: BLE001
                self.tounicode = {}
        enc = doc.resolve(fd.get("Encoding"))
        base = "StandardEncoding" if not enc else enc if isinstance(enc, str) else \
            str(doc.resolve(enc.get("BaseEncoding")) or "WinAnsiEncoding")
        self.table = _base_table(str(base))
        if isinstance(enc, dict):
            code = 0
            for v in doc.resolve(enc.get("Differences")) or []:
                if isinstance(v, int):
                    code = v
                elif isinstance(v, str):
                    ch = _glyph_char(str(v))
                    if ch:
                        self.table[code] = ch
                    code += 1
        self.widths: dict[int, float] = {}
        fc = doc.resolve(fd.get("FirstChar"))
        ws = doc.resolve(fd.get("Widths"))
        if isinstance(fc, int) and isinstance(ws, list):
            for k, w in enumerate(ws):
                w = doc.resolve(w)
                if isinstance(w, (int, float)):
                    self.widths[fc + k] = float(w)
        self.default_w = 556.0 if "Helvetica" in self.name or "Arial" in self.name else 500.0
        if self.two_byte:
            df = doc.resolve((doc.resolve(fd.get("DescendantFonts")) or [None])[0])
            if isinstance(df, dict):
                dw = doc.resolve(df.get("DW"))
                self.default_w = float(dw) if isinstance(dw, (int, float)) else 1000.0
                w = doc.resolve(df.get("W")) or []
                k = 0
                while k < len(w) - 1:
                    a = doc.resolve(w[k])
                    b = doc.resolve(w[k + 1])
                    if isinstance(b, list):
                        for j, x in enumerate(b):
                            if isinstance(x, (int, float)):
                                self.widths[int(a) + j] = float(x)
                        k += 2
                    elif k + 2 < len(w):
                        c = doc.resolve(w[k + 2])
                        for code in range(int(a), int(b) + 1):
                            if isinstance(c, (int, float)) and code - int(a) < 65536:
                                self.widths[code] = float(c)
                        k += 3
                    else:
                        break

    def codes(self, s: bytes) -> list[int]:
        if self.two_byte:
            return [(s[i] << 8) | (s[i + 1] if i + 1 < len(s) else 0) for i in range(0, len(s), 2)]
        return list(s)

    def decode(self, s: bytes) -> tuple[str, float]:
        """(text, advance in text-space units / 1000 * font size)."""
        out = []
        adv = 0.0
        for c in self.codes(s):
            if c in self.tounicode:
                out.append(self.tounicode[c])
            elif self.two_byte:
                out.append(chr(c) if 32 <= c < 0xD800 else "")
            else:
                out.append(self.table.get(c, ""))
            adv += self.widths.get(c, self.default_w)
        return "".join(out), adv / 1000.0

    @staticmethod
    def helvetica_width(text: str) -> float:
        return 0.556 * len(text)


# ====================================================================================== content interpreter
def _mul(a: list[float], b: list[float]) -> list[float]:
    return [a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3],
            a[4] * b[0] + a[5] * b[2] + b[4], a[4] * b[1] + a[5] * b[3] + b[5]]


def _apply(m: list[float], x: float, y: float) -> tuple[float, float]:
    return m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]


def _rgb(op: str, args: list[Any]) -> tuple[float, float, float] | None:
    nums = [float(a) for a in args if isinstance(a, (int, float))]
    if op in ("g", "G") and nums:
        return (nums[0],) * 3  # type: ignore[return-value]
    if op in ("rg", "RG") and len(nums) >= 3:
        return nums[0], nums[1], nums[2]
    if op in ("k", "K") and len(nums) >= 4:
        c, m, y, k = nums[:4]
        return (1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)
    if op in ("sc", "scn", "SC", "SCN"):
        if len(nums) == 1:
            return (1.0 - nums[0],) * 3  # type: ignore[return-value]  (separation tint ~ black ink)
        if len(nums) == 3:
            return nums[0], nums[1], nums[2]
        if len(nums) == 4:
            c, m, y, k = nums
            return (1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)
    return None


class _Interp:
    def __init__(self, doc: _Doc, page: Page):
        self.doc = doc
        self.page = page
        self.font_cache: dict[int, _Font] = {}
        self.seq = 0
        self.depth = 0

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def font(self, res: dict, name: str) -> _Font:
        fonts = self.doc.resolve(res.get("Font")) or {}
        ref = fonts.get(name) if isinstance(fonts, dict) else None
        key = ref.num if isinstance(ref, _Ref) else id(ref)
        if key not in self.font_cache:
            self.font_cache[key] = _Font(self.doc, ref, name)
        return self.font_cache[key]

    def run(self, data: bytes, res: dict, ctm: list[float], depth: int = 0) -> None:
        if depth > 8:
            return
        res = self.doc.resolve(res) or {}
        gs = {"ctm": list(ctm), "stroke": (0.0, 0.0, 0.0), "fill": (0.0, 0.0, 0.0), "lw": 1.0}
        stack: list[dict] = []
        tm = [1.0, 0, 0, 1.0, 0, 0]
        tlm = list(tm)
        ts = {"font": None, "size": 0.0, "tc": 0.0, "tw": 0.0, "tz": 1.0, "tl": 0.0, "rise": 0.0}
        path: list[tuple[str, list[float]]] = []
        args: list[Any] = []
        it = _tokens(data)
        for tok, pos in it:
            if not isinstance(tok, _Op) or tok in ("<<", "[", "true", "false", "null"):
                if tok in ("<<", "["):
                    try:
                        args.append(_parse_value(it, tok))
                    except StopIteration:
                        break
                else:
                    args.append(tok)
                continue
            op = str(tok)
            try:
                if op == "BI":  # inline image: skip to EI
                    end = data.find(b"EI", pos)
                    it = _tokens(data, end + 2 if end >= 0 else len(data))
                elif op == "q":
                    stack.append(dict(gs, ctm=list(gs["ctm"])))
                elif op == "Q":
                    if stack:
                        gs = stack.pop()
                elif op == "cm" and len(args) >= 6:
                    gs["ctm"] = _mul([float(a) for a in args[-6:]], gs["ctm"])
                elif op == "w" and args:
                    gs["lw"] = float(args[-1])
                elif op in ("g", "rg", "k", "sc", "scn"):
                    c = _rgb(op, args)
                    if c:
                        gs["fill"] = c
                elif op in ("G", "RG", "K", "SC", "SCN"):
                    c = _rgb(op, args)
                    if c:
                        gs["stroke"] = c
                elif op == "BT":
                    tm, tlm = [1.0, 0, 0, 1.0, 0, 0], [1.0, 0, 0, 1.0, 0, 0]
                elif op == "Tf" and len(args) >= 2:
                    ts["font"] = self.font(res, str(args[-2]))
                    ts["size"] = float(args[-1])
                elif op == "Tc" and args:
                    ts["tc"] = float(args[-1])
                elif op == "Tw" and args:
                    ts["tw"] = float(args[-1])
                elif op == "Tz" and args:
                    ts["tz"] = float(args[-1]) / 100.0
                elif op == "TL" and args:
                    ts["tl"] = float(args[-1])
                elif op == "Ts" and args:
                    ts["rise"] = float(args[-1])
                elif op == "Tm" and len(args) >= 6:
                    tm = [float(a) for a in args[-6:]]
                    tlm = list(tm)
                elif op in ("Td", "TD") and len(args) >= 2:
                    tx, ty = float(args[-2]), float(args[-1])
                    if op == "TD":
                        ts["tl"] = -ty
                    tlm = _mul([1, 0, 0, 1, tx, ty], tlm)
                    tm = list(tlm)
                elif op == "T*":
                    tlm = _mul([1, 0, 0, 1, 0, -ts["tl"]], tlm)
                    tm = list(tlm)
                elif op in ("Tj", "'", '"', "TJ"):
                    if op in ("'", '"'):
                        tlm = _mul([1, 0, 0, 1, 0, -ts["tl"]], tlm)
                        tm = list(tlm)
                    items = args[-1] if op == "TJ" and args and isinstance(args[-1], list) else \
                        [args[-1]] if args and isinstance(args[-1], bytes) else []
                    tm = self._show(items, tm, ts, gs)
                elif op in ("m", "l", "c", "v", "y", "re", "h"):
                    nums = [float(a) for a in args if isinstance(a, (int, float))]
                    path.append((op, nums))
                elif op in ("S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "n"):
                    if op != "n":
                        self._paint(path, gs, filled=op not in ("S", "s"), stroked=op not in ("f", "F", "f*"))
                    path = []
                elif op == "Do" and args:
                    self._do(str(args[-1]), res, gs["ctm"], depth)
            except (ValueError, TypeError, IndexError, ZeroDivisionError):
                pass
            args = []

    def _show(self, items: list[Any], tm: list[float], ts: dict, gs: dict) -> list[float]:
        f: _Font | None = ts["font"]
        size = ts["size"]
        text_parts: list[str] = []
        start = None
        total = 0.0
        for it in items:
            if isinstance(it, (int, float)):
                shift = -float(it) / 1000.0 * size * ts["tz"]
                if float(it) < -250 and text_parts and not text_parts[-1].endswith(" "):
                    text_parts.append(" ")
                tm = _mul([1, 0, 0, 1, shift, 0], tm)
                total += shift
                continue
            if not isinstance(it, bytes):
                continue
            if f is None:
                s, adv = it.decode("latin-1"), 0.5 * len(it)
            else:
                s, adv = f.decode(it)
            if start is None:
                start = list(tm)
            n_sp = s.count(" ")
            w = (adv * size + ts["tc"] * len(s) + ts["tw"] * n_sp) * ts["tz"]
            text_parts.append(s)
            tm = _mul([1, 0, 0, 1, w, 0], tm)
            total += w
        if start is None:
            return tm
        trm = _mul([size * ts["tz"], 0, 0, size, 0, ts["rise"]], _mul(start, gs["ctm"]))
        x, y = trm[4], trm[5]
        eff = math.hypot(trm[2], trm[3]) or size
        sx = math.hypot(*(_mul(start, gs["ctm"])[:2]))
        text = "".join(text_parts)
        if text:
            self.page.spans.append(Span(round(x, 2), round(y, 2), round(eff, 2), text,
                                        f.name if f else "", tuple(gs["fill"]), round(total * sx, 2),
                                        self.next_seq(), self.depth))
        return tm

    def _paint(self, path: list[tuple[str, list[float]]], gs: dict, filled: bool, stroked: bool) -> None:
        ctm = gs["ctm"]
        color = gs["stroke"] if stroked else gs["fill"]
        lw = gs["lw"] * math.hypot(ctm[0], ctm[1])
        # split into subpaths
        subs: list[list[tuple[str, list[float]]]] = []
        for op, nums in path:
            if op in ("m", "re") or not subs:
                subs.append([])
            subs[-1].append((op, nums))
        for sp in subs:
            ops = [o for o, _ in sp]
            if ops[0] == "re" and len(sp[0][1]) >= 4:
                x, y, w, h = sp[0][1][:4]
                (ax, ay), (bx, by) = _apply(ctm, x, y), _apply(ctm, x + w, y + h)
                self.page.marks.append(Mark("rect", min(ax, bx), min(ay, by), max(ax, bx), max(ay, by),
                                            color, lw, filled, self.next_seq(), self.depth))
                continue
            pts: list[tuple[float, float]] = []
            n_curves = 0
            for op, nums in sp:
                if op in ("m", "l") and len(nums) >= 2:
                    pts.append(_apply(ctm, nums[-2], nums[-1]))
                elif op in ("c", "v", "y") and len(nums) >= 4:
                    pts.append(_apply(ctm, nums[-2], nums[-1]))
                    n_curves += 1
            if len(pts) < 2:
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            if n_curves >= 4 and n_curves >= len(pts) - 2:  # closed curve: circle / ellipse
                # control points are not in pts; the 4 on-curve points span the ellipse
                self.page.marks.append(Mark("circle", min(xs), min(ys), max(xs), max(ys), color, lw, filled,
                                            self.next_seq(), self.depth))
                continue
            if n_curves == 0:
                for (ax, ay), (bx, by) in zip(pts, pts[1:]):
                    self.page.marks.append(Mark("line", ax, ay, bx, by, color, lw, False, self.next_seq(),
                                                self.depth))

    def _do(self, name: str, res: dict, ctm: list[float], depth: int) -> None:
        xo = self.doc.resolve(res.get("XObject")) or {}
        ref = xo.get(name) if isinstance(xo, dict) else None
        d = self.doc.resolve(ref)
        if not isinstance(d, dict) or d.get("Subtype") != "Form":
            return
        m = self.doc.resolve(d.get("Matrix")) or [1, 0, 0, 1, 0, 0]
        m = [float(self.doc.resolve(v)) for v in m][:6]
        sub_res = self.doc.resolve(d.get("Resources")) or res
        self.depth += 1
        try:
            self.run(self.doc.stream(ref), sub_res, _mul(m, ctm), depth + 1)
        finally:
            self.depth -= 1


# ====================================================================================== entry point
def extract(src: str | Path | bytes, include_forms: bool = True) -> list[Page]:
    """All pages of a PDF with their text spans and vector marks (see module docstring)."""
    data = src if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
    doc = _Doc(bytes(data))
    out: list[Page] = []
    for i, pg in enumerate(doc.pages()):
        mb = [float(doc.resolve(v)) for v in (doc.resolve(pg.get("MediaBox")) or [0, 0, 595.28, 841.89])][:4]
        page = Page(i, mb[2] - mb[0], mb[3] - mb[1])
        interp = _Interp(doc, page)
        res = doc.resolve(pg.get("Resources")) or {}
        if not include_forms:
            res = dict(res)
            res.pop("XObject", None)
        contents = doc.resolve(pg.get("Contents"))
        refs = contents if isinstance(contents, list) else [pg.get("Contents")]
        data_all = b"\n".join(doc.stream(r) for r in refs if isinstance(r, _Ref))
        interp.run(data_all, res, [1.0, 0, 0, 1.0, -mb[0], -mb[1]])
        out.append(page)
    if not out:
        raise PdfError("the PDF has no pages")
    return out
