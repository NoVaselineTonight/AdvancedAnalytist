"""A small round "?" button that holds the explanation for the control or line next to it.

Used to keep the screens calm: long hints, how-to text and secondary details live behind a ``HelpIcon``; the
main numbers and controls stay visible. Nothing is thrown away - the text is shown

* when you hover the icon (tooltip, so the existing ``tooltips.missing()`` check counts it as described),
* when you click it, or press Space / Enter / F1 while it has the keyboard focus (a small popover that stays
  open until you click elsewhere or press Esc; it can contain links, which are passed on as ``linkActivated``).

The icon is themed from the palette (works in light and dark), focusable, and has an accessible name
("Help: <topic>") and description (the plain text), so screen readers read it.
"""

from __future__ import annotations

import re
from html import unescape
from typing import Iterable

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QKeyEvent
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QToolButton, QToolTip, QVBoxLayout, QWidget)

ICON_SIZE = 18
POPUP_WIDTH = 360  # px, the popover wraps its text at this width
_TAG = re.compile(r"<[^>]+>")


def plain(text: str) -> str:
    """The text without HTML tags (line breaks kept as spaces) - used for accessibility and the tests."""
    t = re.sub(r"<\s*(br|/p|/li|/tr|/h\d)\s*/?>", " ", text or "", flags=re.I)
    return re.sub(r"\s+", " ", unescape(_TAG.sub("", t))).strip()


class _Popup(QFrame):
    """The popover: a framed, word-wrapped label that closes on a click outside or Esc."""

    linkActivated = Signal(str)

    def __init__(self, owner: "HelpIcon"):
        super().__init__(owner.window(), Qt.WindowType.Popup)
        self.setObjectName("helpPopup")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAutoFillBackground(True)
        self.setStyleSheet("QFrame#helpPopup { background: palette(base); border: 1px solid palette(mid); "
                           "border-radius: 6px; } QLabel { background: transparent; }")
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 8)
        self.label = QLabel()
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setWordWrap(True)
        self.label.setOpenExternalLinks(False)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.label.linkActivated.connect(self.linkActivated)
        v.addWidget(self.label)

    def set_text(self, html: str) -> None:
        self.label.setText(html)
        self.label.setFixedWidth(POPUP_WIDTH)
        self.adjustSize()


class HelpIcon(QToolButton):
    """Round "?" button. ``HelpIcon("What this does", topic="Smoothing")``."""

    linkActivated = Signal(str)  # a link inside the help text was clicked
    opened = Signal()

    def __init__(self, text: str = "", topic: str = "", parent: QWidget | None = None, *,
                 hide_when_empty: bool = True):
        super().__init__(parent)
        self._text = ""
        self._topic = topic
        self._hide_when_empty = hide_when_empty
        self._popup: _Popup | None = None
        self._auto_hidden = False
        self.setObjectName("helpIcon")
        self.setText("?")
        self.setFixedSize(ICON_SIZE, ICON_SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAutoRaise(False)
        self.setStyleSheet(
            "QToolButton#helpIcon { border: 1px solid palette(mid); border-radius: %dpx; background: palette(button);"
            " color: palette(button-text); font-weight: bold; font-size: 11px; padding: 0px; }"
            "QToolButton#helpIcon:hover, QToolButton#helpIcon:focus, QToolButton#helpIcon:checked "
            "{ background: palette(highlight); color: palette(highlighted-text); border-color: palette(highlight); }"
            % (ICON_SIZE // 2))
        self.clicked.connect(self.toggle_popup)
        self.set_help(text)

    # ------------------------------------------------------------ content
    def set_help(self, text: str, topic: str | None = None) -> None:
        """Replace the help text (for lines that change at run time). Empty text hides the icon."""
        self._text = text or ""
        if topic is not None:
            self._topic = topic
        name = f"Help: {self._topic}" if self._topic else "Help"
        self.setAccessibleName(name)
        self.setAccessibleDescription(plain(self._text))
        self.setToolTip(self._tooltip_html())
        if self._popup is not None and self._popup.isVisible():
            self._popup.set_text(self._text)
        if self._hide_when_empty:
            if not self._text:
                self.hide()
                self._auto_hidden = True
            elif self._auto_hidden and self.parentWidget() is not None:
                self._auto_hidden = False
                self.show()

    def help_text(self) -> str:
        return self._text

    def plain_text(self) -> str:
        return plain(self._text)

    @property
    def topic(self) -> str:
        return self._topic

    def _tooltip_html(self) -> str:
        if not self._text:
            return ""
        if len(plain(self._text)) > 70:  # wrap long text (Qt only wraps rich text tooltips that have a width)
            return f"<table width='{POPUP_WIDTH - 40}'><tr><td>{self._text}</td></tr></table>"
        return f"<div>{self._text}</div>"

    # ------------------------------------------------------------ popover
    def is_open(self) -> bool:
        return self._popup is not None and self._popup.isVisible()

    def popup(self) -> _Popup:
        if self._popup is None:
            self._popup = _Popup(self)
            self._popup.linkActivated.connect(self.linkActivated)
        return self._popup

    def show_popup(self) -> None:
        if not self._text:
            return
        QToolTip.hideText()
        pop = self.popup()
        pop.set_text(self._text)
        anchor = self.mapToGlobal(QPoint(0, self.height() + 4))
        scr = (self.screen() or QGuiApplication.primaryScreen())
        if scr is not None:
            g = scr.availableGeometry()
            x = min(max(g.left() + 4, anchor.x()), max(g.left() + 4, g.right() - pop.width() - 4))
            y = anchor.y()
            if y + pop.height() > g.bottom():  # no room below: open above the icon
                y = max(g.top() + 4, self.mapToGlobal(QPoint(0, 0)).y() - pop.height() - 4)
            anchor = QPoint(x, y)
        pop.move(anchor)
        pop.show()
        self.opened.emit()

    def hide_popup(self) -> None:
        if self._popup is not None:
            self._popup.hide()

    def toggle_popup(self, *_a) -> None:
        if self.is_open():
            self.hide_popup()
        else:
            self.show_popup()

    # ------------------------------------------------------------ keyboard
    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() in (Qt.Key.Key_F1, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.toggle_popup()
            e.accept()
            return
        if e.key() == Qt.Key.Key_Escape and self.is_open():
            self.hide_popup()
            e.accept()
            return
        super().keyPressEvent(e)

    def hideEvent(self, e) -> None:  # noqa: N802
        self.hide_popup()
        super().hideEvent(e)


# ---------------------------------------------------------------- layout helpers
def help_row(widget: QWidget, text: str, topic: str = "", *, stretch: bool = True) -> QWidget:
    """``widget`` followed by its "?" in one row. The icon is available as ``row.help_icon``."""
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(4)
    h.addWidget(widget, 1 if stretch else 0)
    icon = HelpIcon(text, topic)
    h.addWidget(icon, 0, Qt.AlignmentFlag.AlignVCenter)
    row.help_icon = icon  # type: ignore[attr-defined]
    return row


def titled(title_html: str, text: str, topic: str = "") -> QWidget:
    """A bold section title with its "?" right after it (``row.help_icon``)."""
    lbl = QLabel(title_html)
    row = help_row(lbl, text, topic or plain(title_html), stretch=False)
    row.layout().addStretch(1)  # type: ignore[union-attr]
    return row


def add_icon(layout, text: str, topic: str = "") -> HelpIcon:
    """Append a "?" to an existing box/flow layout."""
    icon = HelpIcon(text, topic)
    layout.addWidget(icon)
    return icon


# ---------------------------------------------------------------- inspection (tests, docs)
def help_icons(root: QWidget) -> list[HelpIcon]:
    return root.findChildren(HelpIcon)


def help_texts(root: QWidget) -> list[tuple[str, str]]:
    """(topic, plain text) of every help icon under ``root`` (also the hidden/empty ones)."""
    return [(i.topic, i.plain_text()) for i in help_icons(root)]


def help_contains(root: QWidget, *needles: Iterable[str] | str) -> bool:
    """True when every needle (case-insensitive) is in the text of some help icon under ``root``."""
    blob = " ".join(t for _k, t in help_texts(root)).lower()
    return all(str(n).lower() in blob for n in needles)
