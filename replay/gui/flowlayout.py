"""A layout that places widgets left to right and wraps to the next line when the row is full (Qt's flow layout
example). Used for long rows of options so the window can be narrow (laptops, ~1280 px)."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QSizePolicy, QWidget


class FlowLayout(QLayout):
    def __init__(self, parent: QWidget | None = None, h_spacing: int = 6, v_spacing: int = 2):
        super().__init__(parent)
        self._items: list = []
        self._h, self._v = h_spacing, v_spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:  # noqa: N802 - Qt API
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i: int):  # noqa: N802
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i: int):  # noqa: N802
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for it in self._items:
            size = size.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect: QRect, test: bool) -> int:
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = r.x(), r.y(), 0
        for it in self._items:
            w = it.widget()
            if w is not None and w.isHidden():
                continue
            hint = it.sizeHint()
            nx = x + hint.width() + self._h
            if nx - self._h > r.right() and line_h > 0:
                x, y = r.x(), y + line_h + self._v
                nx, line_h = x + hint.width() + self._h, 0
            if not test:
                it.setGeometry(QRect(QPoint(x, y), hint))
            x, line_h = nx, max(line_h, hint.height())
        return y + line_h - rect.y() + m.bottom()


def flow_container(widgets: list[QWidget]) -> QWidget:
    """A widget holding ``widgets`` in a FlowLayout (height follows the width)."""
    w = QWidget()
    lay = FlowLayout(w)
    for x in widgets:
        lay.addWidget(x)
    sp = w.sizePolicy()
    sp.setHeightForWidth(True)
    sp.setHorizontalPolicy(QSizePolicy.Policy.Preferred)
    w.setSizePolicy(sp)
    return w
