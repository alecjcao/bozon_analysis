from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtGui import QTextCharFormat, QColor, QTextCursor
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import logging

MAX_LINES = 1000

class QTextEditLogger(logging.Handler, QObject):
    appendText = pyqtSignal(str, int)

    COLORS = {
        logging.DEBUG: QColor("gray"),
        logging.INFO: QColor("green"),
        logging.WARNING: QColor("orange"),
        logging.ERROR: QColor("red"),
        logging.CRITICAL: QColor("darkred"),
    }

    def __init__(self, parent):
        super().__init__()
        QObject.__init__(self)
        self.widget = QTextEdit(parent)
        self.widget.setReadOnly(True)
        self.appendText.connect(self.append_message)

    def emit(self, record):
        msg = self.format(record)
        level = record.levelno
        self.appendText.emit(msg, level)

    @pyqtSlot(str, int)
    def append_message(self, message, level):
        color = self.COLORS.get(level, QColor("black"))
        text_format = QTextCharFormat()
        text_format.setForeground(color)
        self.widget.setCurrentCharFormat(text_format)
        self.widget.append(message)

        if self.widget.document().blockCount() > MAX_LINES:
            cursor = self.widget.textCursor()
            cursor.movePosition(QTextCursor.Start)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

        self.widget.moveCursor(self.widget.textCursor().End)