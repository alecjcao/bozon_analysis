from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtGui import QTextCharFormat, QColor
import logging

class QTextEditLogger(logging.Handler):
    """Custom logging handler that writes logs to a QTextEdit widget."""

    COLORS = {
        logging.DEBUG: QColor("gray"),
        logging.INFO: QColor("green"),
        logging.WARNING: QColor("orange"),
        logging.ERROR: QColor("red"),
        logging.CRITICAL: QColor("darkred"),
    }

    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.widget.setReadOnly(True)  # Make the log display read-only

    def emit(self, record):
        msg = self.format(record)  # Format the log message
        color = self.COLORS.get(record.levelno, QColor("black"))
        text_format = QTextCharFormat()
        text_format.setForeground(color)
        self.widget.moveCursor(self.widget.textCursor().End)
        self.widget.setCurrentCharFormat(text_format)
        self.widget.append(msg)  # Append to the QTextEdit widget
