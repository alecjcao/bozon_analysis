import logging

class QTextEditLogger(logging.Handler):
    """Custom logging handler that writes logs to a QTextEdit widget."""
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.widget.setReadOnly(True)  # Make the log display read-only

    def emit(self, record):
        msg = self.format(record)  # Format the log message
        self.widget.append(msg)  # Append to the QTextEdit widget
