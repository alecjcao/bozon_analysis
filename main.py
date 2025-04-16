from PyQt5.QtWidgets import QApplication
from bozon_analysis.gui import MainWindow
import sys
import logging

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()  # Logs to console
        ]
        # logging.FileHandler("app.log"),  # Logs to file
    )
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())