from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QTextEdit, QPushButton, QInputDialog, QSplitter, QHBoxLayout, QLabel
)
from PyQt5.QtCore import Qt

import logging

from data_handler import DataHandler
from gui_logger import QTextEditLogger
from image_processing.image_processor import ImageProcessor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MainWindow(QMainWindow):
    def __init__(self):
        ## initialize window
        super().__init__()
        self.setWindowTitle("Data Analyzer")
        self.setGeometry(100, 100, 800, 800)

        #### set up logging display ####
        # Create a widget for displaying logs
        self.log_display = QTextEdit(self)
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("background-color: black; color: white;")
        #   Set up logging to the QTextEdit widget
        log_handler = QTextEditLogger(self.log_display)
        log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(log_handler)

        #### set up data handler ####
        self.data_handler = DataHandler()

        self.set_date_button = QPushButton("Set date", self)
        self.date_button_label = QLabel('Current: ' + self.data_handler.date.strftime('%y%m%d'))
        self.set_date_button.clicked.connect(self.set_date_button_press)
        self.date_button_layout = QVBoxLayout()
        self.date_button_layout.addWidget(self.date_button_label)
        self.date_button_layout.addWidget(self.set_date_button)
        self.date_button_container = QWidget()
        self.date_button_container.setLayout(self.date_button_layout)

        self.set_file_button = QPushButton("Set file", self)
        self.file_button_label = QLabel('Current: ' + str(self.data_handler.file))
        self.set_file_button.clicked.connect(self.set_file_button_press)
        self.file_button_layout = QVBoxLayout()
        self.file_button_layout.addWidget(self.file_button_label)
        self.file_button_layout.addWidget(self.set_file_button)
        self.file_button_container = QWidget()
        self.file_button_container.setLayout(self.file_button_layout)

        self.run_analysis_button = QPushButton("Run analysis", self)
        self.analysis_button_label = QLabel('')
        self.run_analysis_button.clicked.connect(self.run_analysis_button_press)
        self.analysis_button_layout = QVBoxLayout()
        self.analysis_button_layout.addWidget(self.analysis_button_label)
        self.analysis_button_layout.addWidget(self.run_analysis_button)
        self.analysis_button_container = QWidget()
        self.analysis_button_container.setLayout(self.analysis_button_layout)

        #### set up image processor ####
        self.image_processor = ImageProcessor()

        self.enable_disable_crop_button = QPushButton("Disable crop", self)
        self.enable_disable_crop_button.clicked.connect(self.enable_disable_crop_button_press)

        self.enable_disable_convolution_button = QPushButton("Enable convolution", self)
        self.enable_disable_convolution_button.clicked.connect(self.enable_disable_convolution_button_press)

        self.enable_all_sites_button = QPushButton("Enable all sites", self)
        self.enable_all_sites_button.clicked.connect(self.enable_all_sites_button_press)

        self.set_crop_button = QPushButton("Set crop", self)
        self.set_crop_button.clicked.connect(self.set_crop_button_press)

        self.set_offset_button = QPushButton("Set offset", self)
        self.set_offset_button.clicked.connect(self.set_offset_button_press)

        #### set up plot canvas ####
        self.image_process_figure = Figure(figsize = (6,6))
        self.image_process_canvas = FigureCanvas(self.image_process_figure)
        self.image_process_canvas.show()

        #### Set up main window ####
        self.data_handler_button_layout = QHBoxLayout()
        self.data_handler_button_layout.addWidget(self.date_button_container)
        self.data_handler_button_layout.addWidget(self.file_button_container)
        self.data_handler_button_layout.addWidget(self.analysis_button_container)
        self.data_handler_button_container = QWidget()  # Wrap in a QWidget to insert into vertical layout
        self.data_handler_button_container.setLayout(self.data_handler_button_layout) 

        self.image_processor_button_layout = QHBoxLayout()
        self.image_processor_button_layout.addWidget(self.enable_disable_crop_button)
        self.image_processor_button_layout.addWidget(self.enable_disable_convolution_button)
        self.image_processor_button_layout.addWidget(self.enable_all_sites_button)
        self.image_processor_button_layout.addWidget(self.set_crop_button)
        self.image_processor_button_layout.addWidget(self.set_offset_button)
        self.image_processor_button_container = QWidget()  # Wrap in a QWidget to insert into vertical layout
        self.image_processor_button_container.setLayout(self.image_processor_button_layout)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.image_process_canvas)
        self.splitter.addWidget(self.log_display)
        self.splitter.setStretchFactor(0, 1) 
        self.splitter.setStretchFactor(1, 1)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.data_handler_button_container)
        self.main_layout.addWidget(self.image_processor_button_container)
        self.main_layout.addWidget(self.splitter)

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)


    #### DATA HANDLER BUTTONS ####

    def set_date_button_press(self):
        """Prompt the user for a date."""
        text, ok = QInputDialog.getText(self, "Input Dialog", "Enter date:")
        if ok and text:  # If user presses OK and entered text
            try:
                self.data_handler.date = text
                self.date_button_label.setText('Current: ' + self.data_handler.date.strftime('%y%m%d'))
            except Exception as e:
                logging.error(f"Error setting date: {e}")

    def set_file_button_press(self):
        """Prompt the user for a file number."""
        text, ok = QInputDialog.getText(self, "Input Dialog", "Enter file number:")
        if ok and text:  # If user presses OK and entered text
            try:
                self.data_handler.file = text
                self.file_button_label.setText('Current: ' + str(self.data_handler.file))
            except Exception as e:
                logging.error(f"Error setting file number: {e}")


    #### IMAGE PROCESSOR BUTTONS ####

    def enable_disable_crop_button_press(self):
        self.image_processor.crop_enabled = not self.image_processor.crop_enabled
        if self.image_processor.crop_enabled:
            self.enable_disable_crop_button.setText("Disable crop")
        else:
            self.enable_disable_crop_button.setText("Enable crop")

    def enable_disable_convolution_button_press(self):
        self.image_processor.convolution_enabled = not self.image_processor.convolution_enabled
        if self.image_processor.convolution_enabled:
            self.enable_disable_convolution_button.setText("Disable convolution")
        else:
            self.enable_disable_convolution_button.setText("Enable convolution")

    def enable_all_sites_button_press(self):
        self.image_processor.all_sites_enabled = not self.image_processor.all_sites_enabled
        if self.image_processor.all_sites_enabled:
            self.enable_all_sites_button.setText("Disable all sites")
        else:
            self.enable_all_sites_button.setText("Enable all sites")


    def set_crop_button_press(self):
        data = self.data_handler.get_data()
        try:
            self.image_processor.select_crop_region(data, self)
        except Exception as e:
            logging.error(f"Error selecting crop region: {e}")
    
    def set_offset_button_press(self):
        data = self.data_handler.get_data()
        try:
            self.image_processor.select_offset(data, self)
        except Exception as e:
            logging.error(f"Error selecting offset: {e}")
        return
    
    
    def run_analysis_button_press(self):
        """Run the analysis and update the GUI with results and plots."""
        try:
            data = self.data_handler.get_data()
        except FileNotFoundError as e:
            logging.error(e)
            return
        try:
            self.image_processor.process_images(data, self.image_process_figure)
            self.image_process_canvas.show()
        except Exception as e:
            logging.error(f"Error running analysis: {e}")

    