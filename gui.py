from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QTextEdit, QPushButton, 
    QInputDialog, QSplitter, QHBoxLayout, QLabel, QFileDialog
)
from PyQt5.QtCore import Qt

import logging

from data_handler import DataHandler
from image_processor import ImageProcessor
from socket_handler import SocketHandler
from analysis_handler import AnalysisHandler
from gui_logger import QTextEditLogger

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MainWindow(QMainWindow):
    def __init__(self):
        ## initialize window
        super().__init__()
        self.setWindowTitle("Data Analyzer")
        self.setGeometry(100, 100, 800, 800)

        self.image_process_figure = Figure(figsize = (6,6))

        self.data_handler = DataHandler()
        self.image_processor = ImageProcessor(self.data_handler, self.image_process_figure)
        self.analysis_handler = AnalysisHandler(self.data_handler)
        self.socket_handler = SocketHandler(self.data_handler, self.image_processor, self.analysis_handler)

        self.init_ui()

        self.socket_handler.start()

    def init_ui(self):
        ## logging display
        self.log_display = QTextEdit(self)
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("background-color: black; color: white;")
        self.log_handler = QTextEditLogger(self.log_display)
        self.log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(self.log_handler)

        ## figure display
        self.image_process_canvas = FigureCanvas(self.image_process_figure)
        self.image_process_canvas.show()

        ## data handler buttons
        self.set_date_button = QPushButton("Set date", self)
        self.set_date_button.clicked.connect(self.set_date_button_press)
        self.date_label = QLabel(self.data_handler.date.strftime('%y%m%d'))
        self.data_handler.date_updated.connect(self.update_date_label)
        self.date_button_layout = QVBoxLayout()
        self.date_button_layout.addWidget(self.date_label)
        self.date_button_layout.addWidget(self.set_date_button)
        self.date_button_container = QWidget()
        self.date_button_container.setLayout(self.date_button_layout)
        self.set_file_button = QPushButton("Set file", self)
        self.set_file_button.clicked.connect(self.set_file_button_press)
        self.file_label = QLabel(str(self.data_handler.file))
        self.data_handler.file_updated.connect(self.update_file_label)
        self.file_button_layout = QVBoxLayout()
        self.file_button_layout.addWidget(self.file_label)
        self.file_button_layout.addWidget(self.set_file_button)
        self.file_button_container = QWidget()
        self.file_button_container.setLayout(self.file_button_layout)
        self.process_image_button = QPushButton("Only process images", self)
        self.process_image_button.clicked.connect(self.process_image_button_press)
        self.run_analysis_button = QPushButton("Run analysis", self)
        self.run_analysis_button.clicked.connect(self.run_analysis_button_press)
        self.analysis_button_layout = QVBoxLayout()
        self.analysis_button_layout.addWidget(self.process_image_button)
        self.analysis_button_layout.addWidget(self.run_analysis_button)
        self.analysis_button_container = QWidget()
        self.analysis_button_container.setLayout(self.analysis_button_layout)

        ## socket handler buttons
        self.start_stop_socket_button = QPushButton("Stop socket", self)
        self.start_stop_socket_button.clicked.connect(self.start_stop_socket_button_press)
        self.socket_status_label = QLabel('Not connected')
        self.socket_status_label.setStyleSheet("color: red;")
        self.socket_handler.socket_status.connect(self.update_socket_status_label)
        self.socket_button_layout = QVBoxLayout()
        self.socket_button_layout.addWidget(self.socket_status_label)
        self.socket_button_layout.addWidget(self.start_stop_socket_button)
        self.socket_button_container = QWidget()
        self.socket_button_container.setLayout(self.socket_button_layout)

        ## analysis handler buttons
        self.set_analysis_script_button = QPushButton("Set analysis script", self)
        self.set_analysis_script_button.clicked.connect(self.set_analysis_script_button_press)
        self.analysis_script_label = QLabel('None')
        self.analysis_handler.module_updated.connect(self.update_analysis_script_label)
        self.set_analysis_script_layout = QVBoxLayout()
        self.set_analysis_script_layout.addWidget(self.analysis_script_label)
        self.set_analysis_script_layout.addWidget(self.set_analysis_script_button)
        self.set_analysis_script_container = QWidget()
        self.set_analysis_script_container.setLayout(self.set_analysis_script_layout)

        ## image processor buttons
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

        #### Set up main window ####
        self.data_handler_button_layout = QHBoxLayout()
        self.data_handler_button_layout.addWidget(self.date_button_container)
        self.data_handler_button_layout.addWidget(self.file_button_container)
        self.data_handler_button_layout.addWidget(self.socket_button_container)
        self.data_handler_button_layout.addWidget(self.set_analysis_script_container)
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
            except Exception as e:
                logging.error(f"Error setting date: {e}")

    def update_date_label(self):
        self.date_label.setText(self.data_handler.date.strftime('%y%m%d'))

    def set_file_button_press(self):
        """Prompt the user for a file number."""
        text, ok = QInputDialog.getText(self, "Input Dialog", "Enter file number:")
        if ok and text:  # If user presses OK and entered text
            try:
                self.data_handler.file = text
            except Exception as e:
                logging.error(f"Error setting file number: {e}")

    def update_file_label(self):
        self.file_label.setText(str(self.data_handler.file))

    #### SOCKET HANDLER BUTTONS ####
    def start_stop_socket_button_press(self):
        if self.socket_handler.running:
            self.socket_handler.stop()
            self.start_stop_socket_button.setText("Start socket")
        else:
            self.socket_handler.start()
            self.start_stop_socket_button.setText("Stop socket")

    def update_socket_status_label(self):
        if self.socket_handler.connected:
            self.socket_status_label.setStyleSheet("color: green;")
            self.socket_status_label.setText("Connected")
        elif self.socket_handler.running and not self.socket_handler.connected:
            self.socket_status_label.setStyleSheet("color: red;")
            self.socket_status_label.setText("Trying to connect")
        else:
            self.socket_status_label.setStyleSheet("color: red;")
            self.socket_status_label.setText("Not running")

    #### ANALYSIS HANDLER BUTTONS ####
    def set_analysis_script_button_press(self):
        script_name, _ = QFileDialog.getOpenFileName(None, "Open File", "analysis_scripts", "Python Scripts (*.py)")
        if script_name:
            self.analysis_handler.module_name = script_name
    
    def update_analysis_script_label(self,):
        if self.analysis_handler.module_name is None:
            self.analysis_script_label.setText('None')
        else:
            self.analysis_script_label.setText(self.analysis_handler.module_name)

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
        try:
            self.image_processor.select_crop_region(self)
        except Exception as e:
            logging.error(f"Error selecting crop region: {e}")
    
    def set_offset_button_press(self):
        try:
            self.image_processor.select_offset(self)
        except Exception as e:
            logging.error(f"Error selecting offset: {e}")
    
    def process_image_button_press(self):
        try:
            self.image_processor.process_images()
        except FileNotFoundError as e:
            logging.error(e)
        except Exception as e:
            logging.error(f"Error running analysis: {e}")

    
    
    def run_analysis_button_press(self):
        """Run the analysis and update the GUI with results and plots."""
        try:
            self.image_processor.process_images()
            self.analysis_handler.run_analysis_script()
        except FileNotFoundError as e:
            logging.error(e)
        except Exception as e:
            logging.error(f"Error running analysis: {e}")

    