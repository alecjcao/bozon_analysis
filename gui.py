from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QTextEdit, QPushButton, QInputDialog, QSplitter, QHBoxLayout, QLabel
)
from PyQt5.QtCore import Qt

import logging

from data_handler import DataHandler
from plot_handler import PlotCanvas
from gui_logger import QTextEditLogger
from image_processing.image_processer import ImageProcesser

class MainWindow(QMainWindow):
    def __init__(self):
        ## initialize window
        super().__init__()
        self.setWindowTitle("Data Analyzer")
        self.setGeometry(100, 100, 800, 600)

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
        self.analysis_button_layout = QVBoxLayout()
        self.analysis_button_layout.addWidget(self.analysis_button_label)
        self.analysis_button_layout.addWidget(self.run_analysis_button)
        self.analysis_button_container = QWidget()
        self.analysis_button_container.setLayout(self.analysis_button_layout)

        #### set up image processer ####
        self.image_processer = ImageProcesser()

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
        self.canvas = PlotCanvas(self)  # Placeholder for Matplotlib canvas

        #### Set up main window ####
        data_handler_button_layout = QHBoxLayout()
        data_handler_button_layout.addWidget(self.date_button_container)
        data_handler_button_layout.addWidget(self.file_button_container)
        data_handler_button_layout.addWidget(self.analysis_button_container)
        data_handler_button_container = QWidget()  # Wrap in a QWidget to insert into vertical layout
        data_handler_button_container.setLayout(data_handler_button_layout) 

        image_processor_button_layout = QHBoxLayout()
        image_processor_button_layout.addWidget(self.enable_disable_crop_button)
        image_processor_button_layout.addWidget(self.enable_disable_convolution_button)
        image_processor_button_layout.addWidget(self.enable_all_sites_button)
        image_processor_button_layout.addWidget(self.set_crop_button)
        image_processor_button_layout.addWidget(self.set_offset_button)
        image_processor_button_container = QWidget()  # Wrap in a QWidget to insert into vertical layout
        image_processor_button_container.setLayout(image_processor_button_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.log_display)
        splitter.setStretchFactor(0, 1) 
        splitter.setStretchFactor(1, 1)

        main_layout = QVBoxLayout()
        main_layout.addWidget(data_handler_button_container)
        main_layout.addWidget(image_processor_button_container)
        main_layout.addWidget(splitter)

        self.central_widget = QWidget()
        self.central_widget.setLayout(main_layout)
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
        self.image_processer.crop_enabled = not self.image_processer.crop_enabled
        if self.image_processer.crop_enabled:
            self.enable_disable_crop_button.setText("Disable crop")
        else:
            self.enable_disable_crop_button.setText("Enable crop")

    def enable_disable_convolution_button_press(self):
        self.image_processer.convolution_enabled = not self.image_processer.convolution_enabled
        if self.image_processer.convolution_enabled:
            self.enable_disable_convolution_button.setText("Disable convolution")
        else:
            self.enable_disable_convolution_button.setText("Enable convolution")

    def enable_all_sites_button_press(self):
        self.image_processer.all_sites_enabled = not self.image_processer.all_sites_enabled
        if self.image_processer.all_sites_enabled:
            self.enable_all_sites_button.setText("Disable all sites")
        else:
            self.enable_all_sites_button.setText("Enable all sites")


    def set_crop_button_press(self):
        data = self.data_handler.load_data()
        try:
            self.image_processer.select_crop_region(data, self)
        except Exception as e:
            logging.error(f"Error selecting crop region: {e}")
    
    def set_offset_button_press(self):
        data = self.data_handler.load_data()
        try:
            self.image_processer.select_offset(data, self)
        except Exception as e:
            logging.error(f"Error selecting offset: {e}")
        return
    
    

    def run_analysis_button_press(self):
        """Run the analysis and update the GUI with results and plots."""
        try:
            self.update_analysis()
        except Exception as e:
            logging.error(f"Error running analysis: {e}")

    def update_analysis(self):
        return
        # data = self.data_handler.load_data()
        # try:
        #     self.image_processer.select_crop_region(data, self)
        # except Exception as e:
        #     logging.error(f"Error selecting crop region: {e}")


    # def update_analysis(self, file_path):
    #     """Runs the analysis and updates the GUI with results and plots."""
    #     analysis_instance = process_new_data(file_path)

    #     # Update text display with results
    #     results = analysis_instance.run_analysis()
    #     self.result_display.setText(str(results))

    #     # Generate and display the plot
    #     fig = analysis_instance.generate_plot()
    #     self.display_plot(fig)

    # def display_plot(self, fig):
    #     """Embed the Matplotlib figure into the GUI."""
    #     if self.canvas:
    #         self.central_widget.layout().removeWidget(self.canvas)
    #         self.canvas.deleteLater()

    #     self.canvas = FigureCanvas(fig)
    #     self.central_widget.layout().addWidget(self.canvas)