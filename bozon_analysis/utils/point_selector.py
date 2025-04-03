import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class PointSelector(QDialog):
    """Popup for selecting a single point on an image."""
    def __init__(self, image, parent=None, title = "Select a Point"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.image = image
        self.temp_point = None
        self.selected_point = None  # Store the selected point

        # === Matplotlib Figure ===
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        # === Display Image ===
        self.image = image
        self.ax.imshow(self.image, cmap='gray')

        # === UI Buttons ===
        self.confirm_button = QPushButton("Confirm")
        self.cancel_button = QPushButton("Cancel")

        self.confirm_button.clicked.connect(self.confirm_selection)
        self.cancel_button.clicked.connect(self.reject)  # Close on cancel

        # === Layout ===
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.confirm_button)
        layout.addWidget(self.cancel_button)
        self.setLayout(layout)

        # Connect mouse click event
        self.canvas.mpl_connect('button_press_event', self.on_click)
        

    def on_click(self, event):
        """Handles user clicking on the image to select a point."""
        if event.inaxes is None:
            return  # Ignore clicks outside the image

        x, y = int(event.xdata), int(event.ydata)
        if x is None or y is None:
            return  # Ignore invalid clicks

        self.temp_point = (x, y)

        # Clear previous markers and add a new one
        for artist in self.ax.lines:  
            artist.remove()  

        self.ax.plot(x, y, 'ro', markersize=5)
        self.canvas.draw_idle()

    def confirm_selection(self):
        """Return the selected point and close the popup."""
        self.selected_point = self.temp_point
        self.accept()  # Close the popup