from matplotlib.patches import Rectangle
from image_processing.point_selector import PointSelector

CROP_SIZE = (210, 200)

class CropSelector(PointSelector):
    """Popup window for selecting crop region"""
    def __init__(self, image, parent=None, title = "Select a Crop Region"):
        super().__init__(image, parent, title)

        # === Interaction Variables ===
        self.rect = None

    def on_click(self, event):
        """User clicks to set the crop region"""
        if event.inaxes is None:
            return

        x, y = int(event.xdata), int(event.ydata)
        if x is None or y is None:  # Prevent invalid coordinates
            return
        
        self.temp_point = (x, y)

        # Remove previous rectangle
        if self.rect:
            self.rect.remove()

        # Draw new rectangle
        self.rect = Rectangle((x, y), CROP_SIZE[1], CROP_SIZE[0], edgecolor='red', facecolor='none', lw=2)
        self.ax.add_patch(self.rect)
        self.canvas.draw()