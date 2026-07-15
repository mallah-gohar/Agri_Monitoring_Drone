import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

import config


class GridCanvas(FigureCanvasQTAgg):
    cell_clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(6, 6), facecolor="#FAFAFA")
        super().__init__(self.figure)
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111)
        self._rgb_grid = None
        self._path_x: list[int] = []
        self._path_y: list[int] = []
        self._drone_pos = None
        self._connect_events()
        self.draw_empty()

    def _connect_events(self):
        self.mpl_connect("button_press_event", self._on_click)

    def _state_to_rgb(self, grid):
        rgb = np.zeros((config.GRID_ROWS, config.GRID_COLS, 3))
        for r in range(config.GRID_ROWS):
            for c in range(config.GRID_COLS):
                rgb[r, c] = config.COLOR_RGB.get(int(grid[r, c]), config.COLOR_RGB[config.CELL_UNSCANNED])
        return rgb

    def draw_empty(self):
        grid = np.full((config.GRID_ROWS, config.GRID_COLS), config.CELL_UNSCANNED)
        self.update_grid(grid)

    def update_grid(self, scanned_grid, path=None, drone_pos=None):
        self._rgb_grid = self._state_to_rgb(scanned_grid)
        self.ax.clear()
        self.ax.imshow(self._rgb_grid, origin="upper")
        self.ax.set_xticks(range(0, config.GRID_COLS, 5))
        self.ax.set_yticks(range(0, config.GRID_ROWS, 5))
        self.ax.set_xlabel("Column")
        self.ax.set_ylabel("Row")
        self.ax.set_title("Drone Field Scan")

        if path:
            self._path_x = [p[1] for p in path]
            self._path_y = [p[0] for p in path]
            self.ax.plot(self._path_x, self._path_y, color=config.COLOR_PATH_RGB, linewidth=1.5, alpha=0.8)

        if drone_pos:
            self._drone_pos = drone_pos
            self.ax.scatter(
                [drone_pos[1]],
                [drone_pos[0]],
                s=120,
                c="white",
                edgecolors="black",
                linewidths=1.5,
                zorder=5,
            )

        self.ax.set_xlim(-0.5, config.GRID_COLS - 0.5)
        self.ax.set_ylim(config.GRID_ROWS - 0.5, -0.5)
        self.figure.tight_layout()
        self.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if 0 <= row < config.GRID_ROWS and 0 <= col < config.GRID_COLS:
            self.cell_clicked.emit(row, col)


class GridCanvasWidget(QWidget):
    cell_clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = GridCanvas(self)
        self.canvas.cell_clicked.connect(self.cell_clicked.emit)
        layout.addWidget(self.canvas)

    def update_grid(self, scanned_grid, path=None, drone_pos=None):
        self.canvas.update_grid(scanned_grid, path, drone_pos)

    def draw_empty(self):
        self.canvas.draw_empty()
