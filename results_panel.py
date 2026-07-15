import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

import config


class ResultsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = None
        self._csv_bytes = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.summary = QLabel("Run a simulation to see results.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        charts = QHBoxLayout()
        self.pie_canvas = self._make_canvas()
        self.line_canvas = self._make_canvas()
        charts.addWidget(self._wrap_canvas("Disease Distribution", self.pie_canvas))
        charts.addWidget(self._wrap_canvas("NDVI Signal During Scan", self.line_canvas))
        layout.addLayout(charts)

        self.export_btn = QPushButton("Export Scan CSV")
        self.export_btn.clicked.connect(self._export_csv)
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)

    def _make_canvas(self):
        fig = Figure(figsize=(4, 3), facecolor="#FAFAFA")
        return FigureCanvasQTAgg(fig)

    def _wrap_canvas(self, title, canvas):
        box = QVBoxLayout()
        box.addWidget(QLabel(title))
        box.addWidget(canvas)
        wrapper = QWidget()
        wrapper.setLayout(box)
        return wrapper

    def update_results(self, result):
        self._result = result
        counts = result.counts
        scanned = len(result.predictions)
        self.summary.setText(
            f"{result.field_name} — {result.crop_type} | "
            f"{scanned} cells scanned in {result.scan_duration_sec:.2f}s | "
            f"Path: {result.path_algo} | Classifier: {result.classifier_type}"
        )

        df = pd.DataFrame({
            "Row": [p[0] for p in result.path],
            "Col": [p[1] for p in result.path],
            "NDVI": result.ndvi_values,
            "Prediction": [config.CLASS_LABELS[p] for p in result.predictions],
            "Ground Truth": [config.CLASS_LABELS[p] for p in result.ground_truth],
        })
        self._csv_bytes = df.to_csv(index=False).encode("utf-8")
        self.export_btn.setEnabled(True)

        self._draw_pie(counts)
        self._draw_line(result.ndvi_values)

    def _draw_pie(self, counts):
        self.pie_canvas.figure.clear()
        ax = self.pie_canvas.figure.add_subplot(111)
        labels = ["Healthy", "Early", "Severe"]
        values = [counts.get(config.CELL_HEALTHY, 0), counts.get(config.CELL_EARLY, 0), counts.get(config.CELL_SEVERE, 0)]
        colors = [config.COLOR_RGB[config.CELL_HEALTHY], config.COLOR_RGB[config.CELL_EARLY], config.COLOR_RGB[config.CELL_SEVERE]]
        if sum(values) > 0:
            ax.pie(values, labels=labels, colors=colors, autopct="%1.1f%%")
        ax.set_title("Disease Spread")
        self.pie_canvas.draw_idle()

    def _draw_line(self, ndvi_values):
        self.line_canvas.figure.clear()
        ax = self.line_canvas.figure.add_subplot(111)
        ax.plot(ndvi_values, color=config.COLOR_PATH_RGB)
        ax.set_xlabel("Scan Step")
        ax.set_ylabel("NDVI")
        ax.set_title("Live NDVI Signal")
        ax.grid(True, alpha=0.3)
        self.line_canvas.draw_idle()

    def _export_csv(self):
        if not self._csv_bytes:
            return
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "scanned_field_results.csv", "CSV (*.csv)")
        if path:
            with open(path, "wb") as handle:
                handle.write(self._csv_bytes)

    def get_metrics(self):
        if not self._result:
            return {}
        c = self._result.counts
        return {
            "healthy": c.get(config.CELL_HEALTHY, 0),
            "early": c.get(config.CELL_EARLY, 0),
            "severe": c.get(config.CELL_SEVERE, 0),
            "scanned": len(self._result.predictions),
        }

    def reset(self):
        self._result = None
        self._csv_bytes = None
        self.summary.setText("Run a simulation to see results.")
        self.export_btn.setEnabled(False)
        for canvas in (self.pie_canvas, self.line_canvas):
            canvas.figure.clear()
            ax = canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            canvas.draw_idle()
