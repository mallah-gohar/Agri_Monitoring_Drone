from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from core.evaluator import compare_classifiers, compare_paths


class EvaluationPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Evaluation — compare classifiers and path algorithms (lab requirement)."))

        self.classifier_table = QTableWidget(3, 5)
        self.classifier_table.setHorizontalHeaderLabels(["Classifier", "Accuracy %", "Precision", "Recall", "F1"])
        layout.addWidget(QLabel("Classifier Comparison"))
        layout.addWidget(self.classifier_table)

        self.path_table = QTableWidget(2, 6)
        self.path_table.setHorizontalHeaderLabels(
            ["Algorithm", "Path Length", "Coverage %", "Gaps", "Efficiency %", "A* Nodes"]
        )
        layout.addWidget(QLabel("Path Algorithm Comparison"))
        layout.addWidget(self.path_table)

        charts = QHBoxLayout()
        self.f1_canvas = FigureCanvasQTAgg(Figure(figsize=(4, 3)))
        self.cov_canvas = FigureCanvasQTAgg(Figure(figsize=(4, 3)))
        charts.addWidget(self._wrap("F1 Score Comparison", self.f1_canvas))
        charts.addWidget(self._wrap("Coverage Comparison", self.cov_canvas))
        layout.addLayout(charts)

        self.runtime_label = QLabel("Scan runtime: —")
        layout.addWidget(self.runtime_label)

    def _wrap(self, title, canvas):
        box = QVBoxLayout()
        box.addWidget(QLabel(title))
        box.addWidget(canvas)
        w = QWidget()
        w.setLayout(box)
        return w

    def update_evaluation(self, result):
        clf_metrics = compare_classifiers(result.rf_predictions, result.ndvi_predictions, result.ground_truth)
        path_metrics = compare_paths(result.path_boustro, result.path_rw, result.grid, result.astar_nodes_explored)

        self._fill_classifier_table(clf_metrics)
        self._fill_path_table(path_metrics)
        self._draw_f1(clf_metrics)
        self._draw_coverage(path_metrics)
        self.runtime_label.setText(f"Scan runtime: {result.scan_duration_sec:.2f} seconds")

    def _fill_classifier_table(self, metrics):
        rows = list(metrics.items())
        self.classifier_table.setRowCount(len(rows))
        for i, (name, m) in enumerate(rows):
            self.classifier_table.setItem(i, 0, QTableWidgetItem(name))
            self.classifier_table.setItem(i, 1, QTableWidgetItem(f"{m['accuracy']:.1f}"))
            self.classifier_table.setItem(i, 2, QTableWidgetItem(f"{m['precision']:.3f}"))
            self.classifier_table.setItem(i, 3, QTableWidgetItem(f"{m['recall']:.3f}"))
            self.classifier_table.setItem(i, 4, QTableWidgetItem(f"{m['f1']:.3f}"))
        self.classifier_table.resizeColumnsToContents()

    def _fill_path_table(self, metrics):
        rows = list(metrics.items())
        self.path_table.setRowCount(len(rows))
        for i, (name, m) in enumerate(rows):
            self.path_table.setItem(i, 0, QTableWidgetItem(name))
            self.path_table.setItem(i, 1, QTableWidgetItem(str(m["length"])))
            self.path_table.setItem(i, 2, QTableWidgetItem(f"{m['coverage']:.1f}"))
            self.path_table.setItem(i, 3, QTableWidgetItem(str(m["gaps"])))
            self.path_table.setItem(i, 4, QTableWidgetItem(f"{m['efficiency']:.1f}"))
            self.path_table.setItem(i, 5, QTableWidgetItem(str(m["astar_nodes"])))
        self.path_table.resizeColumnsToContents()

    def _draw_f1(self, metrics):
        self.f1_canvas.figure.clear()
        ax = self.f1_canvas.figure.add_subplot(111)
        names = list(metrics.keys())
        values = [metrics[n]["f1"] for n in names]
        ax.bar(names, values, color=["#2E7D32", "#FFB300"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1 Score")
        ax.set_title("Classifier F1 Comparison")
        self.f1_canvas.draw_idle()

    def _draw_coverage(self, metrics):
        self.cov_canvas.figure.clear()
        ax = self.cov_canvas.figure.add_subplot(111)
        names = list(metrics.keys())
        values = [metrics[n]["coverage"] for n in names]
        ax.bar(names, values, color=["#29B6F6", "#9E9E9E"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Coverage %")
        ax.set_title("Path Coverage Comparison")
        self.cov_canvas.draw_idle()

    def reset(self):
        self.classifier_table.setRowCount(0)
        self.path_table.setRowCount(0)
        self.runtime_label.setText("Scan runtime: —")
        for canvas in (self.f1_canvas, self.cov_canvas):
            canvas.figure.clear()
            ax = canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            canvas.draw_idle()
