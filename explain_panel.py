from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ai.explainer import explain_cell, explain_path_choice, get_feature_importance
from core.classifier import load_classifier


class ExplainPanel(QWidget):
    """Rule-based explainability: feature importance and per-cell reasoning."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Rule-based explainability — feature importance, per-cell factors, and path algorithm reasoning. "
            "For Gemini AI explanations, use the AI Explanation tab."
        ))

        top = QHBoxLayout()
        self.importance_canvas = FigureCanvasQTAgg(Figure(figsize=(4, 3)))
        top.addWidget(self._wrap("Random Forest Feature Importance", self.importance_canvas))

        right = QVBoxLayout()
        self.cell_explain = QTextEdit()
        self.cell_explain.setReadOnly(True)
        self.cell_explain.setPlaceholderText("Click a grid cell on the Simulation tab to see per-cell reasoning.")
        right.addWidget(QLabel("Per-Cell Explainability"))
        right.addWidget(self.cell_explain)

        self.path_explain = QTextEdit()
        self.path_explain.setReadOnly(True)
        self.path_explain.setMaximumHeight(120)
        right.addWidget(QLabel("Path Algorithm Reasoning"))
        right.addWidget(self.path_explain)
        top.addLayout(right)
        layout.addLayout(top)

    def _wrap(self, title, canvas):
        box = QVBoxLayout()
        box.addWidget(QLabel(title))
        box.addWidget(canvas)
        w = QWidget()
        w.setLayout(box)
        return w

    def reset(self):
        self._result = None
        self.cell_explain.clear()
        self.path_explain.clear()
        self.importance_canvas.figure.clear()
        ax = self.importance_canvas.figure.add_subplot(111)
        ax.text(0.5, 0.5, "Run simulation to see features", ha="center", va="center")
        self.importance_canvas.draw_idle()

    def update_explainability(self, result, path_comparison):
        self._result = result
        model = load_classifier()
        importance = get_feature_importance(model)
        self._draw_importance(importance)
        self.path_explain.setPlainText(explain_path_choice(path_comparison))
        self.cell_explain.setPlainText("Click any scanned cell on the Simulation tab grid to inspect features.")

    def explain_cell_at(self, row, col):
        if not self._result:
            return
        idx = None
        for i, (r, c) in enumerate(self._result.path):
            if r == row and c == col:
                idx = i
                break
        if idx is None:
            self.cell_explain.setPlainText(f"Cell ({row},{col}) was not visited by the drone.")
            return

        model = load_classifier()
        text = explain_cell(
            self._result.features[row, col],
            self._result.predictions[idx],
            self._result.ground_truth[idx],
            self._result.crop_type,
            model,
        )
        self.cell_explain.setPlainText(text)

    def _draw_importance(self, importance):
        self.importance_canvas.figure.clear()
        ax = self.importance_canvas.figure.add_subplot(111)
        if not importance:
            ax.text(0.5, 0.5, "Train/load Random Forest model", ha="center", va="center")
        else:
            names = list(importance.keys())
            values = list(importance.values())
            ax.barh(names, values, color="#2E7D32")
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance")
        self.importance_canvas.draw_idle()
