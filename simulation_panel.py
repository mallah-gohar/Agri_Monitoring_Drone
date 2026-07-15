from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import config
from ui.grid_canvas import GridCanvasWidget


class SimulationPanel(QWidget):
    scan_finished = pyqtSignal(object)
    reset_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pending_steps = []
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._play_next_step)
        self._paused = False
        self._result = None
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)

        self.canvas = GridCanvasWidget()
        layout.addWidget(self.canvas, stretch=3)

        side = QVBoxLayout()
        controls = QGroupBox("Simulation Controls")
        ctrl_layout = QVBoxLayout(controls)

        btn_row = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.step_btn = QPushButton("Step")
        self.step_btn.clicked.connect(self.step_once)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(self.pause_btn)
        btn_row.addWidget(self.step_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.reset_btn)
        ctrl_layout.addLayout(btn_row)

        self.progress = QProgressBar()
        ctrl_layout.addWidget(self.progress)

        self.status_label = QLabel("Ready")
        self.step_label = QLabel("Step: 0 / 0")
        self.cell_label = QLabel("Cell: —")
        self.counts_label = QLabel("Healthy: 0 | Early: 0 | Severe: 0")

        ctrl_layout.addWidget(self.status_label)
        ctrl_layout.addWidget(self.step_label)
        ctrl_layout.addWidget(self.cell_label)
        ctrl_layout.addWidget(self.counts_label)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)

        side.addWidget(controls)
        side.addWidget(QLabel("Scan Log"))
        side.addWidget(self.log)
        side.addStretch()
        layout.addLayout(side, stretch=1)

        self._speed_ms = 150
        self.set_controls_enabled(False)

    def set_controls_enabled(self, enabled):
        self.pause_btn.setEnabled(enabled)
        self.step_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(enabled)
        self.reset_btn.setEnabled(True)

    def reset(self):
        self._pending_steps.clear()
        self._timer.stop()
        self._paused = False
        self._result = None
        self.progress.setValue(0)
        self.canvas.draw_empty()
        self.status_label.setText("Ready")
        self.step_label.setText("Step: 0 / 0")
        self.cell_label.setText("Cell: —")
        self.counts_label.setText("Healthy: 0 | Early: 0 | Severe: 0")
        self.log.clear()
        self.set_controls_enabled(False)

    def begin_stream(self, speed_ms):
        self.reset()
        self._speed_ms = speed_ms
        self.status_label.setText("Simulating...")
        self.set_controls_enabled(True)
        self.pause_btn.setText("Pause")

    def enqueue_step(self, state):
        self._pending_steps.append(state)
        if not self._timer.isActive() and not self._paused:
            self._timer.start(self._speed_ms)

    def complete(self, result):
        self._result = result
        if not self._pending_steps:
            self._finish(result)

    def _finish(self, result):
        self._timer.stop()
        self.status_label.setText("Scan complete")
        self.progress.setValue(100)
        self.set_controls_enabled(False)
        self.scan_finished.emit(result)

    def _play_next_step(self):
        if not self._pending_steps:
            self._timer.stop()
            if self._result is not None:
                self._finish(self._result)
            return

        state = self._pending_steps.pop(0)
        r, c = state.drone_pos
        pred = config.CLASS_LABELS[state.prediction]
        truth = config.CLASS_LABELS[state.ground_truth]

        self.canvas.update_grid(state.scanned_grid, state.path_so_far, state.drone_pos)
        pct = int(state.step / state.total * 100)
        self.progress.setValue(pct)
        self.step_label.setText(f"Step: {state.step} / {state.total}")
        self.cell_label.setText(
            f"Cell ({r},{c}) — Pred: {pred} | Truth: {truth} | NDVI: {state.ndvi:.3f}"
        )
        counts = state.counts
        self.counts_label.setText(
            f"Healthy: {counts.get(config.CELL_HEALTHY, 0)} | "
            f"Early: {counts.get(config.CELL_EARLY, 0)} | "
            f"Severe: {counts.get(config.CELL_SEVERE, 0)}"
        )

        if state.step % 25 == 0 or state.step == state.total:
            self.log.append(f"Step {state.step}: scanned ({r},{c}) → {pred}")

    def toggle_pause(self):
        if self._paused:
            self._paused = False
            self.pause_btn.setText("Pause")
            if self._pending_steps:
                self._timer.start(self._speed_ms)
        else:
            self._paused = True
            self.pause_btn.setText("Resume")
            self._timer.stop()

    def step_once(self):
        was_paused = self._paused
        self._paused = True
        self._timer.stop()
        self._play_next_step()
        if not was_paused and self._pending_steps:
            self._paused = False
            self.pause_btn.setText("Pause")
            self._timer.start(self._speed_ms)

    def stop(self):
        self._pending_steps.clear()
        self._timer.stop()
        self._result = None
        self.status_label.setText("Stopped")
        self.set_controls_enabled(False)

    def _on_reset(self):
        self.reset()
        self.reset_clicked.emit()
