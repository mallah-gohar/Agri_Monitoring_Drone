from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import config
from core.grid_utils import resolve_field_path


class SetupPanel(QWidget):
    run_requested = pyqtSignal(dict)
    train_requested = pyqtSignal()
    reset_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        form_group = QGroupBox("Field Configuration")
        form = QFormLayout(form_group)

        self.crop_combo = QComboBox()
        self.crop_combo.addItems(config.CROP_TYPES)

        self.field_combo = QComboBox()
        for key in config.FIELD_LAYOUTS:
            label = config.FIELD_LAYOUT_LABELS.get(key, key)
            self.field_combo.addItem(label, key)

        self.path_combo = QComboBox()
        self.path_combo.addItems(["Boustrophedon (A* Detour)", "Random Walk (Baseline)"])

        self.classifier_combo = QComboBox()
        self.classifier_combo.addItems(["Random Forest (ML)", "NDVI Threshold (Baseline)"])

        self.spread_slider = QSlider()
        self.spread_slider.setMinimum(0)
        self.spread_slider.setMaximum(10)
        self.spread_slider.setValue(5)
        self.spread_label = QLabel("5")

        spread_row = QHBoxLayout()
        spread_row.addWidget(self.spread_slider)
        spread_row.addWidget(self.spread_label)
        self.spread_slider.valueChanged.connect(lambda v: self.spread_label.setText(str(v)))

        self.speed_slider = QSlider()
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(10)
        self.speed_slider.setValue(5)
        self.speed_label = QLabel("Medium")
        speed_map = {1: "Very Slow", 2: "Slow", 3: "Slow", 4: "Medium", 5: "Medium",
                     6: "Medium", 7: "Fast", 8: "Fast", 9: "Very Fast", 10: "Very Fast"}
        self.speed_slider.valueChanged.connect(lambda v: self.speed_label.setText(speed_map.get(v, "Medium")))

        form.addRow("Crop Type", self.crop_combo)
        form.addRow("Field Layout", self.field_combo)
        form.addRow("Path Algorithm", self.path_combo)
        form.addRow("Classifier", self.classifier_combo)
        form.addRow("Disease Spread Steps", spread_row)
        form.addRow("Simulation Speed", self.speed_slider)
        form.addRow("", self.speed_label)

        layout.addWidget(form_group)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Start Drone Simulation")
        self.run_btn.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 8px;")
        self.run_btn.clicked.connect(self._emit_run)

        self.train_btn = QPushButton("Train Classifier")
        self.train_btn.clicked.connect(self.train_requested.emit)

        self.browse_btn = QPushButton("Load Custom JSON")
        self.browse_btn.clicked.connect(self._browse_field)

        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.train_btn)
        btn_row.addWidget(self.browse_btn)
        layout.addLayout(btn_row)

        self.reset_btn = QPushButton("Reset Simulation")
        self.reset_btn.setStyleSheet("background-color: #757575; color: white; font-weight: bold; padding: 8px;")
        self.reset_btn.clicked.connect(self.reset_requested.emit)
        layout.addWidget(self.reset_btn)

        self.custom_path = None
        self.custom_label = QLabel("")
        self.custom_label.setWordWrap(True)
        layout.addWidget(self.custom_label)

        info = QLabel(
            "Configure the field, choose AI algorithms, then start the simulation. "
            "The desktop app runs scan steps locally for smooth animation."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addStretch()

    def _browse_field(self):
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(self, "Select Field JSON", str(resolve_field_path("flat_farm").parent), "JSON (*.json)")
        if path:
            self.custom_path = path
            self.custom_label.setText(f"Custom field: {path}")

    def _emit_run(self):
        field_key = self.field_combo.currentData()
        field_path = self.custom_path or str(resolve_field_path(field_key))
        self.run_requested.emit({
            "field_path": field_path,
            "crop_type": self.crop_combo.currentText(),
            "spread_steps": self.spread_slider.value(),
            "path_algo": self.path_combo.currentText(),
            "classifier_type": self.classifier_combo.currentText(),
            "speed": self.speed_slider.value(),
        })

    def set_running(self, running):
        self.run_btn.setEnabled(not running)
        self.train_btn.setEnabled(not running)

    def speed_ms(self):
        return max(20, 450 - self.speed_slider.value() * 40)
