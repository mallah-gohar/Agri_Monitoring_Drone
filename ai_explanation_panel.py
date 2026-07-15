from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ai.llm_handler import check_api_status


class AIExplanationPanel(QWidget):
    llm_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._chat_history = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        header = QGroupBox("Gemini AI Explanation")
        header_layout = QVBoxLayout(header)

        self.api_label = QLabel()
        self._update_api_status()
        header_layout.addWidget(self.api_label)

        info = QLabel(
            "Use Google Gemini to get natural-language explanations of simulation results, "
            "evaluation metrics, algorithm choices, and treatment recommendations."
        )
        info.setWordWrap(True)
        header_layout.addWidget(info)
        layout.addWidget(header)

        btn_row1 = QHBoxLayout()
        self.explain_all_btn = QPushButton("Explain Everything")
        self.explain_all_btn.setStyleSheet("background-color: #1565C0; color: white; font-weight: bold; padding: 8px;")
        self.explain_sim_btn = QPushButton("Explain Simulation")
        self.explain_eval_btn = QPushButton("Explain Evaluation")
        self.explain_all_btn.clicked.connect(lambda: self.llm_requested.emit("explain_all"))
        self.explain_sim_btn.clicked.connect(lambda: self.llm_requested.emit("explain_sim"))
        self.explain_eval_btn.clicked.connect(lambda: self.llm_requested.emit("explain_eval"))
        btn_row1.addWidget(self.explain_all_btn)
        btn_row1.addWidget(self.explain_sim_btn)
        btn_row1.addWidget(self.explain_eval_btn)
        layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.report_btn = QPushButton("Generate Field Report")
        self.spray_btn = QPushButton("Get Spray Advice")
        self.report_btn.clicked.connect(lambda: self.llm_requested.emit("report"))
        self.spray_btn.clicked.connect(lambda: self.llm_requested.emit("spray"))
        btn_row2.addWidget(self.report_btn)
        btn_row2.addWidget(self.spray_btn)
        layout.addLayout(btn_row2)

        layout.addWidget(QLabel("AI Explanation Output"))
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText(
            "Run a simulation, then click any explanation button above.\n"
            "Gemini will explain scan results, severe disease zones, metrics, and recommendations."
        )
        layout.addWidget(self.output)

        chat_group = QGroupBox("Ask Gemini Anything")
        chat_layout = QVBoxLayout(chat_group)
        chat_row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask about severe disease, scan results, algorithms, treatment...")
        self.chat_send = QPushButton("Ask Gemini")
        self.chat_send.clicked.connect(self._send_chat)
        chat_row.addWidget(self.chat_input)
        chat_row.addWidget(self.chat_send)
        chat_layout.addLayout(chat_row)

        self.chat_log = QTextEdit()
        self.chat_log.setReadOnly(True)
        self.chat_log.setMaximumHeight(180)
        chat_layout.addWidget(self.chat_log)
        layout.addWidget(chat_group)

        self.set_buttons_enabled(False)

    def _update_api_status(self):
        status = check_api_status()
        if status == "Connected":
            self.api_label.setText("Gemini API: Connected ✓")
            self.api_label.setStyleSheet("color: #2E7D32; font-weight: bold;")
        else:
            self.api_label.setText("Gemini API: Not configured — add GEMINI_API_KEY to .env (mock text will be used)")
            self.api_label.setStyleSheet("color: #D32F2F; font-weight: bold;")

    def set_buttons_enabled(self, enabled):
        for btn in (
            self.explain_all_btn,
            self.explain_sim_btn,
            self.explain_eval_btn,
            self.report_btn,
            self.spray_btn,
            self.chat_send,
        ):
            btn.setEnabled(enabled)

    def set_output(self, text):
        self.output.setPlainText(text)

    def append_output(self, title, text):
        self.output.append(f"=== {title} ===\n{text}\n")

    def reset(self):
        self._chat_history = []
        self.output.clear()
        self.chat_log.clear()
        self.set_buttons_enabled(False)
        self._update_api_status()

    def on_scan_complete(self):
        self.set_buttons_enabled(True)
        self._update_api_status()
        self.output.setPlainText(
            "Simulation complete. Click 'Explain Everything' for a full Gemini analysis, "
            "or use individual buttons to explain specific parts."
        )

    def append_chat(self, role, content):
        prefix = "You" if role == "user" else "Gemini"
        self.chat_log.append(f"{prefix}: {content}")
        self._chat_history.append({"role": role, "content": content})

    def _send_chat(self):
        question = self.chat_input.text().strip()
        if question:
            self.chat_input.clear()
            self.append_chat("user", question)
            self.llm_requested.emit(f"chat:{question}")

    def chat_history(self):
        return self._chat_history
