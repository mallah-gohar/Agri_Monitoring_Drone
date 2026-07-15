from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QWidget,
)

from core.classifier import load_classifier
from core.evaluator import compare_classifiers, compare_paths
from ui.ai_explanation_panel import AIExplanationPanel
from ui.evaluation_panel import EvaluationPanel
from ui.explain_panel import ExplainPanel
from ui.results_panel import ResultsPanel
from ui.setup_panel import SetupPanel
from ui.simulation_panel import SimulationPanel
from ui.workers import LLMWorker, SimulationWorker, TrainModelWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AgriDrone — Crop Health Desktop Simulator")
        self.resize(1280, 820)
        self._sim_worker = None
        self._train_worker = None
        self._llm_worker = None
        self._last_result = None
        self._classifier_metrics = {}
        self._path_metrics = {}
        self._build_ui()
        self._build_menu()
        self.statusBar().showMessage("Ready — configure field and start simulation.")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        self.setup_panel = SetupPanel()
        self.setup_panel.setMaximumWidth(340)
        root.addWidget(self.setup_panel)

        self.tabs = QTabWidget()
        self.sim_panel = SimulationPanel()
        self.results_panel = ResultsPanel()
        self.eval_panel = EvaluationPanel()
        self.explain_panel = ExplainPanel()
        self.ai_panel = AIExplanationPanel()

        self.tabs.addTab(self.sim_panel, "Simulation")
        self.tabs.addTab(self.results_panel, "Results")
        self.tabs.addTab(self.eval_panel, "Evaluation")
        self.tabs.addTab(self.explain_panel, "Explainability")
        self.tabs.addTab(self.ai_panel, "AI Explanation")
        root.addWidget(self.tabs, stretch=1)

        self.setup_panel.run_requested.connect(self._start_simulation)
        self.setup_panel.train_requested.connect(self._train_model)
        self.setup_panel.reset_requested.connect(self._reset_simulation)
        self.sim_panel.scan_finished.connect(self._on_scan_finished)
        self.sim_panel.reset_clicked.connect(self._reset_simulation)
        self.ai_panel.llm_requested.connect(self._handle_llm_request)
        self.sim_panel.canvas.cell_clicked.connect(self.explain_panel.explain_cell_at)

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("File")
        quit_action = QAction("Exit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _show_about(self):
        QMessageBox.information(
            self,
            "About AgriDrone",
            "AgriDrone Desktop Simulator\n\n"
            "AI Lab Project — Search (A* + Boustrophedon), ML (Random Forest), "
            "and Gemini AI explanations.",
        )

    def _reset_simulation(self):
        if self._sim_worker and self._sim_worker.isRunning():
            self._sim_worker.stop()
            self._sim_worker.wait(2000)

        self._last_result = None
        self._classifier_metrics = {}
        self._path_metrics = {}
        self.setup_panel.set_running(False)
        self.sim_panel.reset()
        self.results_panel.reset()
        self.eval_panel.reset()
        self.explain_panel.reset()
        self.ai_panel.reset()
        self.tabs.setCurrentWidget(self.sim_panel)
        self.statusBar().showMessage("Simulation reset — ready for new run.")

    def _start_simulation(self, params):
        if "Random Forest" in params["classifier_type"] and load_classifier() is None:
            reply = QMessageBox.question(
                self,
                "Model Missing",
                "Random Forest model not found. Train it now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._train_model()
            return

        if self._sim_worker and self._sim_worker.isRunning():
            self._sim_worker.stop()
            self._sim_worker.wait(1000)

        self.tabs.setCurrentWidget(self.sim_panel)
        self.sim_panel.begin_stream(self.setup_panel.speed_ms())
        self.setup_panel.set_running(True)
        self.statusBar().showMessage("Loading field and preparing scan...")

        self._sim_worker = SimulationWorker(
            params["field_path"],
            params["crop_type"],
            params["spread_steps"],
            params["path_algo"],
            params["classifier_type"],
        )
        self._sim_worker.step_ready.connect(self.sim_panel.enqueue_step)
        self._sim_worker.finished_ok.connect(self._on_worker_finished)
        self._sim_worker.failed.connect(self._on_sim_failed)
        self._sim_worker.start()

    def _on_worker_finished(self, result):
        self.sim_panel.complete(result)

    def _on_scan_finished(self, result):
        self._last_result = result
        self.setup_panel.set_running(False)
        self.statusBar().showMessage(
            "Scan complete — review Results, Evaluation, Explainability, and AI Explanation tabs."
        )

        self.results_panel.update_results(result)
        self._classifier_metrics = compare_classifiers(
            result.rf_predictions, result.ndvi_predictions, result.ground_truth
        )
        self._path_metrics = compare_paths(
            result.path_boustro, result.path_rw, result.grid, result.astar_nodes_explored
        )
        self.eval_panel.update_evaluation(result)
        self.explain_panel.update_explainability(result, self._path_metrics)
        self.ai_panel.on_scan_complete()

    def _on_sim_failed(self, message):
        self.setup_panel.set_running(False)
        self.sim_panel.stop()
        self.statusBar().showMessage(f"Error: {message}")
        QMessageBox.critical(self, "Simulation Error", message)

    def _train_model(self):
        self.setup_panel.set_running(True)
        self.statusBar().showMessage("Training Random Forest classifier...")
        self._train_worker = TrainModelWorker()
        self._train_worker.finished_ok.connect(self._on_train_ok)
        self._train_worker.failed.connect(self._on_train_failed)
        self._train_worker.start()

    def _on_train_ok(self):
        self.setup_panel.set_running(False)
        self.statusBar().showMessage("Classifier trained and saved to models/disease_clf.pkl")
        QMessageBox.information(self, "Training Complete", "Random Forest model saved successfully.")

    def _on_train_failed(self, message):
        self.setup_panel.set_running(False)
        self.statusBar().showMessage(f"Training failed: {message}")
        QMessageBox.critical(self, "Training Error", message)

    def _handle_llm_request(self, token):
        if not self._last_result:
            QMessageBox.warning(self, "No Data", "Run a simulation first.")
            return

        metrics = self.results_panel.get_metrics()
        result = self._last_result

        if token.startswith("chat:"):
            question = token.split(":", 1)[1]
            kwargs = {
                "question": question,
                "chat_history": self.ai_panel.chat_history(),
                "metrics": metrics,
                "crop_type": result.crop_type,
                "field_name": result.field_name,
            }
            task = "chat"
            title = "Chat Response"
        elif token == "explain_all":
            kwargs = {
                "result": result,
                "metrics": metrics,
                "classifier_metrics": self._classifier_metrics,
                "path_metrics": self._path_metrics,
            }
            task = "explain_all"
            title = "Complete AI Explanation"
        elif token == "explain_sim":
            kwargs = {"result": result, "metrics": metrics}
            task = "explain_sim"
            title = "Simulation Explanation"
        elif token == "explain_eval":
            kwargs = {
                "classifier_metrics": self._classifier_metrics,
                "path_metrics": self._path_metrics,
                "result": result,
            }
            task = "explain_eval"
            title = "Evaluation Explanation"
        elif token == "report":
            kwargs = {"metrics": metrics, "crop_type": result.crop_type, "field_name": result.field_name}
            task = "report"
            title = "Field Report"
        else:
            kwargs = {"metrics": metrics, "crop_type": result.crop_type}
            task = "spray"
            title = "Spray Advice"

        self.statusBar().showMessage("Contacting Gemini API...")
        self._llm_worker = LLMWorker(task, **kwargs)
        self._llm_worker.finished_ok.connect(lambda t: self._on_llm_response(title, t, task))
        self._llm_worker.failed.connect(lambda m: self.statusBar().showMessage(f"LLM error: {m}"))
        self._llm_worker.start()

    def _on_llm_response(self, title, text, task):
        self.statusBar().showMessage("Gemini response ready.")
        if task == "chat":
            self.ai_panel.append_chat("assistant", text)
        else:
            self.ai_panel.set_output(text)
