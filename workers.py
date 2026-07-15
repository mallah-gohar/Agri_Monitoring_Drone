from PyQt6.QtCore import QThread, pyqtSignal

from ai.disease_model import train_and_save
from core.grid_utils import load_field_from_json, validate_field
from core.scanner import prepare_simulation, scan_generator


class SimulationWorker(QThread):
    step_ready = pyqtSignal(object)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, field_path, crop_type, spread_steps, path_algo, classifier_type):
        super().__init__()
        self.field_path = field_path
        self.crop_type = crop_type
        self.spread_steps = spread_steps
        self.path_algo = path_algo
        self.classifier_type = classifier_type
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            field_data = load_field_from_json(self.field_path)
            validate_field(field_data)
            prepared = prepare_simulation(
                field_data,
                self.crop_type,
                self.spread_steps,
                self.path_algo,
                self.classifier_type,
            )

            for item in scan_generator(prepared):
                if self._stop:
                    return
                if hasattr(item, "field_name"):
                    self.finished_ok.emit(item)
                else:
                    self.step_ready.emit(item)
        except Exception as exc:
            self.failed.emit(str(exc))


class TrainModelWorker(QThread):
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def run(self):
        try:
            train_and_save()
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class LLMWorker(QThread):
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, task, **kwargs):
        super().__init__()
        self.task = task
        self.kwargs = kwargs

    def run(self):
        try:
            from ai.llm_handler import (
                explain_evaluation,
                explain_everything,
                explain_simulation,
                generate_report,
                get_spray_advice,
                query_chatbot,
            )

            if self.task == "report":
                text = generate_report(**self.kwargs)
            elif self.task == "spray":
                text = get_spray_advice(**self.kwargs)
            elif self.task == "chat":
                text = query_chatbot(**self.kwargs)
            elif self.task == "explain_all":
                text = explain_everything(**self.kwargs)
            elif self.task == "explain_sim":
                text = explain_simulation(**self.kwargs)
            elif self.task == "explain_eval":
                text = explain_evaluation(**self.kwargs)
            else:
                raise ValueError(f"Unknown LLM task: {self.task}")
            self.finished_ok.emit(text)
        except Exception as exc:
            self.failed.emit(str(exc))
