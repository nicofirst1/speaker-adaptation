import sys
from typing import Callable


class EarlyStopping:

    def __init__(self, max_patient: int, comparison: Callable[[float, float], bool]):
        self.max_patient = max_patient
        self.patient_couter = 0
        self.comparison = comparison

        # init the best metric based on the operator passed (> or <)
        if comparison(0, 9999):
            self.best_metric = sys.float_info.min
        else:
            self.best_metric = sys.float_info.max

    def should_stop(self, metric_val) -> bool:

        if self.comparison(self.best_metric, metric_val):
            # the metric has a value more acceptable than the stored one, saved and reset counter
            self.best_metric = metric_val
            self.patient_couter = 0
        else:
            self.patient_couter += 1

        # check the value of the counter
        if self.patient_couter >= self.max_patient:
            print(f"Early stopping after {self.patient_couter} epochs!")
            return True
        else:
            return False
