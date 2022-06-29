import operator
import sys
from typing import Literal


class EarlyStopping:
    """
    Class used to early stop training
    """
    def __init__(self, max_patient: int, problem_formulation: Literal['max','min']):
        """

        Parameters
        ----------
        max_patient : max number of epochs before stopping training if no improvement
        problem_formulation : either maximization of metric or minimization
        """

        if problem_formulation=='max':
            self.comparison=operator.le
            self.best_metric = sys.float_info.min

        else:
            self.comparison = operator.ge
            self.best_metric = sys.float_info.max


        self.max_patient = max_patient
        self.patient_couter = 0



    def should_stop(self, metric_val) -> bool:

        if self.comparison(self.best_metric, metric_val):
            # the metric has a value more acceptable than the stored one, saved and reset counter
            self.best_metric = metric_val
            self.patient_couter = 0
        else:
            self.patient_couter += 1

        # check the value of the counter
        if self.patient_couter >= self.max_patient:
            print(
                f"Early stopping after {self.patient_couter} epochs!\n"
                f"Prev metric value '{self.best_metric}' vs current '{metric_val}'"
            )
            return True
        else:
            return False
