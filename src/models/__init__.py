from typing import Literal

from .listener.ListenerModel_hist import ListenerModel_hist
from .listener.ListenerModel_no_hist import ListenerModel_no_hist
from .interpreter.model_interpreter_domain import InterpreterModel_domain
from .interpreter.model_interpreter_hist import InterpreterModel_hist
from .interpreter.model_interpreter_multi import InterpreterModel_multi
from .interpreter.model_interpreter_no_hist import InterpreterModel_no_hist
from .interpreter.model_interpreter_binary import InterpreterModel_binary
from .speaker.model_speaker_hist import SpeakerModel_hist
from .speaker.model_speaker_no_hist import SpeakerModel_no_hist

__all__ = [
    "ListenerModel_hist",
    "SpeakerModel_hist",
    "InterpreterModel_hist",
    "InterpreterModel_no_hist",
    "InterpreterModel_no_hist",
    "ListenerModel_no_hist",
    "SpeakerModel_no_hist",
    "InterpreterModel_binary",
    "InterpreterModel_domain",
    "InterpreterModel_multi"
]


def get_model(model: Literal["list", "speak", "int"], model_type: str):

    if model == "list":
        if model_type=="hist":
            return ListenerModel_hist
        elif model_type == "no_hist":
            return ListenerModel_no_hist
    elif model == "speak":
        if model_type=="hist":
            return SpeakerModel_hist
        elif model_type=="no_hist":
            return SpeakerModel_no_hist
    if model == "int":
        if model_type=="hist":
            return InterpreterModel_hist
        elif model_type== "no_hist":
            return InterpreterModel_no_hist
        elif model_type== "binary":
            return InterpreterModel_binary
        elif model_type== "domain":
            return InterpreterModel_domain
        elif model_type== "multi":
            return InterpreterModel_multi

    else:
        raise KeyError(f"No model type named '{model}'")
