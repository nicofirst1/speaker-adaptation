from typing import Literal

from .listener.ListenerModel_no_hist import ListenerModel_no_hist
from .listener.ListenerModel_hist import ListenerModel_hist
from .simualator.model_simulator_hist import SimulatorModel_hist
from .simualator.model_simulator_no_hist import SimulatorModel_no_hist
from .speaker.model_speaker_hist import SpeakerModel_hist
from .speaker.model_speaker_no_hist import SpeakerModel_no_hist

__all__ = [
    "ListenerModel_hist",
    "SpeakerModel_hist",
    "SimulatorModel_hist",
    "SimulatorModel_no_hist",
    "ListenerModel_no_hist",
    "SpeakerModel_no_hist",
]


def get_model(model_type:Literal["list","speak","sim"], model_hist:str):
    model_hist="hist"== model_hist

    if model_type == "list":
        if model_hist:
            return ListenerModel_hist
        else:
            return ListenerModel_no_hist
    elif model_type == "speak":
        if model_hist:
            return SpeakerModel_hist
        else:
            return SpeakerModel_no_hist
    if model_type == "sim":
        if model_hist:
            return SimulatorModel_hist
        else:
            return SimulatorModel_no_hist
    else:
        raise KeyError(f"No mdoel type named '{model_type}'")