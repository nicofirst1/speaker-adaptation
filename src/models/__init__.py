from typing import Literal

from .listener.ListenerModel_hist import ListenerModel_hist
from .listener.ListenerModel_no_hist import ListenerModel_no_hist
from .simualator.model_simulator_domain import SimulatorModel_domain
from .simualator.model_simulator_hist import SimulatorModel_hist
from .simualator.model_simulator_multi import SimulatorModel_multi
from .simualator.model_simulator_no_hist import SimulatorModel_no_hist
from .simualator.model_simulator_binary import SimulatorModel_binary
from .speaker.model_speaker_hist import SpeakerModel_hist
from .speaker.model_speaker_no_hist import SpeakerModel_no_hist

__all__ = [
    "ListenerModel_hist",
    "SpeakerModel_hist",
    "SimulatorModel_hist",
    "SimulatorModel_no_hist",
    "SimulatorModel_no_hist",
    "ListenerModel_no_hist",
    "SpeakerModel_no_hist",
    "SimulatorModel_binary",
    "SimulatorModel_domain",
    "SimulatorModel_multi"
]


def get_model(model: Literal["list", "speak", "sim"], model_type: str):

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
    if model == "sim":
        if model_type=="hist":
            return SimulatorModel_hist
        elif model_type== "no_hist":
            return SimulatorModel_no_hist
        elif model_type== "binary":
            return SimulatorModel_binary
        elif model_type== "domain":
            return SimulatorModel_domain
        elif model_type== "multi":
            return SimulatorModel_multi

    else:
        raise KeyError(f"No model type named '{model}'")
