from typing import Literal, Optional

from .simulator.SimulatorModel import SimulatorModel
from .listener.ListenerModel import ListenerModel
from .speaker import SpeakerModel
from .speaker.SpeakerModel import SpeakerModel

__all__ = [
    "SpeakerModel",
    "ListenerModel",
    "SimulatorModel",

]


def get_model(model: Literal["list", "speak", "sim"], model_type: Optional[str] = ""):
    if model == "list":
        return ListenerModel
    elif model == "speak":
        return SpeakerModel
    if model == "sim":
        return SimulatorModel

    else:
        raise KeyError(f"No model type named '{model}'")
