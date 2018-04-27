from typing import TYPE_CHECKING, Optional, List, Dict
from abc import ABC, ABCMeta, abstractmethod

import pandas as pd
import numpy as np

from qnn.core.parameters import ParametersNode
from qnn.core import EventHook


class ISeq2SeqModel(object):
    @staticmethod
    @abstractmethod
    def get_parameters_template() -> ParametersNode:
        raise NotImplementedError

    def __init__(self, parameters: ParametersNode):
        self._parameters = parameters

        self.onTrainStep: EventHook = EventHook()  # arguments: time_elapsed (in seconds), dict

    @property
    def parameters(self) -> ParametersNode:
        return self._parameters

    @abstractmethod
    def fit(self,
            inputs: Dict[str, np.ndarray], targets: Dict[str, np.ndarray],
            val_inputs: Dict[str, np.ndarray] = None, val_targets: Dict[str, np.ndarray] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        raise NotImplementedError
