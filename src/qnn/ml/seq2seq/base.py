from typing import TYPE_CHECKING, Optional, List, Dict
from abc import ABC, ABCMeta, abstractmethod

import pandas as pd
import numpy as np

from qnn.core.parameters import ParametersNode
from qnn.core import EventHook
from ..data_provider import IMLDataProvider, CachedTrainingDataProvider


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

    def fit(self,
            inputs: Dict[str, np.ndarray]=None, targets: Dict[str, np.ndarray]=None,
            val_inputs: Dict[str, np.ndarray]=None, val_targets: Dict[str, np.ndarray]=None,
            training_data_provider: IMLDataProvider=None) -> None:
        if training_data_provider is None:
            training_data_provider = CachedTrainingDataProvider(inputs, targets, val_inputs, val_targets)

        return self._fit(training_data_provider)

    @abstractmethod
    def _fit(self, training_data_provider: IMLDataProvider=None):
        raise NotImplementedError

    @abstractmethod
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def save_state(self, folder_path: str):
        raise NotImplementedError

    def restore_state(self, folder_path: str):
        raise NotImplementedError
