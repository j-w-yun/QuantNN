from typing import TYPE_CHECKING, Optional, List, Dict, Tuple, Sequence
from abc import ABC, ABCMeta, abstractmethod

import pandas as pd
import numpy as np

from qnn.core.parameters import ParametersNode
from qnn.core import EventHook
from ..data_provider import IMLDataProvider, CachedTrainingDataProvider


class IRegressionMLModel(object):
    @staticmethod
    @abstractmethod
    def get_parameters_template() -> ParametersNode:
        """
        Get default parameters for this ML regression model.

        Returns:
            A ParametersNode object.
        """
        raise NotImplementedError

    def __init__(self, parameters: ParametersNode):
        self._parameters = parameters

        self.onTrainStep: EventHook = EventHook()  # arguments: time_elapsed (in seconds), loss (float), dict
        self.onValStep: EventHook = EventHook()  # arguments: time_elapsed (in seconds), train_loss (float), val_loss (float), dict

    @property
    def parameters(self) -> ParametersNode:
        return self._parameters

    def fit(self,
            inputs: Dict[str, np.ndarray]=None, targets: Dict[str, np.ndarray]=None,
            val_inputs: Dict[str, np.ndarray]=None, val_targets: Dict[str, np.ndarray]=None,
            training_data_provider: IMLDataProvider=None) -> None:
        """
        Fit regression ML model on provided inputs and targets data.

        You can provide data to train on in two different ways:
            - Set inputs, targets and optionally val_inputs and val_targets.
            - Or set only training_data_provider with a object that provides the training samples
              on demand. That object's class has to be a subclass of IMLDataProvider.

        Args:
            inputs: Dict with string as key and numpy matrix as value.
            targets: Dict with string as key and numpy matrix as value.
            val_inputs: Dict with string as key and numpy matrix as value.
            val_targets: Dict with string as key and numpy matrix as value.
            training_data_provider: You can optionally set this argument to IMLDataProvider subclass object.
                                    When you set this argument the other arguments are ignored.

        Returns:

        """
        if training_data_provider is None:
            training_data_provider = CachedTrainingDataProvider(inputs, targets, val_inputs, val_targets)

        return self._fit(training_data_provider)

    @abstractmethod
    def _fit(self, training_data_provider: IMLDataProvider=None):
        """
        This method is implemented by all subclasses of this class.
        This method actually trains the ML model whereas fit() is a helper method
        to convert input into a "training data provider" object if needed.

        Args:
            training_data_provider: Object whose class is a subclass of IMLDataProvider.

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, List[float]]:
        """
        Predict one series of floats per target.

        Takes a dictionary with string keys and numpy matrix values as input.
        The keys in inputs must be the same as those used in the training data and
        the shape of the numpy matrices is expected to be the same as well.

        Args:
            inputs: Dict of numpy matrices. Note that shape[0] of all matrices has to be how many predictions
                    you want to make and must be the same for every provided input.
                    So reshape input matrices to shape (1, ...) if you only want one prediction to be
                    returned from the inputs.

        Returns:
            Dict of list of floats.
            Dict keys are target names, values are lists of floats.
            Note that lengths of the returned lists are all equal to the amount of inputs specified (which
            is shape[0] of all inputs).
        """
        raise NotImplementedError

    def save_state(self, folder_path: str):
        raise NotImplementedError

    def restore_state(self, folder_path: str):
        raise NotImplementedError
