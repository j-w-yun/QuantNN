from typing import TYPE_CHECKING, Optional, List, Dict, Tuple, Sequence
from abc import ABC, ABCMeta, abstractmethod

import numpy as np


class IMLDataProvider(metaclass=ABCMeta):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def input_shapes(self) -> Dict[str, tuple]:
        """
        Property that returns all inputs' names and shapes.

        Returns:
            Dict with input name (string) as key and shape (tuple of one or more integers) as value.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def target_shapes(self) -> Dict[str, tuple]:
        """
        Property that returns all target names and shapes.

        Returns:
            Dict with target name (string) as key and shape (tuple of one or more integers) as value.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_training_samples(self):
        """
        Number of training samples.

        Returns:
            integer
        """
        raise NotImplementedError

    @abstractmethod
    def get_training_samples(self, indexes: Sequence[int]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Get a batch of training samples by providing indexes of the training samples you want.

        Args:
            indexes: Sequence (e.g. list or tuple) of integers in range 0 <= index < self.num_training_samples

        Returns:
            Returns a tuple with two dicts:
                - dict of inputs (X)
                - dict of targets (Y)
                Both inputs dict and targets dict have input/target name (string) as keys and a numpy array as values.
                The numpy arrays returned are of shapes self.input_shapes and self.target_shapes, however a dimension
                is added to be shape[0] which is always len(indexes) (e.g. batch size).
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_training_samples(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Get all training samples. The same as self.get_training_samples(list(range(self.num_training_samples)))
        but possibly implemented much more efficiently.

        Returns:
            See get_training_samples.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_val_samples(self):
        """
        Number of validation samples.

        Returns:
            integer
        """
        raise NotImplementedError

    @abstractmethod
    def get_val_samples(self, indexes: Sequence[int]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Get batch of validation samples by providing indexes of the validation samples you want.

        Args:
            indexes: Sequence (e.g. list or tuple) of integers in range 0 <= index < self.num_val_samples

        Returns:
            See get_training_samples.
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_val_samples(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Get all validation samples. The same as self.get_val_samples(list(range(self.num_val_samples)))
        but possibly implemented much more efficiently.

        Returns:
            See get_training_samples.
        """
        raise NotImplementedError


class CachedTrainingDataProvider(IMLDataProvider):
    def __init__(self,
                 inputs: Dict[str, np.ndarray]=None, targets: Dict[str, np.ndarray]=None,
                 val_inputs: Dict[str, np.ndarray]=None, val_targets: Dict[str, np.ndarray]=None):
        super().__init__()

        self.inputs = inputs
        self.targets = targets
        self.val_inputs = val_inputs
        self.val_targets = val_targets

        self.__input_shapes = {k: v.shape[1:] for k, v in inputs.items()}
        self.__target_shapes = {k: v.shape[1:] for k, v in targets.items()}
        self.__num_training_samples = list(inputs.values())[0].shape[0] if len(inputs) != 0 else 0
        self.__num_val_samples = list(val_inputs.values())[0].shape[0] if len(val_inputs) != 0 else 0

    @property
    def input_shapes(self) -> Dict[str, tuple]:
        return self.__input_shapes

    @property
    def target_shapes(self) -> Dict[str, tuple]:
        return self.__target_shapes

    @property
    def num_training_samples(self):
        return self.__num_training_samples

    def get_training_samples(self, indexes: Sequence[int]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        return {k: np.array([v[index] for index in indexes]) for k, v in self.inputs.items()}, \
               {k: np.array([v[index] for index in indexes]) for k, v in self.targets.items()}

    def get_all_training_samples(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        return self.inputs, self.targets

    @property
    def num_val_samples(self):
        return self.__num_val_samples

    def get_val_samples(self, indexes: Sequence[int]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        return {k: np.array([v[index] for index in indexes]) for k, v in self.val_inputs.items()}, \
               {k: np.array([v[index] for index in indexes]) for k, v in self.val_targets.items()}

    def get_all_val_samples(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        return self.val_inputs, self.val_targets
