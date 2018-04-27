from abc import ABCMeta, abstractmethod

from qnn.core.parameters import ParametersNode


class INetworkOptimizer(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def get_parameters_template() -> ParametersNode:
        raise NotImplementedError

    def __init__(self, parameters: ParametersNode):
        self._parameters = parameters

    @property
    def parameters(self) -> ParametersNode:
        return self._parameters

    @abstractmethod
    def create_train_op(self, cost, trainable_vars, learning_rate_decay_steps: int):
        """
        Create tensorflow training op.
        :return: train_op, global_step, learning_rate
        """
        raise NotImplementedError
