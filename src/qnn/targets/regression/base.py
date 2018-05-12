from typing import TYPE_CHECKING, Optional, List, Dict, Tuple, Sequence
if TYPE_CHECKING:
    from qnn.market import Symbol
    from qnn.market.data import MarketDataTable

from abc import ABC, ABCMeta, abstractmethod

from qnn.core.parameters import ParametersNode


class IRegressionTarget(object):
    @staticmethod
    @abstractmethod
    def get_parameters_template() -> ParametersNode:
        raise NotImplementedError

    def __init__(self, parameters: ParametersNode, symbol: 'Symbol'):
        self._parameters = parameters
        self._symbol = symbol

    @property
    def parameters(self) -> ParametersNode:
        return self._parameters

    @property
    def symbol(self) -> 'Symbol':
        return self._symbol

    @abstractmethod
    def generate_target(self, data: 'MarketDataTable', current_row: int) -> float:
        raise NotImplementedError
