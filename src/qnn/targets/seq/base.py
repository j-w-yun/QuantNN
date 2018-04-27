from typing import TYPE_CHECKING, Optional, List, Dict, Tuple, Sequence
if TYPE_CHECKING:
    from qnn.market import Symbol
    from qnn.market.data import MarketDataTable

from abc import ABC, ABCMeta, abstractmethod

from qnn.core.parameters import ParametersNode


class ISeqTarget(object):
    @staticmethod
    @abstractmethod
    def get_parameters_template() -> ParametersNode:
        raise NotImplementedError

    def __init__(self, parameters: ParametersNode, symbol: 'Symbol', seqlen: int):
        self._parameters = parameters
        self._symbol = symbol
        self._seqlen = seqlen

    @property
    def parameters(self) -> ParametersNode:
        return self._parameters

    @property
    def symbol(self) -> 'Symbol':
        return self._symbol

    @property
    def seqlen(self) -> int:
        return self._seqlen

    @property
    @abstractmethod
    def output_keys(self) -> Sequence[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_shapes(self) -> Sequence[any]:
        raise NotImplementedError

    @abstractmethod
    def generate_target(self, data: 'MarketDataTable', current_row: int) -> Sequence[List[float]]:
        raise NotImplementedError
