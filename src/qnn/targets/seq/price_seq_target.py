from typing import TYPE_CHECKING, Optional, List, Dict, Tuple, Sequence
if TYPE_CHECKING:
    from qnn.market import Symbol
    from qnn.market.data import MarketDataTable

from abc import ABC, ABCMeta, abstractmethod

from qnn.core.parameters import ParametersNode
from .base import ISeqTarget


class PriceSeqTarget(ISeqTarget):
    @staticmethod
    def get_parameters_template() -> ParametersNode:
        return ParametersNode()

    def __init__(self, parameters: ParametersNode, symbol: 'Symbol', seqlen: int):
        super().__init__(parameters, symbol, seqlen)

    @property
    def output_keys(self) -> Sequence[str]:
        return 'OHLC',

    @property
    def output_shapes(self) -> Sequence[any]:
        return (self.seqlen, 4),

    @abstractmethod
    def generate_target(self, data: 'MarketDataTable', current_row: int) -> Sequence[List[float]]:
        open_series = data[self._symbol.name + '.' + 'open']
        high_series = data[self._symbol.name + '.' + 'high']
        low_series = data[self._symbol.name + '.' + 'low']
        close_series = data[self._symbol.name + '.' + 'close']

        ret = []
        for i in range(self._seqlen):
            ret.append(open_series[current_row + 1 + i])
            ret.append(high_series[current_row + 1 + i])
            ret.append(low_series[current_row + 1 + i])
            ret.append(close_series[current_row + 1 + i])

        return ret,
