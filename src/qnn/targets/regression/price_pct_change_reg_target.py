from typing import TYPE_CHECKING, Optional, List, Dict, Tuple, Sequence
if TYPE_CHECKING:
    from qnn.market import Symbol
    from qnn.market.data import MarketDataTable

from qnn.core.parameters import ParametersNode
from .base import IRegressionTarget


class PricePctChangeRegressionTarget(IRegressionTarget):
    @staticmethod
    def get_parameters_template() -> ParametersNode:
        return ParametersNode()

    def __init__(self, parameters: ParametersNode, symbol: 'Symbol'):
        super().__init__(parameters, symbol)

    def generate_target(self, data: 'MarketDataTable', current_row: int) -> float:
        close_series = data[self._symbol.name + '.' + 'close']
        return (close_series[current_row + 1] - close_series[current_row]) / close_series[current_row]
