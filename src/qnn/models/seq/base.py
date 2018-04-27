from typing import TYPE_CHECKING, Optional, List, Dict
if TYPE_CHECKING:
    from qnn.market.data import MarketDataTable
    from qnn.targets.seq import ISeqTarget
    from qnn.market import Symbol, Timeframe
    from qnn.core.ranges import TimestampRange
    from qnn.core.ranges import IndexRange

from abc import ABC, ABCMeta, abstractmethod

import numpy as np

from qnn.core.parameters import ParametersNode


class ISeqModel(object):
    @staticmethod
    @abstractmethod
    def get_parameters_template() -> ParametersNode:
        raise NotImplementedError

    def __init__(self, parameters: ParametersNode, target: 'ISeqTarget', main_timeframe: 'Timeframe', symbols: List['Symbol']):
        self._parameters: ParametersNode = parameters
        self._target: 'ISeqTarget' = target
        self._main_timeframe: 'Timeframe' = main_timeframe
        self._symbols: List['Symbol'] = symbols

    @property
    def parameters(self) -> ParametersNode:
        return self._parameters

    @property
    def target(self) -> 'ISeqTarget':
        return self._target

    @property
    def main_timeframe(self) -> 'Timeframe':
        return self._main_timeframe

    @property
    def symbols(self) -> List['Symbol']:
        return self._symbols

    @abstractmethod
    def fit(self, market_data_tables: Dict[str, 'MarketDataTable'], index_range: 'IndexRange') -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, market_data_tables: Dict[str, 'MarketDataTable'], index_range: 'IndexRange') -> Dict[str, np.ndarray]:
        raise NotImplementedError
