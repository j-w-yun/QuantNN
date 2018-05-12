from typing import TYPE_CHECKING, Optional, List, Dict
if TYPE_CHECKING:
    from qnn.market.data import MarketDataTable
    from qnn.targets.regression import IRegressionTarget
    from qnn.market import Symbol, Timeframe
    from qnn.core.ranges import TimestampRange
    from qnn.core.ranges import IndexRange

from abc import ABC, ABCMeta, abstractmethod

import numpy as np

from qnn.core.parameters import ParametersNode


class IRegressionModel(object):
    @staticmethod
    @abstractmethod
    def get_parameters_template() -> ParametersNode:
        """
        Get default parameters for this model.

        Returns: ParametersNode object.
        """
        raise NotImplementedError

    def __init__(self, parameters: ParametersNode, targets: Dict[str, 'IRegressionTarget'], main_timeframe: 'Timeframe', symbols: List['Symbol']):
        """
        Create regression model.

        Args:
            parameters: A ParametersNode object (as returned from get_parameters_template(), which returns the defaults).
            targets: Dict of target objects. Target objects specify what we want to predict and are also used to generate targets.
            main_timeframe: Timeframe object that is the timeframe with which to generate the targets.
            symbols: List of symbols whose data can be used as inputs to the model.
        """
        self._parameters: ParametersNode = parameters
        self._targets: Dict[str, 'IRegressionTarget'] = targets
        self._main_timeframe: 'Timeframe' = main_timeframe
        self._symbols: List['Symbol'] = symbols

    @property
    def parameters(self) -> ParametersNode:
        return self._parameters

    @property
    def targets(self) -> Dict[str, 'IRegressionTarget']:
        return self._targets

    @property
    def main_timeframe(self) -> 'Timeframe':
        return self._main_timeframe

    @property
    def symbols(self) -> List['Symbol']:
        return self._symbols

    @abstractmethod
    def fit(self, market_data_tables: Dict[str, 'MarketDataTable'], index_range: 'IndexRange') -> None:
        """
        Fit the regression model on the provided data within the given range of that data.

        Args:
            market_data_tables: Dict of market data tables, one per timeframe. Keys are names of timeframes.
            index_range: Index range (on the main timeframe's market data table) of rows to use from market data table as training data.

        Returns:
            Nothing
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, market_data_tables: Dict[str, 'MarketDataTable'], index_range: 'IndexRange') -> Dict[str, List[float]]:
        """
        Return predictions using given market data tables.
        Predictions should be from the rows in the main market data table as specified by index_range.

        Args:
            market_data_tables: Dict of market data tables, one per timeframe. Keys are names of timeframes.
            index_range: Index range

        Returns:
            Dict of lists of float.
            Dict keys are target names as they were given to the constructor and dict values are a list of floats;
            every float is one prediction within index_range.
        """
        raise NotImplementedError

    def save_state(self, folder_path: str):
        raise NotImplementedError

    def restore_state(self, folder_path: str):
        raise NotImplementedError
