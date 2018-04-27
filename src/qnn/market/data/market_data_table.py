from typing import List, Dict, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from ..timeframe import Timeframe
    from ..symbol import Symbol

import pandas as pd
import numpy as np


class MarketDataTable(object):
    def __init__(self, timeframe: 'Timeframe', symbols: List['Symbol'], df: pd.DataFrame):
        """
        Construct MarketDataTable from timeframe, symbols and pre-constructed dataframe.
        :param timeframe:
        :param symbols:
        :param df: dataframe with datetime.datetime objects as index and SYMBOL.OPEN/HIGH/LOW/CLOSE/VOLUME as row keys.
        """
        self._timeframe: 'Timeframe' = timeframe
        self._symbols: List['Symbol'] = symbols
        self._df: pd.DataFrame = df

    @property
    def timeframe(self) -> 'Timeframe':
        return self._timeframe

    @property
    def symbols(self) -> List['Symbol']:
        return self._symbols

    @property
    def index(self):
        """
        Return index of pandas dataframe.
        :return:
        """
        return self._df.index

    @property
    def df(self) -> pd.DataFrame:
        """
        Return pandas dataframe.
        :return:
        """
        return self._df

    def __getitem__(self, item):
        """
        Get pandas series by name.
        Series names are in format SYMBOL.OPEN (either dot OPEN/HIGH/LOW/CLOSE/VOLUME)
        :param item:
        :return:
        """
        return self._df[item]
