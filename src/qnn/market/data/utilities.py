from typing import Dict, TYPE_CHECKING, List
if TYPE_CHECKING:
    from ..exchange import Exchange
    from ..timeframe import Timeframe
    from ..symbol import Symbol

import os

import pandas as pd

from qnn import settings
from qnn.core.serialization import load_pandas_dataframe_from_csv_file
from .market_data_table import MarketDataTable


def create_market_data_table(timeframe: 'Timeframe', symbol_bardata: Dict['Symbol', pd.DataFrame]) -> MarketDataTable:
    for symbol, df in symbol_bardata.items():
        df.columns = ['%s.%s' % (symbol.name, c) for c in df.columns]

    return MarketDataTable(timeframe, list(symbol_bardata.keys()), pd.concat(list(symbol_bardata.values()), axis=1))


def load_symbols_bardata(timeframe: 'Timeframe', symbols: List['Symbol'], dtype=None):
    symbol_bardata = {}
    for s in symbols:
        dirpath = os.path.join(settings.DB_PATH, s.exchange.name, 'bardata', str(int(timeframe.duration.total_seconds())))
        if not os.path.exists(dirpath):
            raise RuntimeError('Data directory "%s" does not exist' % dirpath)

        symbolpath = os.path.join(dirpath, s.name + '.csv')
        if not os.path.exists(symbolpath):
            raise RuntimeError('Symbol data file "%s" not found' % symbolpath)

        symbol_bardata[s] = load_pandas_dataframe_from_csv_file(symbolpath, dtype=dtype)

    return symbol_bardata
