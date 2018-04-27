"""
This program downloads symbol bar-data from poloniex.
Every symbol gets its own pandas dataframe and is stored as .csv file.
"""

from typing import List, Optional, Tuple
import logging
import os
import sys
import time
import datetime
import pytz

import pandas as pd

from qnn import settings
from qnn.core.serialization import pandas_dataframe_to_csv_file

FETCH_URL = "https://poloniex.com/public?command=returnChartData&currencyPair=%s&start=%d&end=%d&period=%s"  # candlestick period in seconds; valid values are 300, 900, 1800, 7200, 14400, and 86400
COLUMNS = ["date","high","low","open","close","volume","quoteVolume","weightedAverage"]

logger = logging.getLogger(__name__)


def download_poloniex_symbol_list() -> Tuple[List[str], List[str]]:
    df = pd.read_json("https://poloniex.com/public?command=return24hVolume")

    ret = []
    for v in df.columns:
        if '_' in v:
            ret.append('POLO.%s_%s' % (v.split('_')[1], v.split('_')[0]))
        else:
            ret.append(v)
    return list(df.columns), ret


def download_poloniex_bardata(symbol_key: str, timeframe_seconds: int) -> Optional[pd.DataFrame]:
    start_time = int(datetime.datetime(2014, 1, 1, 0, 0, 0, 0, tzinfo=pytz.UTC).timestamp())
    end_time = int(datetime.datetime(2040, 1, 1, 0, 0, 0, 0, tzinfo=pytz.UTC).timestamp())   # Far in the future, poloniex will return data for everything up to now.

    url = FETCH_URL % (symbol_key, start_time, end_time, timeframe_seconds)

    df = None
    for i in range(3):
        try:
            df = pd.read_json(url, convert_dates=False)
            break
        except Exception as e:
            print(e)
            print('Error! Retrying in 10 seconds...')
            time.sleep(10)
    if df is None:
        print('Could not download data for %s! Skipping...' % symbol_key)
        time.sleep(10)
        return None

    if df["date"].iloc[-1] == 0:
        print("No data.")
        return None

    df.set_index('date', inplace=True)

    return df


def main():
    if len(sys.argv) != 2:
        print("Please provide: [timeframe in seconds]")
        return

    timeframe_seconds = int(sys.argv[1])

    dir = os.path.join(settings.DB_PATH, 'poloniex', 'bardata', str(timeframe_seconds))
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)

    pairs_original, pairs = download_poloniex_symbol_list()
    for i, (pair_original, pair) in enumerate(zip(pairs_original, pairs)):
        print('Downloading data for %s (%s/%s)... ' % (pair, i + 1, len(pairs)))

        df = download_poloniex_bardata(pair_original, timeframe_seconds)
        if df is not None:
            pandas_dataframe_to_csv_file(df, os.path.join(settings.DB_PATH, 'poloniex', 'bardata', str(timeframe_seconds), str(pair) + '.csv'))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
