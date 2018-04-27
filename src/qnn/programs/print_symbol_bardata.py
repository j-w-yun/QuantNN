"""
Print information.
"""

import os

from qnn import settings
from qnn.market.utilities import create_market_from_file
from qnn.market.data.utilities import load_symbols_bardata, create_market_data_table


def main():
    market = create_market_from_file(os.path.join(settings.DB_PATH, 'market.yaml'))

    bardata = load_symbols_bardata(market.timeframes_by_name['1D'], [
        market.exchanges_by_name['poloniex'].symbols_by_name['POLO.AMP_BTC'],
        market.exchanges_by_name['poloniex'].symbols_by_name['POLO.BCN_BTC'],
    ])

    for s, data in bardata.items():
        print(s.name + ' range: ', data.index[0], data.index[-1])

if __name__ == '__main__':
    main()
