import datetime

import pandas as pd
import numpy as np

from qnn.market import Market, Timeframe, Symbol, Exchange, Unit
from qnn.market.data import MarketDataTable
from qnn.targets.seq import PriceSeqTarget


def test_price_seq_target():
    # Create testing data
    market = Market()

    a_unit = Unit(1, 'A')
    b_unit = Unit(2, 'B')
    market.add_unit(a_unit)
    market.add_unit(b_unit)

    test_timeframe = Timeframe(1, '1D', datetime.timedelta(days=1))
    market.add_timeframe(test_timeframe)

    test_exchange = Exchange(market, 1, 'TestEx')
    market.add_exchange(test_exchange)

    test_symbol = Symbol(1, 'TEST', test_exchange, a_unit, b_unit)
    test_exchange.add_symbol(test_symbol)

    df = pd.DataFrame({
        'TEST.open': list(range(10)),
        'TEST.high': list(range(10)),
        'TEST.low': list(range(10)),
        'TEST.close': list(range(10)),
    }, dtype=np.int32)

    data_table = MarketDataTable(test_timeframe, [test_symbol,], df)

    # Create target
    target = PriceSeqTarget(PriceSeqTarget.get_parameters_template(), test_symbol, 4)

    # Generate target data
    t = target.generate_target(data_table, 2)

    # Reshape target data to proper shape and turn it into a numpy ndarray
    t = np.reshape(t[0], newshape=target.output_shapes[0])

    # Verify t is what we expect as output
    expected = np.array([
        [3, 3, 3, 3],
        [4, 4, 4, 4],
        [5, 5, 5, 5],
        [6, 6, 6, 6],
    ])

    assert (t == expected).all()
