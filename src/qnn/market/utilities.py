import yaml

from qnn.core.datetime import duration_from_string
from .market import Market
from .timeframe import Timeframe
from .unit import Unit
from .exchange import Exchange
from .symbol import Symbol


def create_market_from_file(filename: str):
    with open(filename, 'r') as f:
        d = yaml.safe_load(f)
    return create_market_from_dict(d)


def create_market_from_dict(d: dict):
    market = Market()

    for name, v in d['timeframes'].items():
        duration = duration_from_string(v['duration'])
        market.add_timeframe(Timeframe(int(v['id']), name, duration))

    for name, v in d['units'].items():
        market.add_unit(Unit(int(v['id']), name))

    for name, v in d['exchanges'].items():
        exchange = Exchange(market, int(v['id']), name)
        for symbol_name, symbol_v in v['symbols'].items():
            symbol = Symbol(int(symbol_v['id']), symbol_name, exchange, market.units_by_name[symbol_v['base_unit']], market.units_by_name[symbol_v['counter_unit']])
            exchange.add_symbol(symbol)
        market.add_exchange(exchange)

    return market
