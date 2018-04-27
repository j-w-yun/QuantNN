from typing import TYPE_CHECKING, Dict, List
if TYPE_CHECKING:
    from .timeframe import Timeframe
    from .unit import Unit
    from .exchange import Exchange
    from .symbol import Symbol


class Market(object):
    def __init__(self):
        self._timeframes_by_id: Dict[int, 'Timeframe'] = {}
        self._timeframes_by_name: Dict[str, 'Timeframe'] = {}

        self._units_by_id: Dict[int, 'Unit'] = {}
        self._units_by_name: Dict[str, 'Unit'] = {}

        self._exchanges_by_id: Dict[int, 'Exchange'] = {}
        self._exchanges_by_name: Dict[str, 'Exchange'] = {}

        self._symbols_by_id: Dict[int, 'Symbol'] = {}
        self._symbols_by_name: Dict[str, 'Symbol'] = {}

    @property
    def timeframes(self) -> List['Timeframe']:
        return list(self._timeframes_by_id.values())

    @property
    def timeframes_by_id(self) -> Dict[int, 'Timeframe']:
        return self._timeframes_by_id

    @property
    def timeframes_by_name(self) -> Dict[str, 'Timeframe']:
        return self._timeframes_by_name

    @property
    def units(self) -> List['Unit']:
        return list(self._units_by_id.values())

    @property
    def units_by_id(self) -> Dict[int, 'Unit']:
        return self._units_by_id

    @property
    def units_by_name(self) -> Dict[str, 'Unit']:
        return self._units_by_name

    @property
    def exchanges(self) -> List['Exchange']:
        return list(self._exchanges_by_id.values())

    @property
    def exchanges_by_id(self) -> Dict[int, 'Exchange']:
        return self._exchanges_by_id

    @property
    def exchanges_by_name(self) -> Dict[str, 'Exchange']:
        return self._exchanges_by_name

    @property
    def symbols(self) -> List['Symbol']:
        return list(self._symbols_by_id.values())

    @property
    def symbols_by_id(self) -> Dict[int, 'Symbol']:
        return self._symbols_by_id

    @property
    def symbols_by_name(self) -> Dict[str, 'Symbol']:
        return self._symbols_by_name

    def add_timeframe(self, timeframe: 'Timeframe'):
        if timeframe.id in self._timeframes_by_id:
            raise RuntimeError('Timeframe with id=%d already in market object.' % timeframe.id)

        if timeframe.name in self._timeframes_by_name:
            raise RuntimeError('Timeframe with name="%s" already in market object.' % timeframe.name)

        self._timeframes_by_id[timeframe.id] = timeframe
        self._timeframes_by_name[timeframe.name] = timeframe

    def add_unit(self, unit: 'Unit'):
        if unit.id in self._units_by_id:
            raise RuntimeError('Unit with id=%d already in market object.' % unit.id)

        if unit.name in self._units_by_name:
            raise RuntimeError('Unit with name=%s already in market object.' % unit.name)

        self._units_by_id[unit.id] = unit
        self._units_by_name[unit.name] = unit

    def add_exchange(self, exchange: 'Exchange'):
        if exchange.id in self._exchanges_by_id:
            raise RuntimeError('Exchange with id=%d already in market object.' % exchange.id)

        if exchange.name in self._exchanges_by_name:
            raise RuntimeError('Exchange with name="%s" already in market object.' % exchange.name)

        self._exchanges_by_id[exchange.id] = exchange
        self._exchanges_by_name[exchange.name] = exchange

    def _add_symbol(self, exchange: 'Exchange', symbol: 'Symbol'):
        """
        Don't ever call this method directly, it is to be called from exchange when you add a symbol there and this process is automated.
        :param exchange:
        :param symbol:
        :return:
        """
        if symbol.id in self._symbols_by_id:
            raise RuntimeError('Symbol with id=%d already exists in market object.' % symbol.id)

        if symbol.name in self._symbols_by_name:
            raise RuntimeError('Symbol with id=%d already exists in market object.' % symbol.name)

        self._symbols_by_id[symbol.id] = symbol
        self._symbols_by_name[symbol.name] = symbol
