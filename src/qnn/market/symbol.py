from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
    from .exchange import Exchange
    from .unit import Unit


class Symbol(object):
    def __init__(self, id: int, name: str, exchange: 'Exchange', base_unit: 'Unit', counter_unit: 'Unit'):
        self._id: int = id
        self._name: str = name
        self._exchange: 'Exchange' = exchange
        self._base_unit: 'Unit' = base_unit
        self._counter_unit: 'Unit' = counter_unit

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def exchange(self) -> 'Exchange':
        return self._exchange

    @property
    def base_unit(self) -> 'Unit':
        return self._base_unit

    @property
    def counter_unit(self) -> 'Unit':
        return self._counter_unit
