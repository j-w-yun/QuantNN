from typing import TYPE_CHECKING, Dict, List
if TYPE_CHECKING:
    from .market import Market
    from .symbol import Symbol


class Exchange(object):
    def __init__(self, market: 'Market', id: int, name: str):
        self._market = market
        self._id: int = id
        self._name: str = name
        self._symbols_by_id: Dict[int, 'Symbol'] = {}
        self._symbols_by_name: Dict[str, 'Symbol'] = {}

    @property
    def market(self) -> 'Market':
        return self._market

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def symbols(self) -> List['Symbol']:
        return list(self._symbols_by_id.values())

    @property
    def symbols_by_id(self) -> Dict[int, 'Symbol']:
        return self._symbols_by_id

    @property
    def symbols_by_name(self) -> Dict[str, 'Symbol']:
        return self._symbols_by_name

    # TODO: symbols by base unit and symbols by counter unit properties

    def add_symbol(self, symbol: 'Symbol'):
        self._market._add_symbol(self, symbol)

        if symbol.id in self._symbols_by_id:
            raise RuntimeError('Symbol with id=%d already in exchange object.' % symbol.id)

        if symbol.name in self._symbols_by_name:
            raise RuntimeError('Symbol with name="%s" already in exchange object.' % symbol.name)

        self._symbols_by_id[symbol.id] = symbol
        self._symbols_by_name[symbol.name] = symbol
