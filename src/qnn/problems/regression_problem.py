from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from qnn.market import Market, Timeframe, Symbol

from qnn.core.ranges import TimestampRange
from qnn.core.parameters import ParametersNode, get_parameters_template
from qnn.targets.regression import REGRESSION_TARGETS_MAP


class RegressionProblem(object):
    def __init__(self,
                 timeframe: 'Timeframe',
                 symbols: List['Symbol'],
                 target_symbols: List['Symbol'],
                 train_range: TimestampRange, test_range: TimestampRange,
                 target_name: str, target_parameters: ParametersNode):
        self._timeframe = timeframe
        self._symbols = symbols
        self._target_symbols = target_symbols
        self._train_range = train_range
        self._test_range = test_range
        self._target_name = target_name
        self._target_parameters = target_parameters

    @property
    def timeframe(self) -> 'Timeframe':
        return self._timeframe

    @property
    def symbols(self) -> List['Symbol']:
        return self._symbols

    @property
    def target_symbols(self) -> List['Symbol']:
        return self._target_symbols

    @property
    def train_range(self) -> TimestampRange:
        return self._train_range

    @property
    def test_range(self) -> TimestampRange:
        return self._test_range

    @property
    def target_name(self) -> str:
        return self._target_name

    @property
    def target_parameters(self) -> ParametersNode:
        return self._target_parameters

    def to_dict(self) -> dict:
        return {
            'timeframe': self._timeframe.name,
            'symbols': [s.name for s in self._symbols],
            'target_symbols': [s.name for s in self._target_symbols],
            'train_range': self._train_range.to_dict(),
            'test_range': self._test_range.to_dict(),
            'target_name': self._target_name,
            'target_parameters': self._target_parameters.to_values_dict(),
        }

    @staticmethod
    def from_dict(market: 'Market', d: dict) -> 'RegressionProblem':
        target_parameters = get_parameters_template(REGRESSION_TARGETS_MAP[d['target_name']])
        target_parameters.update_from_values_dict(d['target_parameters'])

        return RegressionProblem(market.timeframes_by_name[d['timeframe']],
                              [market.symbols_by_name[s] for s in d['symbols']],
                              [market.symbols_by_name[s] for s in d['target_symbols']],
                              TimestampRange.from_dict(d['train_range']),
                              TimestampRange.from_dict(d['test_range']),
                              d['target_name'],
                              target_parameters)
