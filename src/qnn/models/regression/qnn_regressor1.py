from typing import TYPE_CHECKING, Optional, List, Dict, Sequence
if TYPE_CHECKING:
    from qnn.market.data import MarketDataTable
    from qnn.targets.regression import IRegressionTarget
    from qnn.market import Symbol, Timeframe
    from qnn.core.ranges import TimestampRange
    from qnn.core.ranges import IndexRange

from abc import ABC, ABCMeta, abstractmethod
import logging

import numpy as np

from qnn.core.parameters import ParametersNode, IntParameter, FloatParameter, ModelChoiceParameter
from .base import IRegressionModel
from qnn.ml.regression import REGRESSION_ML_MODELS_MAP

logger = logging.getLogger(__name__)


class QNNRegressor1(IRegressionModel):
    @staticmethod
    def get_parameters_template() -> ParametersNode:
        return ParametersNode({
            'ml_model': ModelChoiceParameter.from_models_map(REGRESSION_ML_MODELS_MAP),
            'input_sequence_length': IntParameter(31),
            'val_split': FloatParameter(0.05),
        })

    def __init__(self, parameters: ParametersNode, targets: Dict[str, 'IRegressionTarget'], main_timeframe: 'Timeframe', symbols: List['Symbol']):
        super().__init__(parameters, targets, main_timeframe, symbols)

        self._input_sequence_length = parameters['input_sequence_length'].v
        self._val_split = parameters['val_split'].v

    @property
    def _input_keys(self) -> Sequence[str]:
        return [s.name for s in self._symbols]

    @property
    def _input_shapes(self) -> Sequence[Sequence[int]]:
        return [(self._input_sequence_length, 4) for s in self._symbols]

    def _generate_inputs(self, data_table: 'MarketDataTable', row: int) -> Sequence[List[float]]:
        ret = []
        for s in self._symbols:
            open_values = data_table.df[s.name + '.open'][row - self._input_sequence_length:row]
            high_values = data_table.df[s.name + '.high'][row - self._input_sequence_length:row]
            low_values = data_table.df[s.name + '.low'][row - self._input_sequence_length:row]
            close_values = data_table.df[s.name + '.close'][row - self._input_sequence_length:row]

            ov_mean, ov_std = open_values.mean(), open_values.std()
            hv_mean, hv_std = high_values.mean(), high_values.std()
            lv_mean, lv_std = low_values.mean(), low_values.std()
            cv_mean, cv_std = close_values.mean(), close_values.std()

            values = []
            for ov, hv, lv, cv in zip(open_values, high_values, low_values, close_values):
                values.append((ov - ov_mean) / ov_std)
                values.append((hv - hv_mean) / hv_std)
                values.append((lv - lv_mean) / lv_std)
                values.append((cv - cv_mean) / cv_std)
            ret.append(values)
        return ret

    def fit(self, market_data_tables: Dict[str, 'MarketDataTable'], index_range: 'IndexRange') -> None:
        main_market_data_table = market_data_tables[self._main_timeframe.name]

        # Generate training data
        logger.debug('Generating training data...')
        all_inputs = [[] for _ in range(len(self._input_keys))]
        all_targets = {k: [] for k in self.targets.keys()}

        target_lookahead = 1

        for i in range(index_range.begin + self._input_sequence_length - 1, index_range.end - target_lookahead):
            # Generate inputs
            inputs: Sequence[List[float]] = self._generate_inputs(main_market_data_table, i)

            for dest_list, src_list in zip(all_inputs, inputs):
                dest_list.append(src_list)

            # Generate targets
            for k, t in self.targets.items():
                all_targets[k].append(t.generate_target(main_market_data_table, i))

        num_samples = (index_range.end - target_lookahead) - (index_range.begin + self._input_sequence_length - 1) + 1
        train_up_to_index = int((1.0 - self._val_split) * num_samples)

        inputs: Dict[str, np.ndarray] = {k: np.reshape(values, newshape=(-1, *shape)) for k, shape, values in zip(self._input_keys, self._input_shapes, all_inputs)}
        targets: Dict[str, np.ndarray] = {k: np.reshape(v, newshape=(-1, 1)) for k, v in all_targets.items()}

        del all_inputs
        del all_targets

        # Scale targets
        self._targets_minmax = {k: (v.min(), v.max()) for k, v in targets.items()}
        for k, v in targets.items():
            mm = self._targets_minmax[k]
            targets[k] = (v - mm[0]) / (mm[1] - mm[0])

        train_inputs = {k: v[:train_up_to_index] for k, v in inputs.items()}
        train_targets = {k: v[:train_up_to_index] for k, v in targets.items()}

        val_inputs = {k: v[train_up_to_index:] for k, v in inputs.items()}
        val_targets = {k: v[train_up_to_index:] for k, v in targets.items()}

        del inputs
        del targets

        # Create ML model
        logger.debug('Creating ML model...')
        self._ml_model = REGRESSION_ML_MODELS_MAP[self.parameters['ml_model'].v](self.parameters['ml_model'].parameters)

        # Train ML model
        logger.debug('Training ML model...')
        self._ml_model.fit(train_inputs, train_targets, val_inputs, val_targets)
        logger.debug('Creating ML model finished!')

    def predict(self, market_data_tables: Dict[str, 'MarketDataTable'], index_range: 'IndexRange') -> Dict[str, List[float]]:
        # Generate inputs
        all_inputs = [[] for _ in range(len(self._input_keys))]
        for i in range(index_range.begin, index_range.end):
            inputs: Sequence[List[float]] = self._generate_inputs(market_data_tables[self._main_timeframe.name], i)

            for dest_list, src_list in zip(all_inputs, inputs):
                dest_list.append(src_list)

        inputs: Dict[str, np.ndarray] = {k: np.reshape(values, newshape=(-1, *shape)) for k, shape, values in zip(self._input_keys, self._input_shapes, all_inputs)}

        # Query ML model
        preds = self._ml_model.predict(inputs)

        # Scale predictions back to target scale
        for k, values in preds.items():
            mm = self._targets_minmax[k]
            preds[k] = [v * (mm[1] - mm[0]) + mm[0] for v in values]

        return preds
