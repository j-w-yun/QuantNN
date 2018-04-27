from typing import TYPE_CHECKING, Optional, List, Dict, Sequence
if TYPE_CHECKING:
    from qnn.market.data import MarketDataTable
    from qnn.targets.seq import ISeqTarget
    from qnn.market import Symbol, Timeframe
    from qnn.core.ranges import TimestampRange
    from qnn.core.ranges import IndexRange

import logging

import numpy as np

from qnn.core.parameters import ParametersNode, IntParameter, FloatParameter, ModelChoiceParameter
from .base import ISeqModel
from qnn.ml.seq2seq import SEQ2SEQ_ML_MODELS, SEQ2SEQ_ML_MODELS_MAP

logger = logging.getLogger(__name__)


class QNNSeq1(ISeqModel):
    @staticmethod
    def get_parameters_template() -> ParametersNode:
        return ParametersNode({
            'ml_model': ModelChoiceParameter.from_models_map(SEQ2SEQ_ML_MODELS_MAP),
            'input_sequence_length': IntParameter(20),
            'val_split': FloatParameter(0.1),
            # TODO: input sequence scaler
        })

    def __init__(self, parameters: ParametersNode, target: 'ISeqTarget', main_timeframe: 'Timeframe', symbols: List['Symbol']):
        super().__init__(parameters, target, main_timeframe, symbols)

        self._input_sequence_length = parameters['input_sequence_length'].v
        self._val_split = parameters['val_split'].v
        self._target_sequence_length = self._target.seqlen

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

            values = []
            for ov, hv, lv, cv in zip(open_values, high_values, low_values, close_values):
                values.append(ov)
                values.append(hv)
                values.append(lv)
                values.append(cv)
            ret.append(values)
        return ret

    def fit(self, market_data_tables: Dict[str, 'MarketDataTable'], index_range: 'IndexRange') -> None:
        target_sequence_length = self._target.seqlen
        main_market_data_table = market_data_tables[self._main_timeframe.name]

        # Generate training data
        logger.debug('Generating training data...')
        all_inputs = [[] for _ in range(len(self._input_keys))]
        all_targets = [[] for _ in range(len(self._target.output_keys))]

        for i in range(index_range.begin + self._input_sequence_length - 1, index_range.end - self._target_sequence_length):
            # Generate inputs
            inputs: Sequence[List[float]] = self._generate_inputs(main_market_data_table, i)

            for dest_list, src_list in zip(all_inputs, inputs):
                dest_list.append(src_list)

            # Generate targets
            targets: Sequence[List[float]] = self._target.generate_target(main_market_data_table, i)
            for dest_list, src_list in zip(all_targets, targets):
                dest_list.append(src_list)

        num_samples = (index_range.end - self._target_sequence_length) - (index_range.begin + self._input_sequence_length - 1) + 1
        train_up_to_index = int((1.0 - self._val_split) * num_samples)

        inputs: Dict[str, np.ndarray] = {k: np.reshape(values, newshape=(-1, *shape)) for k, shape, values in zip(self._input_keys, self._input_shapes, all_inputs)}
        targets: Dict[str, np.ndarray] = {k: np.reshape(values, newshape=(-1, *shape)) for k, shape, values in zip(self._target.output_keys, self._target.output_shapes, all_targets)}

        del all_inputs
        del all_targets

        train_inputs = {k: v[:train_up_to_index] for k, v in inputs.items()}
        train_targets = {k: v[:train_up_to_index] for k, v in targets.items()}

        val_inputs = {k: v[train_up_to_index:] for k, v in inputs.items()}
        val_targets = {k: v[train_up_to_index:] for k, v in targets.items()}

        del inputs
        del targets

        # Create ML model
        logger.debug('Creating ML model...')
        self._ml_model = SEQ2SEQ_ML_MODELS_MAP[self.parameters['ml_model'].v](self.parameters['ml_model'].parameters)

        # Train ML model
        logger.debug('Training ML model...')
        self._ml_model.fit(train_inputs, train_targets, val_inputs, val_targets)
        logger.debug('Creating ML model finished!')

    def predict(self, market_data_tables: Dict[str, 'MarketDataTable'], index_range: 'IndexRange') -> Dict[str, np.ndarray]:
        # Generate inputs
        all_inputs = [[] for _ in range(len(self._input_keys))]
        for i in range(index_range.begin, index_range.end):
            inputs: Sequence[List[float]] = self._generate_inputs(market_data_tables[self._main_timeframe.name], i)

            for dest_list, src_list in zip(all_inputs, inputs):
                dest_list.append(src_list)

        inputs: Dict[str, np.ndarray] = {k: np.reshape(values, newshape=(-1, *shape)) for k, shape, values in zip(self._input_keys, self._input_shapes, all_inputs)}

        # Query ML model
        return self._ml_model.predict(inputs)
