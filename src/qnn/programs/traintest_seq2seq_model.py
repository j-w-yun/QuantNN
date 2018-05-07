"""
Train and test a seq2seq ML model on a seq2seq problem.
"""

import os
import sys
import logging

import numpy as np

from qnn import settings
from qnn.core.serialization import load_dict_from_yaml_file
from qnn.core.parameters import ParametersNode, get_parameters_template
from qnn.core.datetime import find_index_range
from qnn.problems import Seq2SeqProblem
from qnn.market.utilities import create_market_from_file
from qnn.targets.seq import SEQ_TARGETS, SEQ_TARGETS_MAP
from qnn.models.seq import SEQ_MODELS, SEQ_MODELS_MAP
from qnn.market.data.utilities import load_symbols_bardata, create_market_data_table

logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) != 3:
        print('Please provide: [seq2seq problem name] [seq model name]')
        return

    problem_name = sys.argv[1]
    model_name = sys.argv[2]

    # Create market
    logger.info('Creating market...')
    market = create_market_from_file(os.path.join(settings.DB_PATH, 'market.yaml'))

    # Create problem
    logger.info('Creating problem...')
    problem_info = load_dict_from_yaml_file(os.path.join('problems', 'seq2seq', problem_name + '.yaml'))
    problem = Seq2SeqProblem.from_dict(market, problem_info)

    Target = SEQ_TARGETS_MAP[problem.target_name]
    target = Target(problem.target_parameters, problem.target_symbols[0], problem.target_seqlen)

    # Load data required by problem
    logger.info('Loading data...')
    data_symbols = set(problem.symbols + problem.target_symbols)
    bardata = load_symbols_bardata(problem.timeframe, list(data_symbols))

    market_data_table = create_market_data_table(problem.timeframe, bardata)

    data = {problem.timeframe.name: market_data_table}

    # Find index ranges for train and test sets
    train_index_range = find_index_range(market_data_table.index, problem.train_range)
    test_index_range = find_index_range(market_data_table.index, problem.test_range)
    test_index_range.end -= problem.target_seqlen

    # Create ML model
    logger.info('Creating model...')
    model_info = load_dict_from_yaml_file(os.path.join('models', 'seq', model_name + '.yaml'))
    Model = SEQ_MODELS_MAP[model_info['model']]
    model_parameters = get_parameters_template(Model)
    model_parameters.update_from_values_dict(model_info)
    model = Model(model_parameters, target, problem.timeframe, problem.symbols)

    # Train model
    logger.info('Fitting model...')
    model.fit(data, train_index_range)

    # Test model
    logger.info('Testing model...')
    predictions_map = model.predict(data, test_index_range)
    predictions = [predictions_map[k] for k in target.output_keys]

    rmse_mean = 0.0
    rmse_n = 0
    for prediction_index, i in enumerate(range(test_index_range.begin, test_index_range.end)):
        targets = target.generate_target(market_data_table, i)

        for prediction, target_seq, target_key, target_output_shape in zip(predictions, targets, target.output_keys, target.output_shapes):
            predicted_seq = prediction[prediction_index]

            # Turn target into numpy array
            target_seq = np.reshape(target_seq, newshape=target_output_shape)

            # Compute rmse for this sequence forecast
            rmse = np.sqrt((predicted_seq - target_seq) ** 2).mean()

            rmse_mean += rmse
            rmse_n += 1

    print('Test set RMSE (using mean of all targets and predictions): %g (n=%d)' % ((rmse_mean / rmse_n), rmse_n))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
