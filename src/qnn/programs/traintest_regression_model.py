"""
Train and test a regression model on a regression problem.
"""

import os
import sys
import logging
import math

import numpy as np
from sklearn import metrics

from qnn import settings
from qnn.core.serialization import load_dict_from_yaml_file
from qnn.core.parameters import ParametersNode, get_parameters_template
from qnn.core.datetime import find_index_range
from qnn.problems import RegressionProblem
from qnn.market.utilities import create_market_from_file
from qnn.targets.regression import REGRESSION_TARGETS_MAP
from qnn.models.regression import REGRESSION_MODELS_MAP
from qnn.market.data.utilities import load_symbols_bardata, create_market_data_table
from qnn.core.confusion_matrix import ConfusionMatrix

logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) != 3:
        print('Please provide: [regression problem name] [regression model name]')
        return

    problem_name = sys.argv[1]
    model_name = sys.argv[2]

    # Create market
    logger.info('Creating market...')
    market = create_market_from_file(os.path.join(settings.DB_PATH, 'market.yaml'))

    # Create problem
    logger.info('Creating problem...')
    problem_info = load_dict_from_yaml_file(os.path.join('problems', 'regression', problem_name + '.yaml'))
    problem = RegressionProblem.from_dict(market, problem_info)

    Target = REGRESSION_TARGETS_MAP[problem.target_name]
    targets = {s.name: Target(problem.target_parameters, s) for s in problem.target_symbols}
    target_lookahead = 1

    # Load data required by problem
    logger.info('Loading data...')
    data_symbols = set(problem.symbols + problem.target_symbols)
    bardata = load_symbols_bardata(problem.timeframe, list(data_symbols))

    market_data_table = create_market_data_table(problem.timeframe, bardata)

    data = {problem.timeframe.name: market_data_table}

    # Find index ranges for train and test sets
    train_index_range = find_index_range(market_data_table.index, problem.train_range)
    test_index_range = find_index_range(market_data_table.index, problem.test_range)
    test_index_range.end -= target_lookahead

    # Create ML model
    logger.info('Creating model...')
    model_info = load_dict_from_yaml_file(os.path.join('models', 'regression', model_name + '.yaml'))
    Model = REGRESSION_MODELS_MAP[model_info['model']]
    model_parameters = get_parameters_template(Model)
    model_parameters.update_from_values_dict(model_info)
    model = Model(model_parameters, targets, problem.timeframe, problem.symbols)

    # Train model
    logger.info('Fitting model...')
    model.fit(data, train_index_range)

    # Test model
    logger.info('Testing model...')
    predictions_map = model.predict(data, test_index_range)

    all_cms = {k: ConfusionMatrix() for k in targets.keys()}
    all_predictions = {k: [] for k in targets.keys()}
    all_targets = {k: [] for k in targets.keys()}
    for prediction_index, i in enumerate(range(test_index_range.begin, test_index_range.end)):
        for k, target in targets.items():
            y = target.generate_target(market_data_table, i)
            y_hat = predictions_map[k][prediction_index]

            all_predictions[k].append(y_hat)
            all_targets[k].append(y)

            # Update confusion matrix
            if y_hat > 0.0:
                if y > 0.0:
                    all_cms[k].tp += 1
                else:
                    all_cms[k].fp += 1
            else:
                if y <= 0.0:
                    all_cms[k].tn += 1
                else:
                    all_cms[k].fn += 1

    logger.info('Printing stats...')

    print('Confusion matrices:')
    for k, cm in all_cms.items():
        print('\t%s CM: actual_positives_ratio=%.4f, prec=%.4f (n=%d), acc=%.2f%%, n=%d' % (k, cm.actual_positives_ratio, cm.precision, cm.precision_n, cm.accuracy * 100.0, cm.n))

    print('Regression scores:')
    for k in targets.keys():
        print('\t%s stats: mse=%.4f, r2=%.4f' % (k, metrics.mean_squared_error(all_targets[k], all_predictions[k]), metrics.r2_score(all_targets[k], all_predictions[k])))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
