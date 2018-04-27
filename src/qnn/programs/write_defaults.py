"""
Write default .yaml configuration fiels for each seq model.
"""

import logging
import os

from qnn import settings
from qnn.models.seq import SEQ_MODELS_MAP
from qnn.core.serialization import dict_to_yaml_file

logger = logging.getLogger(__name__)


def main():
    logging.info('Writing seq model defaults...')
    for model_name, Model in SEQ_MODELS_MAP.items():
        logging.debug('Writing seq model defaults for "%s"...' % model_name)
        dict_to_yaml_file(os.path.join('models', 'seq', model_name + '.yaml'), {'model': model_name, **Model.get_parameters_template().to_dict()})

    logging.info('Done!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
