from typing import List, Dict, Type
from .base import IRegressionMLModel
from .keras_lstm1 import KerasLSTM1

REGRESSION_ML_MODELS: List[Type['IRegressionMLModel']] = [
    KerasLSTM1,
]

REGRESSION_ML_MODELS_MAP: Dict[str, Type['IRegressionMLModel']] = {v.__name__: v for v in REGRESSION_ML_MODELS}
