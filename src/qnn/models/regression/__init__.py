from typing import Type, List, Dict
from .base import IRegressionModel
from .qnn_regressor1 import QNNRegressor1

REGRESSION_MODELS: List[Type[IRegressionModel]] = [
    QNNRegressor1,
]

REGRESSION_MODELS_MAP: Dict[str, Type[IRegressionModel]] = {v.__name__: v for v in REGRESSION_MODELS}
