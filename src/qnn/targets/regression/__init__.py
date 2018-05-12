from typing import List, Dict, Type
from .base import IRegressionTarget
from .price_pct_change_reg_target import PricePctChangeRegressionTarget

REGRESSION_TARGETS: List[Type['IRegressionTarget']] = [
    PricePctChangeRegressionTarget,
]

REGRESSION_TARGETS_MAP: Dict[str, Type['IRegressionTarget']] = {v.__name__: v for v in REGRESSION_TARGETS}
