from typing import List, Dict, Type

from .base import INetworkOptimizer
from .adam import AdamOptimizer

NN_OPTIMIZERS: List[Type[INetworkOptimizer]] = [
    AdamOptimizer,
]

NN_OPTIMIZERS_MAP: Dict[str, Type[INetworkOptimizer]] = {v.__name__: v for v in NN_OPTIMIZERS}
