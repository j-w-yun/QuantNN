from typing import List, Dict, Type
from .base import ISeqModel
from .qnn_seq1 import QNNSeq1

SEQ_MODELS: List[Type['ISeqModel']] = [
    QNNSeq1,
]

SEQ_MODELS_MAP: Dict[str, Type['ISeqModel']] = {v.__name__: v for v in SEQ_MODELS}
