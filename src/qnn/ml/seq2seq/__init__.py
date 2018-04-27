from typing import List, Dict, Type
from .base import ISeq2SeqModel
from .encoder_decoder import EncoderDecoderModel

SEQ2SEQ_ML_MODELS: List[Type['ISeq2SeqModel']] = [
    EncoderDecoderModel,
]

SEQ2SEQ_ML_MODELS_MAP: Dict[str, Type['ISeq2SeqModel']] = {v.__name__: v for v in SEQ2SEQ_ML_MODELS}
