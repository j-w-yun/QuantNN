from typing import List, Optional, Dict
from abc import ABCMeta, abstractmethod

from qnn.core.ranges import TimestampRange


class IDataAPI(metaclass=ABCMeta):
    def __init__(self):
        pass

    def get_symbols_list(self) -> Optional[List[str]]:
        raise NotImplementedError
