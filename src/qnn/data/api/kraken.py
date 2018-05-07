from typing import List, Optional, Dict

import logging
import time
import json

from qnn.core.ranges import TimestampRange
from qnn.core.web import http_get

from .base import IDataAPI

logger = logging.getLogger(__name__)


class KrakenDataAPI(IDataAPI):
    def __init__(self):
        super().__init__()

        self._endpoint = 'https://api.kraken.com/0'

    def get_symbols_list(self) -> Optional[List[str]]:
        data = http_get('%s/public/AssetPairs' % self._endpoint)
        if data is None:
            return None

        data = json.loads(data)
        return [k for k in data['result']]

    def download_historical_trades(self, symbol: str, since=0):
        while True:
            logger.debug('Downloading since=%d' % since)

            data = http_get('%s/public/Trades' % self._endpoint, params={'pair': symbol, 'since': since}, max_retries=15)
            if data is None:
                # Oh no!
                time.sleep(3)
                continue  # Try again infinitely

            data = json.loads(data)

            if not 'result' in data:
                print(data)
                time.sleep(10)
                continue

            new_since = int(data['result']['last'])

            data = data['result'][symbol]

            for trade in data:
                yield trade, new_since

            if new_since == since:  # No more new trades available
                break

            since = new_since
            time.sleep(1)

    def download_historical_bars(self, symbol: str, rangev: Optional[TimestampRange]=None):
        raise NotImplementedError
