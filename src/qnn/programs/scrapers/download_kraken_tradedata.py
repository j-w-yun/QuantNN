from typing import List, Optional, Tuple
import logging
import os
import signal
import sys
import time
import datetime
import pytz

from qnn import settings
from qnn.data import EventsDataFile
from qnn.data.api.kraken import KrakenDataAPI


logger = logging.getLogger(__name__)

current_data_file = None
current_last = None

def signal_handler(signal, frame):
    print('You pressed Ctrl+C! Quitting gracefully...')

    if current_data_file is not None:
        if current_data_file.is_open:
            current_data_file.flush()
        if current_last is not None:
            current_data_file.set_extra_data({'last': current_last})

    print('Bye!')
    sys.exit(0)


def main():
    dir = os.path.join(settings.DATA_PATH, 'kraken', 'trades')
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)

    kraken = KrakenDataAPI()

    symbols = kraken.get_symbols_list()

    signal.signal(signal.SIGINT, signal_handler)

    logger.info('Downloading trades data for %s symbols...' % len(symbols))

    for symbol_i, symbol in enumerate(symbols):
        logger.info('Downloading trades data for "%s"...' % symbol)

        if not os.path.isdir(os.path.join(dir, symbol)):
            os.mkdir(os.path.join(dir, symbol))

        data_file = EventsDataFile(os.path.join(dir, symbol))

        global current_data_file
        current_data_file = data_file

        extra_data = data_file.get_extra_data()
        if extra_data is not None:
            since = int(extra_data['last'])
        else:
            since = 0

        global current_last
        current_last = since

        t = None
        for trade, last in kraken.download_historical_trades(symbol, since):
            dt = datetime.datetime.utcfromtimestamp(float(trade[2]))
            current_last = last

            if t is None or time.time() - t >= 120:
                if data_file.is_open:
                    data_file.flush()
                data_file.set_extra_data({'last': last})

                logger.info('Downloading trades data for "%s" (%d/%d): currently at %s' % (symbol, symbol_i + 1, len(symbols), dt))
                t = time.time()

            if not data_file.is_open:
                data_file.open(dt)

            data_file.append(dt, trade)

        if data_file.is_open:
            data_file.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
