import copy
import json
import sys
import time

import requests
from requests.exceptions import ReadTimeout

from cacheable import Cacheable, StartKeyNotFoundError, EndKeyNotFoundError,\
    StartEndKeysNotFoundError
import dataset_list
import helper
import numpy as np


class Kraken(Cacheable):
    """https://support.kraken.com/hc/en-us/articles/218198197-How-to-pull-all-trade-data-using-the-Kraken-REST-API
    """

    __GRANULARITY = 60

    def __init__(self, save_directory):
        super(Kraken, self).__init__(save_directory)
        self.url = 'https://api.kraken.com/0'
        self.timeout = 600

    def __get(self, path, payload, max_retries=100):
        r = None

        # Invalid format packet
        for retries in range(max_retries):
            try:
                r = requests.get(
                    self.url + path, params=payload, timeout=self.timeout)

                # HTTP not ok
                while not r.ok:
                    print('Kraken | {}'.format(r))
                    time.sleep(3 * retries)
                    r = requests.get(
                        self.url + path, params=payload, timeout=self.timeout)

                # Kraken error
                while len(r.json()['error']) > 0:
                    print('Kraken | {}'.format(r.json()['error']))
                    time.sleep(3 * retries)
                    r = requests.get(
                        self.url + path, params=payload, timeout=self.timeout)
            except:
                time.sleep(60)
                continue
            break

        return r.json()

    def _get_trades(self, pair, start, end):
        trades = []

        params = {'pair': pair,
                  'since': int(start * 1000000000)}
        r = self.__get('/public/Trades', params)

        trades.extend(r['result'][pair])  # old -> new
        last = int(r['result']['last']) / 1000000000

        while last < end:
            params = {'pair': pair,
                      'since': int(last * 1000000000)}
            r = self.__get('/public/Trades', params)
            last_ = int(r['result']['last']) / 1000000000
            if last_ == last:
                print('Krak Trade {}\t| No future trades after {}'.format(
                    pair, last))
                break
            last = last_
            trades.extend(r['result'][pair])

        index = 0  # find last one to include
        for trade in trades:
            timestamp = trade[2]
            if timestamp > end:
                break
            index += 1
        return trades[:index]

    def download_trades(self, **kwargs):
        product = kwargs['product']
        start = kwargs['start']
        end = kwargs['end']

        filename = product + '_trades'
        MAX_SIZE = 300

        print('Krak Trade {}\t| Requested data from {} to {}'.format(
            product, start, end))

        # initial carry forward value
        last_seen_row = [
            1 if label == dataset_list.IS_CARRIED else 0 for label in dataset_list.LABELS[dataset_list.POLO_TRADE]]

        # load if present
        try:
            self.check_cache(filename, (start, end - self.__GRANULARITY))
            print('Krak Trade {}\t| Cached data is complete'.format(product))
            return filename
        except FileNotFoundError:
            print('Krak Trade {}\t| Cache for does not exist. Starting from time {}'.format(
                product, start))
        except EndKeyNotFoundError:
            last_seen_row = self.get_last_cache(filename)
            latest_time = int(last_seen_row[0])
            start = latest_time + self.__GRANULARITY
            print('Krak Trade {}\t| Continuing from latest time {}'.format(
                product, latest_time))
        except (StartKeyNotFoundError, StartEndKeysNotFoundError) as e:
            # TODO: Handle importing disjoint historical data
            print(e)
            sys.exit()

        slice_range = self.__GRANULARITY * MAX_SIZE
        slice_start = start
        while slice_start != end:
            slice_end = min(slice_start + slice_range, end)

            # fetch slice
            t = self._get_trades(product, slice_start, slice_end)

            if len(t) == 0:
                print('Krak Trade {}\t| Returned data for {} -> {} is length zero'.format(
                    product, slice_start, slice_end))
                for timestamp in range(slice_start, slice_end, self.__GRANULARITY):
                    last_seen_row[0] = timestamp  # correct time

                    # do not carry non-continuous values
                    last_seen_row[5] = 0  # volume
                    last_seen_row[6] = 0  # num trades

                    last_seen_row[-1] = 1  # is_carried == True
                    self.append_cache(filename, last_seen_row)
                # next slice
                slice_start = slice_end
                continue

            # carry forward and save
            current_index = 0
            trade = t[current_index]
            for timestamp in range(slice_start, slice_end, self.__GRANULARITY):
                rate_list = []
                total_list = []
                weighted_sum = []

                while(trade[2] <= timestamp and
                      trade[2] > (timestamp - self.__GRANULARITY)):
                    trade = t[current_index]

                    rate_list.append(float(trade[0]))
                    total_list.append(float(trade[1]))
                    weighted_sum.append(total_list[-1] * rate_list[-1])

                    # break loop if last index is reached
                    if (current_index + 1) == len(t):
                        break
                    current_index += 1

                if np.sum(total_list) != 0:
                    row = [timestamp,
                           np.min(rate_list),  # low
                           np.max(rate_list),  # high
                           rate_list[0],  # open
                           rate_list[-1],  # close
                           np.sum(total_list),  # volume
                           len(total_list),  # num trades
                           np.sum(weighted_sum) / np.sum(total_list),  # w avg
                           0]
                else:
                    last_seen_row[0] = timestamp  # correct time

                    # do not carry non-continuous values
                    last_seen_row[5] = 0  # volume
                    last_seen_row[6] = 0  # num trades

                    last_seen_row[-1] = 1  # is_carried == True
                    row = last_seen_row

                # last seen
                last_seen_row = copy.deepcopy(row)

                # save
                self.append_cache(filename, row)

            # console print
            progress = 100 * (slice_end - start) / (end - start)
            print('Krak Trade {}\t| {:6.2f}% | {} -> {} | {} -> {} | {}'.format(
                  product,
                  progress,
                  int(t[0][2]),
                  int(t[-1][2]),
                  helper.unix_to_iso(int(t[0][2])),
                  helper.unix_to_iso(int(t[-1][2])),
                  len(t)))

            # next slice
            slice_start = slice_end

        return filename

    def get_trades(self, **kwargs):
        start = kwargs['start']
        end = kwargs['end']

        # download trades
        filename = self.download_trades(**kwargs)

        # return window
        r = self.get_cache(filename, range_keys=(start, end - 60))
        return r

    def validate_trades(self, filename):
        filename += '_trades'

        r = self.get_cache(filename)
        current_time = r[0][0]
        print('Krak {}\t| Start processing from time {}'.format(
            filename, current_time))

        new_data = []
        for row in r:
            # check for invalid values
            for elem in row:
                if elem is None:
                    raise ValueError('Krak {}\t| Invalid value {} encountered in row {}'.format(
                        filename, elem, row))
            # check for odd number of columns
            if len(row) != len(dataset_list.LABELS[dataset_list.POLO_TRADE]):
                raise ValueError('Krak {}\t| Invalid number of columns {}. Expected {}'.format(
                    filename, len(row), len(dataset_list.LABELS[dataset_list.POLO_TRADE])))
            # check for invalid self.__GRANULARITY
            if row[0] % 60 != 0:
                raise ValueError('Krak {}\t| Invalid interval of time for row {}'.format(
                    filename, row))
            # keep if correct time
            elif row[0] == current_time:
                new_data.append(row)
                current_time += self.__GRANULARITY
            # identified lack of order in data
            # possibly a result of running parallel downloads
            elif row[0] < current_time:
                # do not increment current_time
                print('Krak {}\t| Duplicate time {} found. Expected time {}'.format(
                    filename, row[0], current_time))

        if len(new_data) != len(r):
            self.set_cache(filename, new_data)
            print('Krak {}\t| Set new cache'.format(filename))
        print('Krak {}\t| Validated'.format(filename))


if __name__ == '__main__':
    client = Kraken('test_dir')
    start = time.time()
    if start % 60 != 0:
        delta = start % 60
        start = start - delta
    end = start + 600
    r = client.get_trades(product='XETHZUSD',
                          start=int(start),
                          end=int(end))
