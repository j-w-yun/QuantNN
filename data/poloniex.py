import copy
import json
import time

import requests
from requests.exceptions import ReadTimeout

from cacheable import Cacheable, StartKeyNotFoundError, EndKeyNotFoundError,\
    StartEndKeysNotFoundError
import dataset_list
import helper
import numpy as np


class Poloniex(Cacheable):
    """https://poloniex.com/support/api/
    """

    __GRANULARITY = 60

    def __init__(self, save_directory):
        super(Poloniex, self).__init__(save_directory)
        self.url = 'https://poloniex.com'
        self.timeout = 600

    def __get(self, path, payload):
        try:
            r = requests.get(
                self.url + path, params=payload, timeout=self.timeout)
        except ReadTimeout:
            # try one more time
            time.sleep(60)
            r = requests.get(
                self.url + path, params=payload, timeout=self.timeout)
        retries = 0
        while not r.ok:
            print('Poloniex | {}'.format(r))
            retries += 1
            time.sleep(3 * retries)
            r = requests.get(self.url + path, params=payload,
                             timeout=self.timeout)
#         print(json.dumps(r.json(), indent=4))
        return r.json()

    def _get_trades(self, pair, start, end):
        params = {'currencyPair': pair,
                  'start': start,
                  'end': end}
        r = {'message': None}
        while 'message' in r:
            r = self.__get('/public?command=returnTradeHistory', params)
        if len(r) < 50000:
            return r
        else:
            r = []
            start_1 = start
            end_1 = start + ((end - start) // 2)
            start_2 = end_1
            end_2 = end

            part2 = self._get_trades(pair, start_2, end_2)
            r.extend(part2)
#             for row in part2[-10:]:
#                 print('1 :', row)
            part1 = self._get_trades(pair, start_1, end_1)
            r.extend(part1)
#             for row in part1[:10]:
#                 print('2 :', row)
            return r

    def _get_chart(self, pair, start, end, granularity):
        params = {
            'currencyPair': pair,
            'start': start,
            'end': end,
            'period': granularity
        }
        r = {'message': None}
        while 'message' in r:
            r = self.__get('/public?command=returnChartData', params)
        return r

    def download_trades(self, pair, start, end):
        filename = pair + '_trades'
        MAX_SIZE = 5000

        print('Polo Trade {}\t| Requested data from {} to {}'.format(pair, start, end))

        # carry forward
        last_seen_row = [0, 0, 0, 0, 0, 1]

        # load if present
        try:
            self.check_cache(filename, (start, end - self.__GRANULARITY))
            print('Polo Trade {}\t| Cached data is complete'.format(pair))
            return filename
        except FileNotFoundError:
            print('Polo Trade {}\t| Cache for does not exist. Starting from time {}'.format(
                pair, start))
        except EndKeyNotFoundError:
            last_seen_row = self.get_last_cache(filename)
            latest_time = int(last_seen_row[0])
            start = latest_time + self.__GRANULARITY
            print('Polo Trade {}\t| Continuing from latest time {}'.format(
                pair, latest_time))
        except (StartKeyNotFoundError) as e:
            # slow and may result in disjoint data if prematurely terminated
            print(e)
            raise Warning('Deprecated')
        except (StartEndKeysNotFoundError) as e:
            # TODO: Handle importing disjoint historical data
            print(e)
            raise ValueError('Data is out of range')

        slice_range = self.__GRANULARITY * MAX_SIZE
        slice_start = start
        while slice_start != end:
            slice_end = min(slice_start + slice_range, end)

            # fetch slice
            t = self._get_trades(pair, slice_start, slice_end)
            t = t[::-1]

            if len(t) == 0:
                print('Polo Trade {}\t| Returned data for {} -> {} is length zero'.format(
                    pair, slice_start, slice_end))
                for timestamp in range(slice_start, slice_end, self.__GRANULARITY):
                    last_seen_row[0] = timestamp  # correct time
                    last_seen_row[1] = 0
                    last_seen_row[2] = 0
                    last_seen_row[-1] = 1  # is_carried == True
                    self.append_cache(filename, last_seen_row)
                # next slice
                slice_start = slice_end
                continue

            # carry forward and save
            slice_index = 0
            trade = t[slice_index]
            for timestamp in range(slice_start, slice_end, self.__GRANULARITY):
                volume_list = []
                rate_list = []
                total_list = []
                weighted_avg = []

                while(helper.iso_to_unix(trade['date']) <= timestamp and
                      helper.iso_to_unix(trade['date']) > (timestamp - self.__GRANULARITY)):
                    trade = t[slice_index]

                    volume_list.append(float(trade['amount']))
                    rate_list.append(float(trade['rate']))
                    total_list.append(float(trade['total']))
                    weighted_avg.append(volume_list[-1] * rate_list[-1])

                    # break loop if last index is reached
                    if (slice_index + 1) == len(t):
                        break
                    slice_index += 1

                if len(volume_list) > 0 and np.sum(volume_list) != 0:
                    row = [
                        timestamp,
                        len(volume_list),  # num buys
                        np.sum(total_list),  # buy total
                        np.sum(rate_list) / len(rate_list),  # avg
                        np.sum(weighted_avg) / np.sum(volume_list),  # w avg
                        0
                    ]
                else:
                    last_seen_row[0] = timestamp  # correct time
                    last_seen_row[1] = 0
                    last_seen_row[2] = 0
                    last_seen_row[-1] = 1  # is_carried == True
                    row = last_seen_row

                # last seen
                last_seen_row = copy.deepcopy(row)

                # save
                self.append_cache(filename, row)

            # console print
            progress = 100 * (slice_end - start) / (end - start)
            print('Polo Trade {}\t| {:6.2f}% | {} -> {} | {} -> {} | {}'.format(
                  pair,
                  progress,
                  helper.iso_to_unix(t[0]['date']),
                  helper.iso_to_unix(t[-1]['date']),
                  t[0]['date'].replace(' ', 'T'),
                  t[-1]['date'].replace(' ', 'T'),
                  len(t)))

            # next slice
            slice_start = slice_end

        return filename

    def download_charts(self, pair, start, end):
        filename = pair
        MIN_GRANULARITY = 300
        MAX_SIZE = 1000

        print('Polo Chart {}\t| Requested data from {} to {}'.format(pair, start, end))

        # carry forward
        # date, low, high, open, close, volume, quoteVolume, weightedAverage, is_carried
        last_seen_row = [0, 0, 0, 0, 0, 0, 0, 0, 1]

        # load if present
        try:
            self.check_cache(filename, (start, end - self.__GRANULARITY))
            print('Polo Chart {}\t| Cached data is complete'.format(pair))
            return filename
        except FileNotFoundError:
            print('Polo Chart {}\t| Cache for does not exist. Starting from time {}'.format(
                pair, start))
        except EndKeyNotFoundError:
            last_seen_row = self.get_last_cache(filename)
            latest_time = int(last_seen_row[0])
            start = latest_time + self.__GRANULARITY
            print('Polo Chart {}\t| Continuing from latest time {}'.format(
                pair, latest_time))
        except (StartKeyNotFoundError) as e:
            # slow and may result in disjoint data if prematurely terminated
            print(e)
            raise Warning('Deprecated')
        except (StartEndKeysNotFoundError) as e:
            # TODO: Handle importing disjoint historical data
            print(e)
            raise ValueError('Data is out of range')

        slice_range = MIN_GRANULARITY * MAX_SIZE
        slice_start = start
        while slice_start != end:
            slice_end = min(slice_start + slice_range, end)

            # fetch slice
            r = self._get_chart(pair, slice_start, slice_end, MIN_GRANULARITY)

            # carry forward and save
            slice_index = 0
            for timestamp in range(slice_start, slice_end, self.__GRANULARITY):
                current_row_dict = r[slice_index]
                current_row = [
                    current_row_dict['date'],
                    current_row_dict['low'],
                    current_row_dict['high'],
                    current_row_dict['open'],
                    current_row_dict['close'],
                    current_row_dict['volume'],
                    current_row_dict['quoteVolume'],
                    current_row_dict['weightedAverage']
                ]
                if current_row[0] == timestamp:
                    current_row.append(0)  # is_carried == False
                    # some values received may be None
                    for elem_index, elem in enumerate(current_row):
                        if elem is None:
                            current_row[elem_index] = last_seen_row[elem_index]
                    self.append_cache(filename, current_row)
                    last_seen_row = copy.deepcopy(current_row)  # last seen
                    slice_index = min(slice_index + 1, len(r) - 1)
                else:
                    last_seen_row[0] = timestamp  # correct time
                    last_seen_row[-1] = 1  # is_carried == True
                    self.append_cache(filename, last_seen_row)

            # console print
            progress = 100 * (slice_end - start) / (end - start)
            print('Polo Chart {}\t| {:6.2f}% | {} -> {} | {} -> {} | {} -> {}'.format(
                  pair,
                  progress,
                  r[0]['date'],
                  r[-1]['date'],
                  helper.unix_to_iso(r[0]['date']),
                  helper.unix_to_iso(r[-1]['date']),
                  len(r),
                  (slice_end - slice_start) // self.__GRANULARITY))

            # next slice
            slice_start = slice_end

        return filename

    def get_trades(self, pair, start, end):
        # download trades
        filename = self.download_trades(pair, start, end)

        # return window
        r = self.get_cache(filename, range_keys=(start, end - 60))
        return r

    def get_charts(self, pair, start, end):
        # download charts
        filename = self.download_charts(pair, start, end)

        # return window
        r = self.get_cache(filename, range_keys=(start, end - 60))
        return r

    def validate_data(self, filename):
        r = self.get_cache(filename)
        current_time = r[0][0]
        print('Polo {}\t| Start processing from time {}'.format(
            filename, current_time))

        new_data = []
        for row in r:
            # check for invalid values
            for elem in row:
                if elem is None:
                    raise ValueError('Polo {}\t| Invalid value {} encountered in row {}'.format(
                        filename, elem, row))
            # check for odd number of columns
            if '_trades' in filename:
                if len(row) != len(dataset_list.LABELS[dataset_list.POLO_TRADE]):
                    raise ValueError('Polo {}\t| Invalid number of columns {}. Expected {}'.format(
                        filename, len(row), len(dataset_list.LABELS[dataset_list.POLO_TRADE])))
            else:
                if len(row) != len(dataset_list.LABELS[dataset_list.POLO_CHART]):
                    raise ValueError('Polo {}\t| Invalid number of columns {}. Expected {}'.format(
                        filename, len(row), len(dataset_list.LABELS[dataset_list.POLO_CHART])))
            # check for invalid self.__GRANULARITY
            if row[0] % 60 != 0:
                raise ValueError('Polo {}\t| Invalid interval of time for row {}'.format(
                    filename, row))
            # keep if correct time
            elif row[0] == current_time:
                new_data.append(row)
                current_time += self.__GRANULARITY
            # identified lack of order in data
            # possibly a result of running parallel downloads
            elif row[0] < current_time:
                # do not increment current_time
                print('Polo {}\t| Duplicate time {} found. Expected time {}'.format(
                    filename, row[0], current_time))

        if len(new_data) != len(r):
            self.set_cache(filename, new_data)
            print('Polo {}\t| Set new cache'.format(filename))
        print('Polo {}\t| Validated'.format(filename))


if __name__ == '__main__':
    client = Poloniex('test_dir')
    r = client.get_trades('USDT_ETH', 1488357000, 1518357000)
