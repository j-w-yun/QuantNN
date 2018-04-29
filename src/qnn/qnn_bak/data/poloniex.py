import copy
import sys
import time

import requests

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

    def __get(self, path, payload, max_retries=100):
        r = None

        # Invalid format packet
        for retries in range(max_retries):
            try:
                r = requests.get(
                    self.url + path, params=payload, timeout=self.timeout)

                # HTTP not OK or Poloniex error
                while not r.ok or 'message' in r.json():
                    time.sleep(3 * retries)
                    r = requests.get(
                        self.url + path, params=payload, timeout=self.timeout)
            except:
                time.sleep(60)
                continue
            break

        return r.json()

    def _get_trades(self, pair, start, end):
        params = {'currencyPair': pair,
                  'start': start,
                  'end': end}
        r = self.__get('/public?command=returnTradeHistory', params)

        if len(r) < 50000:
            return r
        else:
            r = []
            start_1 = start
            end_1 = start + ((end - start) // 2)
            start_2 = end_1
            end_2 = end
            r.extend(self._get_trades(pair, start_2, end_2))
            r.extend(self._get_trades(pair, start_1, end_1))
            return r

    def download_trades(self, **kwargs):
        product = kwargs['product']
        start = kwargs['start']
        end = kwargs['end']

        filename = product + '_trades'
        MAX_SIZE = 300

        print('Polo Trade {}\t| Requested data from {} to {}'.format(
            product, start, end))

        # initial carry forward value
        last_seen_row = [
            1 if label == dataset_list.IS_CARRIED else 0 for label in dataset_list.LABELS[dataset_list.POLO_TRADE]]

        # load if present
        try:
            self.check_cache(filename, (start, end - self.__GRANULARITY))
            print('Polo Trade {}\t| Cached data is complete'.format(product))
            return filename
        except FileNotFoundError:
            print('Polo Trade {}\t| Cache for does not exist. Starting from time {}'.format(
                product, start))
        except EndKeyNotFoundError:
            last_seen_row = self.get_last_cache(filename)
            latest_time = int(last_seen_row[0])
            start = latest_time + self.__GRANULARITY
            print('Polo Trade {}\t| Continuing from latest time {}'.format(
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
            t = t[::-1]

            if len(t) == 0:
                print('Polo Trade {}\t| Returned data for {} -> {} is length zero'.format(
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

                while(helper.iso_to_unix(trade['date']) <= timestamp and
                      helper.iso_to_unix(trade['date']) > (timestamp - self.__GRANULARITY)):
                    trade = t[current_index]

                    rate_list.append(float(trade['rate']))
                    total_list.append(float(trade['total']))
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
            print('Polo Trade {}\t| {:6.2f}% | {} -> {} | {} -> {} | {}'.format(
                  product,
                  progress,
                  helper.iso_to_unix(t[0]['date']),
                  helper.iso_to_unix(t[-1]['date']),
                  t[0]['date'].replace(' ', 'T'),
                  t[-1]['date'].replace(' ', 'T'),
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
            if len(row) != len(dataset_list.LABELS[dataset_list.POLO_TRADE]):
                raise ValueError('Polo {}\t| Invalid number of columns {}. Expected {}'.format(
                    filename, len(row), len(dataset_list.LABELS[dataset_list.POLO_TRADE])))
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
