import copy
import json
import time

import requests

from cacheable import Cacheable, StartKeyNotFoundError, EndKeyNotFoundError,\
    StartEndKeysNotFoundError
import dataset_list
import helper


class Gdax(Cacheable):
    """https://docs.gdax.com/
    """

    __GRANULARITY = 60

    def __init__(self, save_directory):
        super(Gdax, self).__init__(save_directory)
        self.url = 'https://api.gdax.com'
        self.timeout = 600

    def __get(self, path, payload):
        r = requests.get(self.url + path, params=payload, timeout=self.timeout)
        retries = 0
        while not r.ok or 'message' in r:
            print('GDAX | {}'.format(r))
            retries += 1
            time.sleep(3 * retries)
            r = requests.get(self.url + path, params=payload,
                             timeout=self.timeout)
#         print(json.dumps(r.json(), indent=4))
        return r.json()

    def download_charts(self, pair, start, end):
        filename = pair
        MAX_SIZE = 300

        print('GDAX Chart {}\t| Requested data from {} to {}'.format(pair, start, end))

        # carry forward
        # time, low, high, open, close, volume, is_carried
        last_seen_row = [0, 0, 0, 0, 0, 0, 1]

        # load if present
        try:
            self.check_cache(filename, (start, end - self.__GRANULARITY))
            print('GDAX Chart {}\t| Cached data is complete'.format(pair))
            return filename
        except FileNotFoundError:
            print('GDAX Chart {}\t| Cache for does not exist. Starting from time {}'.format(
                pair, start))
        except EndKeyNotFoundError:
            last_seen_row = self.get_last_cache(filename)
            latest_time = int(last_seen_row[0])
            start = latest_time + self.__GRANULARITY
            print('GDAX Chart {}\t| Continuing from latest time {}'.format(
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
            params = {'start': helper.unix_to_iso(slice_start),
                      'end': helper.unix_to_iso(slice_end),
                      'self.__GRANULARITY': self.__GRANULARITY}
            r = self.__get('/products/{}/candles'.format(pair), params)
            r = r[::-1]

            # may return length zero r
            if len(r) == 0:
                print('GDAX Chart {}\t| Returned data for {} -> {} is length zero'.format(
                    pair, slice_start, slice_end))
                for timestamp in range(slice_start, slice_end, self.__GRANULARITY):
                    last_seen_row[0] = timestamp  # correct time
                    last_seen_row[5] = 0  # volume
                    last_seen_row[-1] = 1  # is_carried == True
                    self.append_cache(filename, last_seen_row)
                # next slice
                slice_start = slice_end
                continue

            # carry forward and save
            slice_index = 0
            for timestamp in range(slice_start, slice_end, self.__GRANULARITY):
                current_row = r[slice_index]
                if (current_row[0] == timestamp):
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
                    last_seen_row[5] = 0  # volume
                    last_seen_row[-1] = 1  # is_carried == True
                    self.append_cache(filename, last_seen_row)

            # console print
            progress = 100 * (slice_end - start) / (end - start)
            print('GDAX Chart {}\t| {:6.2f}% | {} -> {} | {} -> {} | {} -> {}'.format(
                  pair,
                  progress,
                  r[0][0],
                  r[-1][0],
                  helper.unix_to_iso(r[0][0]),
                  helper.unix_to_iso(r[-1][0]),
                  len(r),
                  (slice_end - slice_start) // self.__GRANULARITY))

            # next slice
            slice_start = slice_end

        return filename

    def get_charts(self, pair, start, end):
        # download charts
        filename = self.download_charts(pair, start, end)

        # return window
        r = self.get_cache(filename, range_keys=(start, end - 60))
        return r

    def validate_data(self, filename):
        r = self.get_cache(filename)
        current_time = r[0][0]
        print('GDAX {}\t| Start processing from time {}'.format(
            filename, current_time))

        new_data = []
        for row in r:
            # check for invalid values
            for elem in row:
                if elem is None:
                    raise ValueError('GDAX {}\t| Invalid value {} encountered in row {}'.format(
                        filename, elem, row))
            # check for odd number of columns
            if len(row) != len(dataset_list.LABELS[dataset_list.GDAX_CHART]):
                raise ValueError('GDAX {}\t| Invalid number of columns {}. Expected {}'.format(
                    filename, len(row), len(dataset_list.LABELS[dataset_list.GDAX_CHART])))
            # check for invalid self.__GRANULARITY
            if row[0] % 60 != 0:
                raise ValueError('GDAX {}\t| Invalid interval of time for row {}'.format(
                    filename, row))
            # keep if correct time
            elif row[0] == current_time:
                new_data.append(row)
                current_time += self.__GRANULARITY
            # identified lack of order in data
            # possibly a result of running parallel downloads
            elif row[0] < current_time:
                # do not increment current_time
                print('GDAX {}\t| Duplicate time {} found. Expected time {}'.format(
                    filename, row[0], current_time))

        if len(new_data) != len(r):
            self.set_cache(filename, new_data)
            print('GDAX {}\t| Set new cache'.format(filename))
        print('GDAX {}\t| Validated'.format(filename))
