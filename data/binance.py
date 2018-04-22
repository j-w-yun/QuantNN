import copy
import json
import time

import requests

from cacheable import Cacheable, StartKeyNotFoundError, EndKeyNotFoundError,\
    StartEndKeysNotFoundError
import dataset_list
import helper


class Binance(Cacheable):
    """https://github.com/binance-exchange/binance-official-api-docs/blob/master/rest-api.md
    """

    __GRANULARITY = 60

    def __init__(self, save_directory):
        super(Binance, self).__init__(save_directory)
        self.url = 'https://api.binance.com'
        self.timeout = 600

    def __get(self, path, payload):
        r = requests.get(self.url + path, params=payload, timeout=self.timeout)
        retries = 0
        while not r.ok:
            print('Binance | {}'.format(r))
            retries += 1
            time.sleep(3 * retries)
            r = requests.get(self.url + path, params=payload,
                             timeout=self.timeout)
#         print(json.dumps(r.json(), indent=4))
        return r.json()

    def download_charts(self, pair, start, end):
        filename = pair
        MAX_SIZE = 500

        print('Bina Chart {}\t| Requested data from {} to {}'.format(pair, start, end))

        # carry forward
        # time, low, high, open, close, volume, quote_asset_vol, num_trades, taker_buy_base_asset_vol, taker_buy_quote_asset_vol, is_carried
        last_seen_row = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        # load if present
        try:
            self.check_cache(filename, (start, end - self.__GRANULARITY))
            print('Bina Chart {}\t| Cached data is complete'.format(pair))
            return filename
        except FileNotFoundError:
            print('Bina Chart {}\t| Cache for does not exist. Starting from time {}'.format(
                pair, start))
        except EndKeyNotFoundError:
            last_seen_row = self.get_last_cache(filename)
            latest_time = int(last_seen_row[0])
            start = latest_time + self.__GRANULARITY
            print('Bina Chart {}\t| Continuing from latest time {}'.format(
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
            params = {
                'symbol': pair,
                'startTime': slice_start * 1000,
                'endTime': slice_end * 1000,
                'interval': '1m',
                'limit': MAX_SIZE
            }
            r = self.__get('/api/v1/klines', params)

            # binance returns length zero data during mid Feb
            if len(r) == 0:
                print('Bina Chart {}\t| Returned data for {} -> {} is length zero'.format(
                    pair, slice_start, slice_end))
                for timestamp in range(slice_start, slice_end, self.__GRANULARITY):
                    last_seen_row[0] = timestamp  # correct time
                    last_seen_row[-1] = 1  # is_carried == True
                    self.append_cache(filename, last_seen_row)
                # next slice
                slice_start = slice_end
                continue

            # carry forward and save
            slice_index = 0
            for timestamp in range(slice_start, slice_end, self.__GRANULARITY):
                current_row_list = r[slice_index]

                rounded_time = round(current_row_list[6] / 1000)
                if rounded_time % 60 != 0:
                    offset = rounded_time % 60
                    offset = 60 - offset
                    rounded_time = rounded_time + offset

                current_row = [
                    # close time. change to open time?
                    rounded_time,
                    float(current_row_list[3]),  # low
                    float(current_row_list[2]),  # high
                    float(current_row_list[1]),  # open
                    float(current_row_list[4]),  # close
                    float(current_row_list[5]),  # volume
                    float(current_row_list[7]),  # quote asset vol
                    current_row_list[8],  # num trades
                    float(current_row_list[9]),  # taker buy base asset vol
                    float(current_row_list[10])  # taker buy quote asset vol
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
            print('Bina Chart {}\t| {:6.2f}% | {} -> {} | {} -> {} | {} -> {}'.format(
                  pair,
                  progress,
                  round(r[0][0] / 1000),
                  round(r[-1][0] / 1000),
                  helper.unix_to_iso(round(r[0][0] / 1000)),
                  helper.unix_to_iso(round(r[-1][0] / 1000)),
                  len(r), (slice_end - slice_start) // self.__GRANULARITY))

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
        print('Bina {}\t| Start processing from time {}'.format(
            filename, current_time))

        new_data = []
        for row in r:
            # check for invalid values
            for elem in row:
                if elem is None:
                    raise ValueError('Bina {}\t| Invalid value {} encountered in row {}'.format(
                        filename, elem, row))
            # check for odd number of columns
            if len(row) != len(dataset_list.LABELS[dataset_list.BINA_CHART]):
                raise ValueError('Bina {}\t| Invalid number of columns {}. Expected {}'.format(
                    filename, len(row), len(dataset_list.LABELS[dataset_list.BINA_CHART])))
            # check for invalid self.__GRANULARITY
            if row[0] % 60 != 0:
                raise ValueError('Bina {}\t| Invalid interval of time for row {}'.format(
                    filename, row))
            # keep if correct time
            elif row[0] == current_time:
                new_data.append(row)
                current_time += self.__GRANULARITY
            # identified lack of order in data
            # possibly a result of running parallel downloads
            elif row[0] < current_time:
                # do not increment current_time
                print('Bina {}\t| Duplicate time {} found. Expected time {}'.format(
                    filename, row[0], current_time))

        if len(new_data) != len(r):
            self.set_cache(filename, new_data)
            print('Bina {}\t| Set new cache'.format(filename))
        print('Bina {}\t| Validated'.format(filename))
