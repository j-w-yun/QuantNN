import json
from multiprocessing.pool import ThreadPool
import sys
import time

from requests.exceptions import HTTPError
from web3 import Web3, HTTPProvider

from cacheable import Cacheable, StartKeyNotFoundError, EndKeyNotFoundError,\
    StartEndKeysNotFoundError
import dataset_list
import numpy as np


class Ethereum(Cacheable):

    __GRANULARITY = 60
    __NUM_PROC = None

    def __init__(self, save_directory, worker_id='Main'):
        super(Ethereum, self).__init__(save_directory)
        providers = [
            # https://infura.io/
            HTTPProvider('https://mainnet.infura.io/4fjXq9goTOJaeQr6fz03'),
            HTTPProvider('https://api.myetherapi.com/eth')]
        self.w3 = Web3(providers)
        self.cache_key = None
        self.cache = None
        self.worker_id = worker_id

    def get_block_info(self, block_num):
        while self.cache is None or self.cache_key != block_num:
            self.cache_key = block_num
            fetch_successful = False
            while not fetch_successful:
                try:
                    self.cache = self.w3.eth.getBlock(block_num, True)
                    fetch_successful = True
                except HTTPError as e:
                    print(e)
        return self.cache

    def get_latest_block(self):
        return self.get_block_info('latest')['number']

    def get_block_time(self, block_num):
        return self.get_block_info(block_num)['timestamp']

    def get_block_transactions(self, block_num):
        return self.get_block_info(block_num)['transactions']  # list of dict

    def _worker_download_data(self, start, end):
        filename = 'ethereum_{}_{}_temp'.format(int(start), int(end))

        print('Ethereum Slice {}\t| Requested data from {} to {}'.format(
            self.worker_id, start, end))

        # load if present
        try:
            self.check_cache(filename, range_keys=(start, end - 60))
            print('Ethereum Slice {}\t| Cached data is complete'.format(
                self.worker_id))
            return filename
        except FileNotFoundError:
            print('Ethereum Slice {}\t| Cache does not exist. Starting from time {}'.format(
                self.worker_id, start))
        except EndKeyNotFoundError:
            latest_block_time = int(self.get_last_cache(filename)[0])
            start = latest_block_time + self.__GRANULARITY
            print('Ethereum Slice {}\t| Continuing from latest block time {}'.format(
                self.worker_id, latest_block_time))
        except (StartKeyNotFoundError, StartEndKeysNotFoundError) as e:
            # TODO: Handle importing disjoint historical data
            print(e)
            sys.exit()

        # get latest block height
        block_num = self.get_latest_block()

        # approximate time for the latest block
        curr_t = time.time()

        # block time interval
        APPROX_BLOCK_TIME = 20

        # negative if older block, positive if newer
        delta_t = curr_t - (start - 60)

        # find start block
        delta_block = delta_t / APPROX_BLOCK_TIME
        potential_start_block = int(block_num - delta_block)

        while(delta_block != 0):

            # if actual block time is smaller than start, go up one block
            block_time = self.get_block_time(potential_start_block)
            delta_t = block_time - (start - 60)
            delta_block = delta_t / APPROX_BLOCK_TIME

            # if within a 100 blocks, just iterate
            if abs(delta_t) < APPROX_BLOCK_TIME * 100:
                # while block is new
                while(delta_t > 0):
                    potential_start_block -= 10
                    block_time = self.get_block_time(
                        potential_start_block)
                    delta_t = block_time - (start - 60)
                # while block is old
                while(delta_t < 0):
                    potential_start_block += 1
                    block_time = self.get_block_time(
                        potential_start_block)
                    delta_t = block_time - (start - 60)
                break

            # new potential start block
            potential_start_block = int(
                potential_start_block - delta_block)
            if potential_start_block > self.get_latest_block():
                potential_start_block = self.get_latest_block()
            if potential_start_block < 0:
                potential_start_block = 0

        print('Ethereum Slice {}\t| Start block {}'.format(
            self.worker_id, potential_start_block))

        # start block
        current_block = potential_start_block

        for timestamp in range(start, end, self.__GRANULARITY):
            difficulty_list = []
            total_difficulty_list = []
            gas_used_list = []
            size_list = []
            num_transactions_list = []
            value_list = []
            num_contracts_list = []
            contract_gas_list = []
            contract_value_list = []

            while(self.get_block_time(current_block) < timestamp):
                block_info = self.get_block_info(current_block)

                difficulty = block_info['difficulty']
                difficulty_list.append(difficulty)

                total_difficulty = block_info['totalDifficulty']
                total_difficulty_list.append(total_difficulty)

                gas_used = block_info['gasUsed']
                gas_used_list.append(gas_used)

                size = block_info['size']
                size_list.append(size)

                transactions = self.get_block_transactions(current_block)
                num_transactions_list.append(len(transactions))

                value = 0
                num_contracts = 0
                contract_gas = 0
                contract_value = 0
                for tx in transactions:
                    value += tx['value']
                    if tx['input'] != '0x':
                        num_contracts += 1
                        contract_gas += tx['gas']
                        contract_value += contract_gas * tx['gasPrice']
                value_list.append(value / 1e18)  # wei to eth
                num_contracts_list.append(num_contracts)
                contract_gas_list.append(contract_gas)
                contract_value_list.append(contract_value / 1e18)  # wei to eth

                current_block += 1

            row = None
            if len(difficulty_list) > 0:
                row = [timestamp,
                       len(difficulty_list),  # num blocks in interval
                       sum(difficulty_list) / len(difficulty_list),
                       sum(total_difficulty_list) / len(total_difficulty_list),
                       sum(gas_used_list),
                       sum(size_list),
                       sum(num_transactions_list),
                       sum(value_list),
                       sum(num_contracts_list),
                       sum(contract_gas_list),
                       sum(contract_value_list)]
                # difficulty is large of a number. tone it down by magnitudes
                row[2] = row[2] / 1E10
                row[3] = row[3] / 1E15
            else:
                row = [0 for _ in dataset_list.LABELS[dataset_list.ETHE_BLOCK]]
                row[0] = timestamp

            # display progress
            progress = 100 * (timestamp - start) / (end - start)
            print('Ethereum Slice {}\t| {:6.2f}% | {}'.format(
                self.worker_id, progress, row))

            self.append_cache(filename, row)

        print('Ethereum Slice {}\t| End block {}'.format(
            self.worker_id, current_block))

        return filename

    def __concurrent_download_operation(self):
        # new instance for new web3 connection
        client = Ethereum(self.save_directory, worker_id=str(self.num_workers))
        self.num_workers += 1
        return client._worker_download_data

    def download_data(self, **kwargs):
        product = kwargs['product']
        start = kwargs['start']
        end = kwargs['end']

        filename = product

        print('Ethereum Slice {}\t| Requested data from {} to {}'.format(
            self.worker_id, start, end))

        # data to return
        data = []

        # load if present
        try:
            self.check_cache(filename, range_keys=(start, end - 60))
            print('Ethereum Slice {}\t| Cached data is complete'.format(
                self.worker_id))
            return filename
        except FileNotFoundError:
            print('Ethereum Slice {}\t| Cache does not exist. Starting from time {}'.format(
                self.worker_id, start))
        except EndKeyNotFoundError:
            latest_block_time = int(self.get_last_cache(filename)[0])
            start = latest_block_time + self.__GRANULARITY
            print('Ethereum Slice {}\t| Continuing from latest block time {}'.format(
                self.worker_id, latest_block_time))
            data.append(self.get_cache(filename))
        except (StartKeyNotFoundError, StartEndKeysNotFoundError) as e:
            # TODO: Handle importing disjoint historical data
            print(e)
            raise ValueError('Data is out of range')

        # divide up work
        interval = 9000
        buckets = []
        slice_start = start
        slice_end = start + interval
        while slice_end < end:
            buckets.append([slice_start, slice_end])
            slice_start = slice_end
            slice_end += interval
        # add last slice
        buckets.append([slice_start, end])

        # download concurrently
        self.num_workers = 0
        pool = ThreadPool(processes=self.__NUM_PROC)
        async_ops = []
        temp_filenames = []  # keep to delete temp files
        for slice_rng in buckets:
            async_ops.append(
                pool.apply_async(
                    self.__concurrent_download_operation(),
                    args=(slice_rng[0], slice_rng[1])))
        for async_op in async_ops:
            temp_filenames.append(async_op.get())
        print('Waiting for workers to complete...')
        pool.close()
        pool.join()
        print('All workers complete\n')

        # concatenate work
        for slice_rng, tempfile in zip(buckets, temp_filenames):
            data.append(
                self.get_cache(
                    tempfile, range_keys=(slice_rng[0], slice_rng[1] - 60)))
            print('Reading slice : {}'.format(slice_rng))
        print('Concatenating slices...')
        data = np.concatenate(data, axis=0)
        print('Setting new cache...')
        self.set_cache(filename, data, safe_overwrite=True)
        print('Done\n')

        # delete temp files
        for tempfile in temp_filenames:
            self.delete_cache(tempfile)

        return filename

    def get_blocks(self, **kwargs):
        start = kwargs['start']
        end = kwargs['end']

        # download data
        filename = self.download_data(**kwargs)

        # return window
        return self.get_cache(filename, range_keys=(start, end - 60))

    def validate_data(self, filename):
        r = self.get_cache(filename)
        current_time = r[0][0]
        print('Ethereum\t| Start processing from time {}'.format(current_time))

        new_data = []
        for row in r:
            # check for invalid values
            for elem in row:
                if elem is None:
                    raise ValueError('Ethereum\t| Invalid value {} encountered in row {}'.format(
                        elem, row))
            # check for odd number of columns
            if len(row) != len(dataset_list.LABELS[dataset_list.ETHE_BLOCK]):
                raise ValueError('Ethereum\t| Invalid number of columns {}. Expected {}'.format(
                    len(row), len(dataset_list.LABELS[dataset_list.ETHE_BLOCK])))
            # check for invalid self.__GRANULARITY
            if row[0] % 60 != 0:
                raise ValueError('Ethereum\t| Invalid interval of time for row {}'.format(
                    row))
            # keep if correct time
            elif row[0] == current_time:
                new_data.append(row)
                current_time += self.__GRANULARITY
            # identified lack of order in data
            # possibly a result of running parallel downloads
            elif row[0] < current_time:
                # do not increment current_time
                print('Ethereum\t| Duplicate time {} found. Expected time {}'.format(
                    row[0], current_time))

        if len(new_data) != len(r):
            self.set_cache(filename, new_data)
            print('Ethereum\t| Set new cache')
        print('Ethereum\t| Validated')
