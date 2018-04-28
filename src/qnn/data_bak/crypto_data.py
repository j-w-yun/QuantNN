from datetime import datetime
from multiprocessing.pool import ThreadPool
import shutil
import sys

from binance import Binance
import dateutil.tz
from gdax import Gdax
from poloniex import Poloniex

from cacheable import Cacheable
import dataset_list
from ethereum import Ethereum
import helper
from kraken import Kraken
import numpy as np


class DataSets:
    """Separately contains train and test datasets.
    """

    def __init__(self,
                 train,
                 test):
        self._train = train
        self._test = test

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test


class DataSet:
    """Separately contains inputs and targets for a dataset.
    """

    def __init__(self,
                 inputs,
                 input_labels,
                 targets,
                 target_labels,
                 raw_targets,
                 input_seq_length,
                 target_seq_length):
        assert inputs.shape[0] == targets.shape[0], (
            'inputs.shape: %s targets.shape: %s' % (inputs.shape, targets.shape))

        self._inputs = inputs
        self._input_labels = input_labels
        self._targets = targets
        self._target_labels = target_labels
        self._raw_targets = raw_targets
        self._input_seq_length = input_seq_length
        self._target_seq_length = target_seq_length

        self._input_depth = inputs.shape[1]
        self._target_depth = targets.shape[1]

    @property
    def inputs(self):
        return self._inputs

    @property
    def input_labels(self):
        return self._input_labels

    @property
    def targets(self):
        return self._targets

    @property
    def target_labels(self):
        return self._target_labels

    @property
    def raw_targets(self):
        return self._raw_targets

    @property
    def input_seq_length(self):
        return self._input_seq_length

    @property
    def target_seq_length(self):
        return self._target_seq_length

    @property
    def num_examples(self):
        """Number of total examples in this dataset.
        """
        return (self.inputs.shape[0] -
                (self.input_seq_length + self.target_seq_length - 1))

    @property
    def input_depth(self):
        """Number of input features.
        """
        return self._input_depth

    @property
    def target_depth(self):
        """Number of target features.
        """
        return self._target_depth

    def get_example(self, index):
        """Retrieve single input-target data pair.
        """
        input_start = index
        input_end = input_start + self.input_seq_length
        output_start = input_end
        output_end = output_start + self.target_seq_length
        x = self.inputs[input_start:input_end]
        y = self.targets[output_start:output_end]
        x = np.reshape(x, newshape=[1, -1, self.input_depth])
        y = np.reshape(y, newshape=[1, -1, self.target_depth])
        return (x, y)

    def get_example_set(self, indices):
        """Retrieve a set of input-target data dataset_list.
        """
        batch_x, batch_y = [], []
        for index in indices:
            x, y = self.get_example(index)
            batch_x.append(x)
            batch_y.append(y)
        batch_x = np.concatenate(batch_x, axis=0)
        batch_y = np.concatenate(batch_y, axis=0)
        return (batch_x, batch_y)


class Crypto(Cacheable):
    # by specifying 'use_test_dir == True', data can be cached and processed in
    # the test directory, without modifying original data directory contents
    TEST_DIRECTORY = 'test_directory'

    # number of concurrent threads to use during data download
    __NUM_PROC = 128

    def __init__(self, cache_directory):
        super(Crypto, self).__init__(save_directory=cache_directory)

    @property
    def dataset(self):
        return self._dataset

    @property
    def start_unix(self):
        return self._start_unix

    @property
    def end_unix(self):
        return self._end_unix

    @property
    def start_date(self):
        return self._start_date

    @property
    def end_date(self):
        return self._end_date

    def set_dataset(self, dataset):
        """Select data to download or fetch from disk cache to use in dataset.
        """
        self._dataset = {}

        if dataset_list.GDAX_CHART in dataset:
            self._dataset[dataset_list.GDAX_CHART] = {
                'dir': 'data\\exchange\\gdax',
                'client': Gdax,
                'data_list': dataset[dataset_list.GDAX_CHART],
                'data_label': dataset_list.LABELS[dataset_list.GDAX_CHART],
                'download_op': 'download_charts',
                'fetch_op': 'get_charts',
                'validate_op': 'validate_charts'}

        if dataset_list.POLO_TRADE in dataset:
            self._dataset[dataset_list.POLO_TRADE] = {
                'dir': 'data\\exchange\\poloniex',
                'client': Poloniex,
                'data_list': dataset[dataset_list.POLO_TRADE],
                'data_label': dataset_list.LABELS[dataset_list.POLO_TRADE],
                'download_op': 'download_trades',
                'fetch_op': 'get_trades',
                'validate_op': 'validate_trades'}

        if dataset_list.KRAK_TRADE in dataset:
            self._dataset[dataset_list.KRAK_TRADE] = {
                'dir': 'data\\exchange\\kraken',
                'client': Kraken,
                'data_list': dataset[dataset_list.KRAK_TRADE],
                'data_label': dataset_list.LABELS[dataset_list.KRAK_TRADE],
                'download_op': 'download_trades',
                'fetch_op': 'get_trades',
                'validate_op': 'validate_trades'}

        if dataset_list.ETHE_BLOCK in dataset:
            self._dataset[dataset_list.ETHE_BLOCK] = {
                'dir': 'data\\blockchain\\ethereum',
                'client': Ethereum,
                'data_list': dataset[dataset_list.ETHE_BLOCK],
                'data_label': dataset_list.LABELS[dataset_list.ETHE_BLOCK],
                'download_op': 'download_data',
                'fetch_op': 'get_blocks',
                'validate_op': 'validate_data'}

    def set_target(self, exchange_list, product_list, labels_list):
        """Set a currency pair to use as prediction target sequence.
        """
        self.target_labels = []
        for exchange, product, labels in zip(
                exchange_list, product_list, labels_list):

            # debug
            if not exchange in self.dataset:
                raise ValueError('Exchange {} not found in working dataset'.format(
                    exchange))
            if not product in self.dataset[exchange]['data_list']:
                raise ValueError('Product {} not found for exchange {}'.format(
                    product, exchange))
            for label in labels:
                if not label in self.dataset[exchange]['data_label']:
                    raise ValueError('Label {} not found for exchange{}'.format(
                        exchange))

            # add to target label for processing target later
            for label in labels:
                self.target_labels.append([exchange, product, label])

    def set_start_unix(self, start_unix):
        """Set the start UNIX timestamp of data.
        """
        if start_unix % 60 != 0:
            start_unix -= start_unix % 60
        self._start_unix = start_unix
        self._start_date = helper.unix_to_utc(start_unix)

    def set_end_unix(self, end_unix):
        """Set the end UNIX timestamp of data.
        """
        if end_unix % 60 != 0:
            end_unix -= end_unix % 60
        self._end_unix = end_unix
        self._end_date = helper.unix_to_utc(end_unix)

    def set_start_date(self, date):
        """Set the start date of data.
        """
        start_date_local = datetime(
            year=date['year'],
            month=date['month'],
            day=date['day'],
            hour=date['hour'],
            minute=date['minute'],
            tzinfo=dateutil.tz.tzlocal())

        # convert local tz datetime to unix
        utctz = dateutil.tz.tzutc()
        start_date = start_date_local.astimezone(utctz)
        start_unix = helper.date_to_unix(start_date)
        self._start_unix = start_unix - start_unix % 60
        self._start_date = helper.unix_to_utc(start_unix)

    def set_end_date(self, date):
        """Set the end date of data.
        """
        end_date_local = datetime(
            year=date['year'],
            month=date['month'],
            day=date['day'],
            hour=date['hour'],
            minute=date['minute'],
            tzinfo=dateutil.tz.tzlocal())

        # convert local tz datetime to unix
        utctz = dateutil.tz.tzutc()
        end_date = end_date_local.astimezone(utctz)
        end_unix = helper.date_to_unix(end_date)
        self._end_unix = end_unix - end_unix % 60
        self._end_date = helper.unix_to_utc(end_unix)

    def del_test_dir(self):
        """Delete the test directory.
        """
        shutil.rmtree(self.TEST_DIRECTORY, ignore_errors=True)
        print('Deleted directory {} and all of its contents\n'.format(
            self.TEST_DIRECTORY))

    def _validate_data(self):
        """Check file correctness. Delete duplicate rows, if any exist.
        """
        # concurrent workers
        pool = ThreadPool(processes=None)

        # define concurrent ops
        async_ops = []
        for key in sorted(self.dataset):
            for product in self.dataset[key]['data_list']:
                client = self.dataset[key]['client'](self.dataset[key]['dir'])
                validate_op = getattr(client, self.dataset[key]['validate_op'])
                async_ops.append(pool.apply_async(
                    func=validate_op, args=(product,)))

        # execute concurrent ops
        for op in async_ops:
            op.get()
        pool.close()
        pool.join()

    def _download_data(self, validate_data=False):
        """Concurrently download all data specified in dataset.
        """
        if validate_data:
            self._validate_data()

        # concurrent workers
        tp = ThreadPool(processes=self.__NUM_PROC)

        # define concurrent ops
        async_ops = []
        for key in sorted(self.dataset):
            for product in self.dataset[key]['data_list']:
                client = self.dataset[key]['client'](self.dataset[key]['dir'])
                op = getattr(client, self.dataset[key]['download_op'])
                async_ops.append(tp.apply_async(
                    func=op, kwds={'product': product,
                                   'start': self.start_unix,
                                   'end': self.end_unix}))

        # execute concurrent ops
        for async_op in async_ops:
            async_op.get()
        tp.close()
        tp.join()

        print('Download complete\n')

    def _get_raw_data(self, validate_data=False, load_cache=False):
        """Process all individual data from cache into a single data structure.
        """
        if load_cache:
            try:
                # raw_data shape
                shape = np.load('{}\\shape.npy'.format(self.save_directory))

                # raw_data column labels
                labels = np.load('{}\\labels.npy'.format(self.save_directory))

                # unprocessed data
                raw_data = np.memmap(
                    '{}\\raw_data'.format(self.save_directory),
                    dtype='float64',
                    mode='r',
                    shape=(int(shape[0]), int(shape[1])))

                # debug
                print('shape.npy {}'.format(shape))
                print('labels.npy {}'.format(labels.shape))
                print('raw_data {}'.format(raw_data.shape))

                # return now if cache fetch successful
                return raw_data, labels

            except FileNotFoundError as e:
                print('Could not load from cache\n', e)
                load_cache = False

        # get raw data
        print('Not using cache. Fetching from API and/or individual files...')

        # concurrently fetch data
        self._download_data(validate_data)

        # remove some data according to label in data to return
        col_labels_to_del = [dataset_list.TIME,
                             dataset_list.IS_CARRIED]

        # count the columns to add, finding final column size as a result
        col_counter = 0
        for key in sorted(self.dataset):
            for _product in self.dataset[key]['data_list']:
                for col_label in self.dataset[key]['data_label']:
                    if col_label not in col_labels_to_del:
                        col_counter += 1

        # memmap shape. memmap holds final raw data
        num_cols = col_counter
        num_rows = len(range(self.start_unix, self.end_unix, 60))
        # keep on disk as data can be too large for RAM
        raw_data = np.memmap(
            '{}\\raw_data'.format(self.save_directory),
            dtype='float64',
            mode='w+',
            shape=(num_rows, num_cols))

        def memmap_hstack(memmap, data, indices, num_cols):
            """Horizontally stack working columns to disk.
            """
            col_start = indices[0]
            indices[1] += num_cols
            col_end = indices[1]
            memmap[:, col_start:col_end] = data
            indices[0] = col_end

        # collect labels for each column for processing raw data later
        data_labels = []

        # window of working column indices during horizontal stacking
        indices = [0, 0]

        # begin horizontally stacking individual data to memmap
        for key in sorted(self.dataset):
            client = self.dataset[key]['client'](
                self.dataset[key]['dir'])
            for product in self.dataset[key]['data_list']:
                fetch_op = getattr(client, self.dataset[key]['fetch_op'])
                pair_data = fetch_op(product=product,
                                     start=self.start_unix,
                                     end=self.end_unix)
                cols_to_del = []
                new_labels = []
                for col_i, col_label in enumerate(self.dataset[key]['data_label']):
                    if col_label in col_labels_to_del:
                        cols_to_del.append(col_i)
                    else:
                        new_labels.append([key, product, col_label])
                pair_data = np.delete(pair_data, cols_to_del, axis=1)
                memmap_hstack(raw_data, pair_data, indices, len(new_labels))
                data_labels.extend(new_labels)
                # debug
                print('{} {} added to memmap'.format(key, product))

        # write and flush
        raw_data.flush()
        del(raw_data)
        print('Raw data memmap of shape {}, {} saved\n'.format(
            num_rows, num_cols))

        # save data shape
        np.save('{}\\shape'.format(self.save_directory),
                np.array([num_rows, num_cols]))

        # concatenate column labels and save for loading memmap
        np.save('{}\\labels'.format(self.save_directory), data_labels)

        # open as read-only
        raw_data = np.memmap(
            '{}\\raw_data'.format(self.save_directory),
            dtype='float64',
            mode='r',
            shape=(num_rows, num_cols))

        return raw_data, data_labels

    def _process_data(self, data, labels):
        """Process raw data into final input and target data structures.
        """
        def remove_zero_vals(col_data):
            """Remove zeros by forward or backward carry."""
            new_col_data = []
            carry_back = []
            # carry forwards
            last_seen_val = 0
            for row_index, row_elem in enumerate(col_data):
                if row_elem == 0:
                    if last_seen_val == 0:  # log all first zeros
                        carry_back.append(row_index)
                    else:
                        new_col_data.append(last_seen_val)
                else:
                    if len(carry_back) != 0:
                        for _ in carry_back:  # fill all first zeros
                            new_col_data.append(row_elem)
                        carry_back = []
                    last_seen_val = row_elem
                    new_col_data.append(row_elem)
            return np.array(new_col_data)

        def ln_ratio(col_data, scale=100.0):
            """Log ratio of change."""
            col_data = np.reshape(col_data, [-1])
            col_data = remove_zero_vals(col_data)
            y = col_data[1:]  # t+1
            y_ = col_data[:-1]  # t
            y = np.log(y / y_) * scale
            return y

        def truncate(col_data, factor=5.0):
            """Truncates by standard deviation."""
            mean = np.mean(col_data, axis=0)
            std = np.std(col_data, axis=0)
            clip_min = mean - factor * std
            clip_max = mean + factor * std
            return np.clip(col_data, clip_min, clip_max)

        def amt_change(col_data):
            """Amount of change from t to t+1."""
            col_data = np.reshape(col_data, [-1])
            col_data = remove_zero_vals(col_data)
            y = col_data[1:]  # t+1
            y_ = col_data[:-1]  # t
            y = y - y_
            return y

        def z_score(col_data):
            """Z-score."""
            col_data = np.reshape(col_data, [-1])
            mean = np.mean(col_data, axis=0)
            std = np.std(col_data, axis=0)
            y = (col_data - mean) / std
            return y[1:]

        # find all target columns
        target_cols = []
        for label_index, data_label in enumerate(labels):
            for target_label in self.target_labels:
                if(data_label[0] == target_label[0] and  # exchange
                   data_label[1] == target_label[1] and  # product
                   data_label[2] == target_label[2]):  # column
                    print('{} {} {} added to target sequence'.format(
                        data_label[0], data_label[1], data_label[2]))
                    col = data[:, label_index]  # get column
                    col = remove_zero_vals(col)  # remove zeros before avg
                    col = np.expand_dims(col, 1)  # resize
                    target_cols.append(col)
        target_cols = np.hstack(target_cols)

        # take the average of target columns to form one target sequence
        y = np.mean(target_cols, axis=1)

        # cache raw targets
        np.save('{}\\raw_targets'.format(self.save_directory), y)

        # convert to log ratio
        targets = truncate(ln_ratio(y))
        targets = np.expand_dims(targets, 1)

        # cache targets
        np.save('{}\\targets'.format(self.save_directory), targets)
        print('Targets computed. Shape {}\n'.format(targets.shape))

        # columns to transform into logarithmic change
        log_diff_cols = [dataset_list.LOW,
                         dataset_list.HIGH,
                         dataset_list.OPEN,
                         dataset_list.CLOSE,
                         dataset_list.WEIGHTED_AVERAGE,
                         dataset_list.AVERAGE_DIFFICULTY,
                         dataset_list.AVERAGE_TOTAL_DIFFICULTY]

        # cache inputs
        inputs = np.memmap(
            '{}\\inputs'.format(self.save_directory),
            dtype='float64',
            mode='w+',
            shape=(data.shape[0] - 1, data.shape[1]))

        for col_index, label in enumerate(labels):
            x = data[:, col_index]

            if label[2] in log_diff_cols:
                print('log:', label[0], label[1], label[2])
                inputs[:, col_index] = truncate(ln_ratio(x))
            else:
                print('z-s:', label[0], label[1], label[2])
                inputs[:, col_index] = z_score(amt_change(x))

        # write to disk
        print('Inputs computed. Shape {}\n'.format(inputs.shape))
        inputs.flush()
        del(inputs)
        print('Inputs memmap of shape {}, {} saved'.format(
            data.shape[0] - 1, data.shape[1]))

        # read
        inputs = np.memmap(
            '{}\\inputs'.format(self.save_directory),
            dtype='float64',
            mode='r',
            shape=(data.shape[0] - 1, data.shape[1]))

        return inputs, targets

    def _split_data(self, data, train_ratio):
        """Split data into training and testing data.
        """
        train_data_end = int(len(data) * train_ratio)
        return data[:train_data_end], data[train_data_end:]

    def get_datasets(self,
                     input_seq_length,
                     target_seq_length,
                     train_ratio,
                     load_cache=False,
                     use_test_dir=False,
                     validate_data=False):
        """Fetch and process data and encapsulate them in DataSet objects.
        """
        if use_test_dir:
            # set this class working dir to test dir
            orig_dir = self.save_directory
            test_dir = '{}\\{}'.format(self.TEST_DIRECTORY, orig_dir)
            self.save_directory = test_dir
            # set each data class working dir to test dir
            for key in sorted(self.dataset):
                orig_dir = self.dataset[key]['dir']
                test_dir = '{}\\{}'.format(self.TEST_DIRECTORY, orig_dir)
                self.dataset[key]['dir'] = test_dir
        self.make_directory()

        # fetch raw data
        raw_data, labels = self._get_raw_data(validate_data, load_cache)

        # labels
        input_labels = labels
        target_labels = self.target_labels

        # try loading inputs and targets
        inputs = None
        targets = None
        if load_cache:
            try:
                # target data
                targets = np.load('{}\\targets.npy'.format(
                    self.save_directory))

                # input data
                inputs = np.memmap(
                    '{}\\inputs'.format(self.save_directory),
                    dtype='float64',
                    mode='r',
                    shape=(raw_data.shape[0] - 1, raw_data.shape[1]))

                print('targets.npy {}'.format(targets.shape))
                print('inputs {}'.format(inputs.shape))

            except FileNotFoundError as e:
                print('Could not load from cache\n', e)
                load_cache = False

        if not load_cache:
            # process data
            inputs, targets = self._process_data(
                raw_data, labels)
            print('Inputs shape {} Targets shape {}\n'.format(
                inputs.shape, targets.shape))

        # raw target data
        raw_targets = np.load('{}\\raw_targets.npy'.format(
            self.save_directory))
        print('raw_targets.npy {}\n'.format(raw_targets.shape))

        # get unprocessed data for reference
        raw_targets = raw_targets[1:]
        train_raw_targets, test_raw_targets = self._split_data(
            raw_targets, train_ratio)

        # split data
        train_inputs, test_inputs = self._split_data(inputs, train_ratio)
        train_targets, test_targets = self._split_data(targets, train_ratio)

        # train data
        train_dataset = DataSet(
            inputs=train_inputs,
            input_labels=input_labels,
            targets=train_targets,
            target_labels=target_labels,
            raw_targets=train_raw_targets,
            input_seq_length=input_seq_length,
            target_seq_length=target_seq_length)

        # test data
        test_dataset = DataSet(
            inputs=test_inputs,
            input_labels=input_labels,
            targets=test_targets,
            target_labels=target_labels,
            raw_targets=test_raw_targets,
            input_seq_length=input_seq_length,
            target_seq_length=target_seq_length)

        # sanity check
        self.__print_data('Sampled rows of raw data', raw_data, 5)
        self.__print_data('Sampled rows of input data', inputs, 5)
        self.__print_data('Sampled rows of raw target data', raw_targets, 5)
        self.__print_data('Sampled rows of target data', targets, 5)
#         # generate text files for debugging
#         self.set_cache('raw_data', raw_data[:1440], safe_overwrite=True)
#         self.set_cache('targets', targets[:1440], safe_overwrite=True)
#         self.set_cache('inputs', inputs[:1440], safe_overwrite=True)

        return train_dataset, test_dataset

    def __print_data(self, title, data, num_rows):
        """Console output helper method for debugging.
        """
        print(title)
        for i, datum in enumerate(data[:num_rows]):
            print('row {} : {}'.format(i, datum.tolist()))
        print('...')
        for i, datum in enumerate(data[-num_rows:]):
            print('row {} : {}'.format(len(data) - i - 1, datum.tolist()))
        print(end='\n')


if __name__ == '__main__':

    crypto = Crypto(cache_directory='data\\processed_data')

    # start date of data
    start = {'year': 2018,
             'month': 1,
             'day': 1,
             'hour': 0,
             'minute': 0}
    crypto.set_start_date(start)

    # end date of data
    end = {'year': 2018,
           'month': 1,
           'day': 2,
           'hour': 0,
           'minute': 0}
    crypto.set_end_date(end)

    # select items to put in data
    dataset = dataset_list.JUN2016
    crypto.set_dataset(dataset)

    # select the values to average for target sequence
    exchanges = [dataset_list.GDAX_CHART,
                 dataset_list.GDAX_CHART,
                 dataset_list.GDAX_CHART]
    products = ['ETH-USD',
                'ETH-BTC',
                'BTC-USD']
    labels = [(dataset_list.HIGH, dataset_list.LOW),
              (dataset_list.HIGH, dataset_list.LOW),
              (dataset_list.HIGH, dataset_list.LOW)]
    crypto.set_target(exchanges, products, labels)

    # if start and end dates in UNIX are not evenly divisible by 60 seconds,
    # the mutator methods automatically rounds the date down to the nearest t
    print('UTC Start Date  : {}'.format(crypto.start_date))
    print('UTC End Date    : {}'.format(crypto.end_date))
    print('UNIX Start Date : {}'.format(crypto.start_unix))
    print('UNIX End Date   : {}\n'.format(crypto.end_unix))

    # fetch data. this downloads data if cache file is not present
    train_dataset, test_dataset, train_data, test_data = crypto.get_datasets(
        input_seq_length=1,
        target_seq_length=1,
        train_ratio=0.5,
        load_cache=False,
        use_test_dir=True,
        validate_data=False)

    # check correctness of data shape
    print('Train Data Shape : {}'.format(train_data.shape))
    print('Test Data Shape : {}\n'.format(test_data.shape))

    print('Train Data Inputs Shape : {}'.format(train_dataset.inputs.shape))
    print('Train Data Targets Shape : {}'.format(train_dataset.targets.shape))
    print('Test Data Inputs Shape : {}'.format(test_dataset.inputs.shape))
    print('Test Data Targets Shape : {}\n'.format(test_dataset.targets.shape))

    example = train_dataset.get_example(0)
    print('Example Input Shape : {}'.format(example[0].shape))
    print('Example Target Shape : {}'.format(example[1].shape))
    print('Train Num Examples : {}'.format(train_dataset.num_examples))
    print('Test Num Examples : {}'.format(test_dataset.num_examples))
    print('Train Input Depth : {}'.format(train_dataset.input_depth))
    print('Train Target Depth : {}'.format(train_dataset.target_depth))
    print('Test Input Depth : {}'.format(test_dataset.input_depth))
    print('Test Target Depth : {}\n'.format(test_dataset.target_depth))
