import os


class Cacheable:
    """Stores and fetches 2-dimensional data structures.
    """

    def __init__(self, save_directory):
        self.save_directory = save_directory

    def __encode_row(self, row, fmt_str):
        r = ''
        for i, elem in enumerate(row):
            r += fmt_str.format(elem)
            if i == len(row) - 1:
                return (r + '\n')
            r += ','

    def __decode_row(self, row, cast_op):
        r = []
        elems = row.split(',')
        for elem in elems:
            r.append(cast_op(elem))
        return r

    def _float_encode_row(self, row):
        return self.__encode_row(row, fmt_str='{:20.10f}')

    def _float_decode_row(self, row):
        return self.__decode_row(row, float)

    def _string_decode_row(self, row):
        return self.__decode_row(row, str)

    def make_directory(self):
        if not os.path.exists(self.save_directory):
            try:
                os.makedirs(self.save_directory)
            except FileExistsError:
                pass

    def get_cwd(self):
        cwd = os.getcwd()
        return '{}\\{}'.format(cwd, self.save_directory)

    def get_filepath(self, filename):
        cwd = os.getcwd()
        return '{}\\{}\\{}.txt'.format(cwd, self.save_directory, filename)

    def cache_exists(self, filename):
        try:
            filepath = '{}/{}.txt'.format(self.save_directory, filename)
            with open(filepath, 'r') as f:
                f.readline()
            return True
        except FileNotFoundError:
            return False

    def delete_cache(self, filename):
        """Deletes cached data.

        Args:
            filename: Name of file to delete.
        """
        filepath = '{}/{}.txt'.format(self.save_directory, filename)
        try:
            os.remove(filepath)
        except FileNotFoundError:
            pass

    def set_cache(self, filename, data, safe_overwrite=True):
        """Caches data.

        If file does not exist, this method creates a new file.
        If file exists, this method will overwrite the file.

        Args:
            filename: Name of file to set cached data.
            data: Data to write to cache.
        """
        self.make_directory()

        filepath = '{}/{}.txt'.format(self.save_directory, filename)

        if safe_overwrite:
            temp = '{}/temp_{}.txt'.format(self.save_directory, filename)
            with open(temp, 'w') as f:
                for row in data:
                    f.write(self._float_encode_row(row))
            if self.cache_exists(filename):
                os.remove(filepath)
            os.rename(temp, filepath)
        else:
            with open(filepath, 'w') as f:
                for row in data:
                    f.write(self._float_encode_row(row))

    def append_cache(self, filename, row):
        """Appends data to cache.

        If file does not exist, this method creates a new file.

        Args:
            filename: Name of file to append.
            row: Data to append to cache.
        """
        self.make_directory()

        filepath = '{}/{}.txt'.format(self.save_directory, filename)
        with open(filepath, 'a') as f:
            f.write(self._float_encode_row(row))

    def prepend_cache(self, filename, row):
        """Prepends data to cache.

        If file does not exist, this method creates a new file.

        Args:
            filename: Name of file to prepend.
            row: Data to prepend to cache.
        """
        self.make_directory()

        data = self.get_cache(filename)
        data.insert(0, row)
        self.set_cache(filename, data)

    def check_cache(self, filename, range_keys):
        """Checks cached data for date range.

        Args:
            filename: Name of file to check.
            range_keys : A tuple of start key element found in first column to
                signify first row of returned data and an end key element found
                in first column to signify last row of returned data.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If range_keys tuple is not length 2.
            StartKeyNotFoundError: If start key is not found.
            EndKeyNotFoundError: If end key is not found.
            StartEndKeysNotFoundError: If neither start nor end keys are found.
        """
        if len(range_keys) != 2:
            raise ValueError('Range tuple should contain two elements')

        first_row = self.get_first_cache(filename)[0]
        last_row = self.get_last_cache(filename)[0]
        if first_row > range_keys[0] and last_row < range_keys[1]:
            raise StartEndKeysNotFoundError('Neither start key {} nor end key {} found in file {}'.format(
                range_keys[0], range_keys[1], filename))
        elif first_row > range_keys[0]:
            raise StartEndKeysNotFoundError('Neither start key {} nor end key {} found in file {}'.format(
                range_keys[0], range_keys[1], filename))
        elif last_row < range_keys[1]:
            raise EndKeyNotFoundError('End key {} not found in file {}'.format(
                range_keys[1], filename))

    def get_cache(self, filename, range_keys=None):
        """Fetches cached data.

        Args:
            filename: Name of file to get.
            range_keys (optional): A tuple of start key element found in first
                column to signify first row of returned data and an end key
                element found in first column to signify last row of returned
                data.

        Returns:
            Cached data.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If range_keys tuple is not length 2.
            StartKeyNotFoundError: If start key is not found.
            EndKeyNotFoundError: If end key is not found.
            StartEndKeysNotFoundError: If neither start nor end keys are found.
        """
        if not os.path.exists(self.save_directory):
            raise FileNotFoundError('Directory {} does not exist'.format(
                self.save_directory))

        filepath = '{}/{}.txt'.format(self.save_directory, filename)
        with open(filepath, 'r') as f:
            if range_keys is None:
                data = []
                for row in f:
                    current_row = self._float_decode_row(row)
                    data.append(current_row)
                return data
            else:
                if len(range_keys) != 2:
                    raise ValueError('Range tuple should contain two elements')

                # TODO: check correctness
                first_row = self.get_first_cache(filename)[0]
                last_row = self.get_last_cache(filename)[0]
                if first_row > range_keys[0] and last_row < range_keys[1]:
                    raise StartEndKeysNotFoundError('Neither start key {} nor end key {} found in file {}'.format(
                        range_keys[0], range_keys[1], filename))
                elif first_row > range_keys[0]:
                    raise StartEndKeysNotFoundError('Neither start key {} nor end key {} found in file {}'.format(
                        range_keys[0], range_keys[1], filename))
                elif last_row < range_keys[1]:
                    raise EndKeyNotFoundError('End key {} not found in file {}'.format(
                        range_keys[1], filename))

                # start appending when start key is encountered then break when
                # end key is encountered
                data = []
                append = False
                for row in f:
                    current_row = self._float_decode_row(row)
                    if current_row[0] == range_keys[0]:
                        append = True
                    if append:
                        data.append(current_row)
                    if current_row[0] == range_keys[1]:
                        if not append:
                            raise StartKeyNotFoundError('Start key {} not found in file {}'.format(
                                range_keys[0], filename))
                        append = False
                        break
                if append:
                    raise EndKeyNotFoundError('End key {} not found in file {}'.format(
                        range_keys[1], filename))
                if len(data) == 0:
                    raise StartEndKeysNotFoundError('Neither start key {} nor end key {} found in file {}'.format(
                        range_keys[0], range_keys[1], filename))
                return data

    def get_last_cache(self, filename):
        """Fetches the last cached data.

        Args:
            filename: Name of file to get.

        Returns:
            Last row of cached data.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        if not os.path.exists(self.save_directory):
            raise FileNotFoundError(
                'Directory {} does not exist'.format(self.save_directory))

        filepath = '{}/{}.txt'.format(self.save_directory, filename)

#         with open(filepath, 'r') as f:
#             for line in f:
#                 pass
#             return self._float_decode_row(line)

        # https://stackoverflow.com/questions/3346430/what-is-the-most-efficient-way-to-get-first-and-last-line-of-a-text-file
        with open(filepath, 'rb') as f:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
            last = str(f.readline(), 'utf-8')
            return self._float_decode_row(last)

    def get_first_cache(self, filename):
        """Fetches the first cached data.

        Args:
            filename: Name of file to get.

        Returns:
            First row of cached data.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        if not os.path.exists(self.save_directory):
            raise FileNotFoundError('Directory {} does not exist'.format(
                self.save_directory))

        filepath = '{}/{}.txt'.format(self.save_directory, filename)
        with open(filepath, 'r') as f:
            first = f.readline()
            return self._float_decode_row(first)


class StartKeyNotFoundError(Exception):
    def __init__(self, message):
        super(StartKeyNotFoundError, self).__init__(message)


class EndKeyNotFoundError(Exception):
    def __init__(self, message):
        super(EndKeyNotFoundError, self).__init__(message)


class StartEndKeysNotFoundError(Exception):
    def __init__(self, message):
        super(StartEndKeysNotFoundError, self).__init__(message)
