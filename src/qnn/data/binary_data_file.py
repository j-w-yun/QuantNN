from typing import Tuple, List, Dict
import os
import datetime
import struct

import pytz
import json

from qnn.core.ranges import TimestampRange


class BinaryDataFile(object):
    """
    This class provides a way to store data in binary files in a csv-like way.
    It will automatically write to a separate data file for every month of data.

    The files are stored in the specified folder. The class the will create a new file or append to an
    existing file automatically.

    Format refers to what you want to store. E.g. integers? or floats?
    Timestamps are added automatically to every row of data, so no need to include the timestamp in the format.
    Every value that is in a row of data has to refer to one character in the format string.
    See https://docs.python.org/3/library/struct.html for more information on the format of the string in particular.

    Example usage for writing:
        import datetime
        import pytz

        def now():
            return datetime.datetime.utcnow().replace(tzinfo=pytz.UTC)

        f = BinaryDataFile('data_folder', 'dd')  # 'dd' means two doubles (which are high precision floats)
        f.open(now())

        # Receive and write data:
        price = 123.4
        volume = 4915.0
        f.append(now(), (price, volume))

        # Always close the file when you are done writing
        f.close()


    Example usage for reading:
        # You have to specify the same format that you used when writing.
        f = BinaryDataFile('data_folder', 'dd')

        for dt, row in f.get_rows(f.range):
            # Do something with row, which in this case is a tuple with two floats (that were stored with double precision)
            # dt is the datetime of the row of data.
            pass

        # You don't need to close the file here, files opened for reading are automatically closed.
    """

    def __init__(self, folderpath: str, format: str):
        self._folderpath: str = folderpath
        self._format: str = '<Q' + format
        self._ts_size = struct.calcsize('<Q')
        self._size = struct.calcsize(self._format)
        self._range_info_path = os.path.join(self._folderpath, 'range_info.json')
        self._extra_data_path = os.path.join(self._folderpath, 'extra_data.json')
        self._range = TimestampRange(None, None)

        if not os.path.isdir(self._folderpath):
            os.mkdir(self._folderpath)

        if not os.path.isfile(self._range_info_path):
            self._write_range_info()
        else:
            with open(self._range_info_path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            self._range = TimestampRange.from_dict(d)

        self._f = None
        self._f_year = None
        self._f_month = None

    def set_extra_data(self, d: dict):
        with open(self._extra_data_path, 'w', encoding='utf-8') as f:
            json.dump(d, f)

    def get_extra_data(self):
        if not os.path.isfile(self._extra_data_path):
            return None

        with open(self._extra_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @property
    def folderpath(self):
        return self._folderpath

    @property
    def range(self) -> TimestampRange:
        return self._range

    @property
    def is_open(self):
        return self._f is not None

    def _write_range_info(self):
        with open(self._range_info_path, 'w', encoding='utf-8') as f:
            json.dump(self._range.to_dict(), f)

    def open(self, dt: datetime.datetime):
        assert not self.is_open

        filepath = os.path.join(self._folderpath, str(dt.year), '%d.bin' % dt.month)

        if not os.path.isdir(os.path.join(self._folderpath, str(dt.year))):
            os.mkdir(os.path.join(self._folderpath, str(dt.year)))

        if os.path.isfile(filepath):
            flags = 'r+b'
        else:
            flags = 'wb'

        self._f = open(filepath, flags)
        self._f_year = dt.year
        self._f_month = dt.month

    def append(self, dt: datetime.datetime, values: tuple):
        assert self.is_open

        if self._range.begin is None:
            self._range.begin = dt
        else:
            assert dt > self._range.begin

        self._range.end = dt

        if self._f_year != dt.year or self._f_month != dt.month:
            self.close()
            self.open(dt)

        self._f.write(struct.pack(self._format, (int(dt.timestamp() * 1000), *values)))

    def flush(self):
        assert self.is_open

        self._f.flush()
        self._write_range_info()

    def close(self):
        assert self.is_open

        self.flush()
        self._f.close()
        self._f = None

    def get_rows(self, rangev: TimestampRange):
        start_year, start_month = rangev.begin.year, rangev.begin.month
        end_year, end_month = rangev.end.year, rangev.end.month

        for y in range(start_year, end_year + 1):
            for m in range(1 if start_year != y else start_month, (12 if end_year != y else end_month) + 1):
                filepath = os.path.join(self._folderpath, str(y), '%d.bin' % m)

                with open(filepath, 'rb') as f:
                    bytes = f.read(self._size)
                    if bytes == b'':
                        return

                    data = struct.unpack(self._format, bytes)
                    dt = datetime.datetime.utcfromtimestamp(data[0] / 1000).replace(tzinfo=pytz.UTC)

                    if dt >= rangev.begin and dt <= rangev.end:
                        yield dt, *data[1:]

    def get_latest_row(self):
        if self._range.end is None:
            return None

        filepath = os.path.join(self._folderpath, str(self._range.end.year), '%d.bin' % self._range.end.month)

        with open(filepath, 'rb') as f:
            f.seek(-self._size, os.SEEK_END)

            bytes = f.read(self._size)
            if bytes == b'':
                return None

            data = struct.unpack(self._format, bytes)
            dt = datetime.datetime.utcfromtimestamp(data[0] / 1000).replace(tzinfo=pytz.UTC)
            return dt, *data[1:]
