from typing import Tuple, List, Dict
import os
import datetime

import json

from qnn.core.ranges import TimestampRange


class EventsDataFile(object):
    def __init__(self, folderpath: str):
        self._folderpath: str = folderpath
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

        filepath = os.path.join(self._folderpath, str(dt.year), '%d.json' % dt.month)

        if not os.path.isdir(os.path.join(self._folderpath, str(dt.year))):
            os.mkdir(os.path.join(self._folderpath, str(dt.year)))

        if os.path.isfile(filepath):
            flags = 'a'
        else:
            flags = 'w'

        self._f = open(filepath, flags, encoding='utf-8')
        self._f_year = dt.year
        self._f_month = dt.month

    def append(self, dt: datetime.datetime, d: dict):
        assert self.is_open

        if self._range.begin is None:
            self._range.begin = dt
        else:
            assert dt > self._range.begin

        self._range.end = dt

        if self._f_year != dt.year or self._f_month != dt.month:
            self.close()
            self.open(dt)

        json.dump({'dt': dt.timestamp(), 'd': d}, self._f)
        self._f.write('\n')

    def flush(self):
        assert self.is_open

        self._f.flush()
        self._write_range_info()

    def close(self):
        assert self.is_open

        self.flush()
        self._f.close()
        self._f = None

    def get_events(self, rangev: TimestampRange):
        start_year, start_month = rangev.begin.year, rangev.begin.month
        end_year, end_month = rangev.end.year, rangev.end.month

        for y in range(start_year, end_year + 1):
            for m in range(1 if start_year != y else start_month, (12 if end_year != y else end_month) + 1):
                filepath = os.path.join(self._folderpath, str(y), '%d.json' % m)

                with open(filepath, 'r', encoding='utf-8') as f:
                    line = f.readline()
                    if line is None:
                        return

                    yield json.loads(line)

    def get_latest_event(self):
        if self._range.end is None:
            return None

        filepath = os.path.join(self._folderpath, str(self._range.end.year), '%d.json' % self._range.end.month)

        line = None  # Inefficient way to get last line
        with open(filepath, 'r', encoding='utf-8') as f:
            line = f.readline()

        if line is None:
            return None

        return json.loads(line)
