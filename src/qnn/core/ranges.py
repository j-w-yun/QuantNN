from typing import Optional
import datetime


class TimestampRange(object):
    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

    def __init__(self, begin: Optional[datetime.datetime], end: Optional[datetime.datetime]):
        self.begin: Optional[datetime.datetime] = begin
        self.end: Optional[datetime.datetime] = end

    def to_dict(self) -> dict:
        return {
            'begin': self.begin.strftime(self.DATETIME_FORMAT) if self.begin else '',
            'end': self.end.strftime(self.DATETIME_FORMAT) if self.end else '',
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TimestampRange':
        return TimestampRange(datetime.datetime.strptime(d['begin'], cls.DATETIME_FORMAT) if d['begin'] else None,
                              datetime.datetime.strptime(d['end'], cls.DATETIME_FORMAT) if d['end'] else None)

    def __str__(self):
        return 'TimestampRange(%s, %s)' % (self.begin, self.end)


class IndexRange(object):
    def __init__(self, begin: int, end: int):
        self.begin: int = begin
        self.end: int = end

    def to_dict(self) -> dict:
        return {
            'begin': self.begin,
            'end': self.end,
        }

    @staticmethod
    def from_dict(d: dict) -> 'IndexRange':
        return IndexRange(int(d['begin']), int(d['end']))

    def __str__(self):
        return 'IndexRange(%d, %d)' % (self.begin, self.end)
