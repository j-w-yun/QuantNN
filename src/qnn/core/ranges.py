import datetime


class TimestampRange(object):
    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

    def __init__(self, begin: datetime.datetime, end: datetime.datetime):
        self.begin: datetime.datetime = begin
        self.end: datetime.datetime = end

    def to_dict(self) -> dict:
        return {
            'begin': self.begin.strftime(self.DATETIME_FORMAT),
            'end': self.end.strftime(self.DATETIME_FORMAT),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TimestampRange':
        return TimestampRange(datetime.datetime.strptime(d['begin'], cls.DATETIME_FORMAT),
                              datetime.datetime.strptime(d['end'], cls.DATETIME_FORMAT))

    def __str__(self):
        return 'TimestampRange(%s, %s)' % (self.begin, self.end)


class IndexRange(object):
    def __init__(self, begin: int, end: int):
        self.begin: int = begin
        self.end: int = end

    def to_dict(self) -> dict:
        return {
            'begins': self.begin,
            'end': self.end,
        }

    @staticmethod
    def from_dict(d: dict) -> 'IndexRange':
        return IndexRange(int(d['begin_ts']), int(d['end_ts']))

    def __str__(self):
        return 'IndexRange(%d, %d)' % (self.begin, self.end)
