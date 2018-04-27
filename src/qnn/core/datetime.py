import datetime

from .ranges import TimestampRange, IndexRange


def duration_from_string(text: str):
    unit = text[-1]
    amount = int(text[:-1])

    if unit == 'M':
        return datetime.timedelta(minutes=amount)

    elif unit == 'D':
        return datetime.timedelta(days=amount)

    else:
        raise RuntimeError('Unknown duration unit "%s"' % unit)


def find_index_range(datetime_index, timestamp_range: TimestampRange) -> IndexRange:
    begin = None
    for i, ts in enumerate(datetime_index):
        if ts >= timestamp_range.begin:
            begin = i
            break

    end = None
    for i, ts in reversed(list(enumerate(datetime_index))):
        if ts <= timestamp_range.end:
            end = i
            break

    return IndexRange(begin, end)
