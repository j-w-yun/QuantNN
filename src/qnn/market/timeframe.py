import datetime


class Timeframe(object):
    def __init__(self, id: int, name: str, duration: datetime.timedelta):
        self._id: int = id
        self._name: str = name
        self._duration: datetime.timedelta = duration

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def duration(self) -> datetime.timedelta:
        return self._duration
