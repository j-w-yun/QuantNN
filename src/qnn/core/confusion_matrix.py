import math


class ConfusionMatrix(object):
    def __init__(self):
        self.tp: int = 0
        self.fp: int = 0
        self.tn: int = 0
        self.fn: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        if self.tp + self.fp + self.tn + self.fn == 0:
            return math.nan

        return (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)

    @property
    def precision_n(self) -> int:
        return self.tp + self.fp

    @property
    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return math.nan

        return self.tp / (self.tp + self.fp)

    @property
    def actual_positives(self) -> int:
        return self.tp + self.fn

    @property
    def actual_positives_ratio(self) -> float:
        if self.n == 0:
            return math.nan

        return self.actual_positives / self.n
