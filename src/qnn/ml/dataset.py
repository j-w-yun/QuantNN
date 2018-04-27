from typing import Dict, List, Tuple
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np


class IDataSet(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    @property
    def inputs(self):
        raise NotImplementedError

    @abstractmethod
    @property
    def targets(self):
        raise NotImplementedError

    @abstractmethod
    @property
    def input_seq_length(self):
        raise NotImplementedError

    @abstractmethod
    @property
    def target_seq_length(self):
        raise NotImplementedError

    @abstractmethod
    @property
    def num_examples(self):
        raise NotImplementedError

    @abstractmethod
    @property
    def input_shapes(self):
        raise NotImplementedError

    @abstractmethod
    @property
    def target_shapes(self):
        raise NotImplementedError

    @abstractmethod
    def get_example(self, index):
        raise NotImplementedError

    @abstractmethod
    def get_example_set(self, indices):
        raise NotImplementedError

    @abstractmethod
    def split(self, splits: OrderedDict[str, float], no_overlap: bool=True):
        raise NotImplementedError


class DataSet(IDataSet):
    """Separately contains inputs and targets for a dataset.
    """

    def __init__(self,
                 inputs: Dict[str, np.ndarray],
                 targets: Dict[str, np.ndarray],
                 input_seq_length: int,
                 target_seq_length: int):
        super().__init__()

        assert len(inputs) != 0
        assert len(targets) != 0

        self._shape0 = next(iter(inputs.values())).shape[0]

        for k, v in inputs.items():
            assert v.shape[0] == self._shape0, ('inputs[%s].shape: %s, should have shape[0]=%d' % (k, v.shape, self._shape0))

        for k, v in targets.items():
            assert v.shape[0] == self._shape0, ('targets[%s].shape: %s, should have shape[0]=%d' % (k, v.shape, self._shape0))

        self._inputs: Dict[str, np.ndarray] = inputs
        self._targets: Dict[str, np.ndarray] = targets
        self._input_seq_length: int = input_seq_length
        self._target_seq_length: int = target_seq_length
        self._input_shapes: dict = {k: v.shape[1:] for k, v in inputs.items()}
        self._target_shapes: dict = {k: v.shape[1:] for k, v in targets.items()}

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

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
        return self._shape0 - (self.input_seq_length + self.target_seq_length - 1)

    @property
    def input_shapes(self):
        """Input shapes.
        """
        return self._input_shapes

    @property
    def target_shapes(self):
        """Target shapes.
        """
        return self._target_shapes

    def get_example(self, index):
        """Retrieve single input-target data pair.
        """
        input_start = index
        input_end = input_start + self.input_seq_length
        output_start = input_end
        output_end = output_start + self.target_seq_length

        x = {}
        y = {}

        for k, v in self._inputs.items():
            x[k] = np.reshape(v[input_start:input_end], newshape=(1, self.input_seq_length, *v.shape[1:]))

        for k, v in self._targets.items():
            y[k] = np.reshape(v[output_start:output_end], newshape=(1, self.target_seq_length, *v.shape[1:]))

        return x, y

    def _get_example_indices(self, index):
        input_start = index
        input_end = input_start + self.input_seq_length
        output_start = input_end
        output_end = output_start + self.target_seq_length

        return input_start, input_end, output_start, output_end

    def get_example_set(self, indices):
        """Retrieve a set of input-target data dataset_list.
        """

        batch_x, batch_y = {}, {}

        for k, v in self._inputs.items():
            values = []
            for index in indices:
                input_start, input_end, output_start, output_end = self._get_example_indices(index)
                values.extend(v[input_start:input_end])

            batch_x[k] = np.reshape(values, newshape=(-1, self.input_seq_length, *v.shape[1:]))

        for k, v in self._targets.items():
            values = []
            for index in indices:
                input_start, input_end, output_start, output_end = self._get_example_indices(index)
                values.extend(v[output_start:output_end])

            batch_y[k] = np.reshape(values, newshape=(-1, self.target_seq_length, *v.shape[1:]))

        return batch_x, batch_y

    def split(self, splits: OrderedDict[str, float], no_overlap: bool=True):
        assert sum(list(splits.values())) <= 1.0

        ret = OrderedDict()

        n = self.num_examples
        indices = list(range(n))

        from_n = 0
        for k, v in splits.items():
            up_to_n = from_n + int(n * v)
            assert up_to_n <= n

            ret[k] = DataSetView(self, indices[from_n:(up_to_n if not no_overlap else up_to_n - self.target_seq_length)])
            from_n = up_to_n

        return ret


class DataSetView(IDataSet):
    def __init__(self, dataset: DataSet, indices: List[int]):
        super().__init__()

        self._dataset: DataSet = dataset
        self._indices: List[int] = indices

    @property
    def inputs(self):
        ret: Dict[str, np.ndarray] = {
            k: np.reshape([v[i] for i in self._indices], newshape=(len(self._indices), self.input_seq_length, *self.input_shapes[k])) for k, v in self._dataset.inputs.items()
        }
        return ret

    @property
    def targets(self):
        ret: Dict[str, np.ndarray] = {
            k: np.reshape([v[i] for i in self._indices], newshape=(len(self._indices), self.target_seq_length, *self.target_shapes[k])) for k, v in self._dataset.targets.items()
        }
        return ret

    @property
    def input_seq_length(self):
        return self._dataset.input_seq_length

    @property
    def target_seq_length(self):
        return self._dataset.target_seq_length

    @property
    def num_examples(self):
        return len(self._indices)

    @property
    def input_shapes(self):
        return self._dataset.input_shapes

    @property
    def target_shapes(self):
        return self._dataset.target_shapes

    def get_example(self, index):
        return self._dataset.get_example(self._indices[index])

    def get_example_set(self, indices):
        return self._dataset.get_example_set([self._indices[i] for i in indices])

    def split(self, splits: OrderedDict[str, float], no_overlap: bool=True):
        assert sum(list(splits.values())) <= 1.0

        ret = OrderedDict()

        n = self.num_examples
        indices = self._indices

        from_n = 0
        for k, v in splits.items():
            up_to_n = from_n + int(n * v)
            assert up_to_n <= n

            ret[k] = DataSetView(self._dataset, indices[from_n:(up_to_n if not no_overlap else up_to_n - self.target_seq_length)])
            from_n = up_to_n

        return ret
