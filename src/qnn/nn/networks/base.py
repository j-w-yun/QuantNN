from typing import TYPE_CHECKING, Union, Optional, List, Dict, Callable, Tuple, Sequence
if TYPE_CHECKING:
    from ..optimizers import INetworkOptimizer
import logging
import copy
from enum import Enum

import tensorflow as tf
import numpy as np
from qnn.nn.common import FLOATX
from qnn.nn.util import get_num_params

logger = logging.getLogger(__name__)


class NetworkMode(Enum):
    TRAIN = 1
    TEST = 2


class Network(object):
    def __init__(self, sess):
        self.sess = sess

        self._mode = NetworkMode.TRAIN
        self._placeholders = {}
        self._variables = []
        self._regularizer_terms = []

        self._num_parameters = None

    @property
    def mode(self) -> NetworkMode:
        return self._mode

    @mode.setter
    def mode(self, value: NetworkMode):
        self._mode = value

    @property
    def num_parameters(self) -> Optional[int]:
        return self._num_parameters

    def compute_var_distance(self):
        return self.sess.run(self._distance)

    def create_placeholder(self, shape, name: str):
        v = tf.placeholder(FLOATX, shape)
        self._placeholders[name] = v
        return v

    def create_variable(self, shape, initializer: Callable[[Sequence[int]], object]):
        initv = initializer(shape)
        v = tf.Variable(initv)
        self._variables.append(v)
        return v

    def create_regularizer(self, variable, regularizer):
        self._regularizer_terms.append(regularizer(variable))

    def compile(self,
                y_: Union[Dict[str, tf.Tensor], tf.Tensor],
                y: Union[Dict[str, tf.Tensor], tf.Tensor],
                loss: tf.Tensor,
                optimizer: 'INetworkOptimizer',
                learning_rate_decay_steps: int,
                batch_size: int=64,
                extra_values: Dict[tf.Tensor, np.ndarray]=None):

        self._use_dict_target = isinstance(y_, dict)

        if self._use_dict_target:
            assert isinstance(y, dict)

        self._trainable_vars = tf.trainable_variables()
        self._num_parameters = get_num_params()

        self._distance = 0
        for trainable in tf.trainable_variables():
            self._distance += tf.nn.l2_loss(trainable)

        self.y_: Union[Dict[str, tf.Tensor], tf.Tensor] = y_
        self.y: Union[Dict[str, tf.Tensor], tf.Tensor] = y
        self.loss: tf.Tensor = loss
        self.optimizer: 'INetworkOptimizer' = optimizer
        self.loss_reg: tf.Tensor = self.loss
        if len(self._regularizer_terms) > 0:
            self.loss_reg += tf.add_n(self._regularizer_terms)
        self.batch_size: int = batch_size
        self.extra_values: Dict[tf.Tensor, np.ndarray] = extra_values or {}
        self._train_step, self._global_step, self._learning_rate = optimizer.create_train_op(self.loss_reg, self._trainable_vars, learning_rate_decay_steps)

        self.sess.run(tf.global_variables_initializer())

    def fit(self,
            inputs: Union[Dict[str, np.ndarray], np.ndarray], targets: Union[Dict[str, np.ndarray], np.ndarray],
            num_epochs: int=1,
            val_inputs: Optional[dict]=None, val_targets=None,
            train_step_callbacks: list=None,
            val_step_callbacks: list=None,
            eval_interval: int=200):
        """
        Fit model on data.
        :param inputs:
        :param targets:
        :param num_epochs:
        :param val_inputs:
        :param val_targets:
        :param train_step_callbacks: Called after every training step with arguments: epoch, current_train_sample_index, n_train_samples, train_loss, train_loss_reg, current_global_step, current_learning_rate
        :return:
        """

        if isinstance(inputs, dict):
            n_train_samples = list(inputs.values())[0].shape[0]
        else:
            n_train_samples = inputs.shape[0]

        train_step_callbacks = train_step_callbacks or []
        val_step_callbacks = val_step_callbacks or []

        logger.info('Fitting on %s samples...' % n_train_samples)

        for epoch in range(num_epochs):
            print('Epoch %d...' % (epoch + 1))
            self.mode = NetworkMode.TRAIN

            shuffle = np.random.permutation(range(n_train_samples))

            for i in range(0, n_train_samples - self.batch_size, self.batch_size):
                # Set batch inputs
                feed_dict = copy.copy(self.extra_values)
                for k, v in inputs.items():
                    feed_dict[self._placeholders[str(k)]] = v[shuffle[i:i + self.batch_size]]

                # Set batch targets
                if not self._use_dict_target:
                    feed_dict[self.y] = targets[shuffle[i:i + self.batch_size]]
                else:
                    for k, v in targets.items():
                        feed_dict[self.y[k]] = v[shuffle[i:i + self.batch_size]]

                # Run train step
                _, train_loss, train_loss_reg = self.sess.run([self._train_step, self.loss, self.loss_reg], feed_dict=feed_dict)

                # Get global step and learning rate
                current_global_step, current_learning_rate = self.sess.run([self._global_step, self._learning_rate])

                # Call callbacks
                for c in train_step_callbacks:
                    c(epoch + 1, i, n_train_samples, train_loss, train_loss_reg, current_global_step, current_learning_rate)

                if current_global_step % eval_interval == 0:
                    self.mode = NetworkMode.TEST
                    for c in val_step_callbacks:
                        c()

                    self.mode = NetworkMode.TRAIN

            # End of epoch

    def test(self,
             inputs: Union[Dict[str, np.ndarray], np.ndarray], targets: Union[Dict[str, np.ndarray], np.ndarray]):
        # TODO: Use batched testing here

        feed_dict = copy.copy(self.extra_values)
        for k, v in inputs.items():
            feed_dict[self._placeholders[str(k)]] = v

        if not self._use_dict_target:
            feed_dict[self.y] = targets
        else:
            for k, v in targets.items():
                feed_dict[self.y[k]] = v

        return self.sess.run(self.loss, feed_dict=feed_dict)

    def query(self, inputs: dict, extra_feed_dict_kwargs=None):
        feed_dict = copy.copy(self.extra_values)
        if extra_feed_dict_kwargs is not None:
            feed_dict.update(extra_feed_dict_kwargs)
        for k, v in inputs.items():
            feed_dict[self._placeholders[str(k)]] = v

        if not self._use_dict_target:
            return self.y_.eval(feed_dict=feed_dict)
        else:
            items = list(self.y_.items())

            out_values = self.sess.run([item[1] for item in items], feed_dict=feed_dict)

            return {item[0]: out_value for item, out_value in zip(items, out_values)}
