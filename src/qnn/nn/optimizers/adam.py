import tensorflow as tf

from .base import INetworkOptimizer
from qnn.core.parameters import ParametersNode, FloatParameter


class AdamOptimizer(INetworkOptimizer):
    @staticmethod
    def get_parameters_template() -> ParametersNode:
        return ParametersNode({
            'max_gradient_norm': FloatParameter(5.0),
            'learning_rate_init': FloatParameter(0.0002),
            'learning_rate_decay_rate': FloatParameter(0.9),
            'learning_rate_decay_epochs': FloatParameter(2.0),
        })

    def __init__(self, parameters: ParametersNode):
        super().__init__(parameters)

        self._max_gradient_norm = self.parameters['max_gradient_norm'].v
        self._learning_rate_init = self.parameters['learning_rate_init'].v
        self._learning_rate_decay_rate = self.parameters['learning_rate_decay_rate'].v
        self._learning_rate_decay_epochs = self.parameters['learning_rate_decay_epochs'].v

    def create_train_op(self, cost, trainable_vars, learning_rate_decay_steps: int):
        # compute gradients
        gradients = tf.gradients(cost, trainable_vars)

        # clip gradients
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)

        # incremented per train step
        global_step = tf.get_variable(
            'global_step',
            shape=[],
            trainable=False,
            initializer=tf.zeros_initializer)

        # learning rate decay rate
        learning_rate = tf.train.exponential_decay(
            self._learning_rate_init,
            global_step,
            learning_rate_decay_steps,
            self._learning_rate_decay_rate,
            staircase=True)

        # optimization
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, trainable_vars), global_step=global_step)

        return train_op, global_step, learning_rate
