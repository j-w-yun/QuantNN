from typing import TYPE_CHECKING, Optional, List, Dict
import logging
import time
import math

import pandas as pd
import numpy as np
import tensorflow as tf

from ..base import ISeq2SeqModel
from qnn.core.parameters import ParametersNode, IntParameter, FloatParameter, ModelChoiceParameter
from qnn.nn.networks import Network
from qnn.nn.optimizers import NN_OPTIMIZERS_MAP
from qnn.nn.util import get_num_params

logger = logging.getLogger(__name__)


class EncoderDecoderModel(ISeq2SeqModel):
    @staticmethod
    def get_parameters_template():
        return ParametersNode({
            # Neural network architecture parameters
            'num_encoder_layers': IntParameter(3),
            'encoder_layers_size': IntParameter(1024),

            'num_decoder_layers': IntParameter(3),
            'decoder_layers_size': IntParameter(1024),

            # Decoder parameters
            'decoder_dropout_keep_prob': FloatParameter(0.90),
            'decoder_sampling_prob': FloatParameter(0.10),

            # Training parameters
            'epochs': IntParameter(2),
            'learning_rate_decay_epochs': IntParameter(2),
            'batch_size': IntParameter(128),
            'optimizer': ModelChoiceParameter.from_models_map(NN_OPTIMIZERS_MAP),
        })

    def __init__(self, parameters):
        super().__init__(parameters)

        self._num_encoder_layers = self.parameters['num_encoder_layers'].v
        self._encoder_layers_size = self.parameters['encoder_layers_size'].v

        self._num_decoder_layers = self.parameters['num_decoder_layers'].v
        self._decoder_layers_size = self.parameters['decoder_layers_size'].v

        self._decoder_dropout_keep_prob = self.parameters['decoder_dropout_keep_prob'].v
        self._decoder_sampling_prob = self.parameters['decoder_sampling_prob'].v

        self._epochs = self.parameters['epochs'].v
        self._learning_rate_decay_epochs = self.parameters['learning_rate_decay_epochs'].v
        self._batch_size = self.parameters['batch_size'].v

    def fit(self,
            inputs: Dict[str, np.ndarray], targets: Dict[str, np.ndarray],
            val_inputs: Dict[str, np.ndarray]=None, val_targets: Dict[str, np.ndarray]=None) -> None:
        from .network import EncoderDecoderNetwork

        if len(inputs) != 1:
            raise RuntimeError("This model only works with exactly 1 input")

        if len(targets) != 1:
            raise RuntimeError("This model only works with exactly 1 target")

        input = list(inputs.values())[0]
        target = list(targets.values())[0]
        self._target_key = list(targets.keys())[0]
        val_target = list(val_targets.values())[0] if val_targets is not None else None

        input_seq_len, input_depth = input.shape[1], input.shape[2]
        target_seq_len, target_depth = target.shape[1], target.shape[2]
        self._target_seq_len, self._target_depth = target_seq_len, target_depth

        num_train = input.shape[0]
        num_train_batches = int(np.ceil(num_train / self._batch_size))

        learning_rate_decay_steps = num_train_batches * self._learning_rate_decay_epochs

        encoder_layers = [self._encoder_layers_size for _ in range(self._num_encoder_layers)]
        decoder_layers = [self._decoder_layers_size for _ in range(self._num_decoder_layers)]

        self._sess = tf.Session()

        with self._sess.as_default():
            self._network = Network(self._sess)

            self._input_placeholders = {}
            for k, v in inputs.items():
                self._input_placeholders[k] = self._network.create_placeholder((None, *v.shape[1:]), str(k))
            input_sequence = list(self._input_placeholders.values())[0]

            self._target_placeholders = {}
            for k, v in targets.items():
                self._target_placeholders[k] = self._network.create_placeholder((None, *v.shape[1:]), str(k))
            target_sequence = list(self._target_placeholders.values())[0]

            self._output_keep_prob_placeholder = self._network.create_placeholder((), 'output_keep_prob')
            self._sampling_prob_placeholder = self._network.create_placeholder((), 'sampling_prob')

            # build model
            encoder_decoder = EncoderDecoderNetwork(
                encoder_layers=encoder_layers,
                decoder_layers=decoder_layers,
                input_length=input_seq_len,
                input_depth=input_depth,
                target_length=target_seq_len,
                target_depth=target_depth)
            outputs = encoder_decoder.build(
                input_ph=input_sequence,
                target_ph=target_sequence,
                output_keep_prob=self._output_keep_prob_placeholder,
                sampling_prob=self._sampling_prob_placeholder)

            # cost
            cost = tf.losses.mean_squared_error(
                labels=target_sequence,
                predictions=outputs) / (self._batch_size * target_seq_len)

            optimizer = NN_OPTIMIZERS_MAP[self.parameters['optimizer'].v](self.parameters['optimizer'].parameters)

            extra_values_train = {
                self._sampling_prob_placeholder: self._decoder_sampling_prob,
                self._output_keep_prob_placeholder: self._decoder_dropout_keep_prob,
            }
            extra_values_test = {
                self._sampling_prob_placeholder: 1.0,
                self._output_keep_prob_placeholder: 1.0,
            }
            self._network.compile(outputs, target_sequence, cost, optimizer, learning_rate_decay_steps, batch_size=self._batch_size, extra_values=extra_values_train)

            start_time = time.time()

            step_list_temp = []
            train_loss_list_temp = []
            learning_rate_list_temp = []

            global_step_list = []
            learning_rate_list = []
            train_loss_list = []
            val_loss_list = []
            var_l2_distance_list = []

            def on_train_step(epoch, current_train_sample_index, n_train_samples, train_loss, train_loss_reg, current_global_step, current_learning_rate):
                stats = {
                    'epoch': epoch,
                    'current_train_sample_index': current_train_sample_index,
                    'n_train_samples': n_train_samples,
                    'train_loss': train_loss,
                    'train_loss_reg': train_loss_reg,
                    'current_global_step': current_global_step,
                    'current_learning_rate': current_learning_rate,
                    #'var_distance': self._network.compute_var_distance(),
                }

                step_list_temp.append(current_global_step)
                train_loss_list_temp.append(train_loss)
                learning_rate_list_temp.append(current_learning_rate)

                print('[step %d] Epoch %d: % 3.2f%% lr=%.6f train_loss=%g' % (current_global_step, epoch, current_train_sample_index / n_train_samples * 100.0, current_learning_rate, train_loss))

                self.onTrainStep(time.time() - start_time, stats)

            def on_val_step():
                global_step_list.append(step_list_temp[-1])
                learning_rate_list.append(np.mean(learning_rate_list_temp))
                train_loss_list.append(np.mean(train_loss_list_temp))
                var_l2_distance_list.append(self._network.compute_var_distance())

                step_list_temp.clear()
                train_loss_list_temp.clear()
                learning_rate_list_temp.clear()

                self._network.extra_values = extra_values_test

                if val_inputs is not None:
                    val_loss = self._network.test(val_inputs, val_target)
                else:
                    val_loss = math.nan
                val_loss_list.append(val_loss)

                self._network.extra_values = extra_values_train

                print('eval: var_l2_distance: %g, val_loss=%g' % (self._network.compute_var_distance(), val_loss))

                # TODO: create plot file

            self._network.fit(inputs, target, self._epochs, val_inputs, val_target,
                              train_step_callbacks=[on_train_step], val_step_callbacks=[on_val_step], eval_interval=num_train_batches)

            self._network.extra_values = extra_values_test

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        extra_feed_dict_kwargs = {
            list(self._target_placeholders.values())[0]: np.zeros((list(inputs.values())[0].shape[0], self._target_seq_len, self._target_depth)),
        }

        with self._sess.as_default():
            return {
                self._target_key: self._network.query(inputs, extra_feed_dict_kwargs=extra_feed_dict_kwargs),
            }
