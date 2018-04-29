from tensorflow.contrib.rnn.python.ops.rnn_cell import LayerNormLSTMCell
from tensorflow.python.layers.core import Dense

import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq


class EncoderDecoder:

    def __init__(self,
                 num_units,
                 num_layers,
                 input_length,
                 input_depth,
                 target_length,
                 target_depth):
        """Encoder-decoder model.
        """
        self._num_units = num_units
        self._num_layers = num_layers
        self._input_length = input_length
        self._input_depth = input_depth
        self._target_length = target_length
        self._target_depth = target_depth

    @property
    def num_units(self):
        return self._num_units

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def input_length(self):
        return self._input_length

    @property
    def input_depth(self):
        return self._input_depth

    @property
    def target_length(self):
        return self._target_length

    @property
    def target_depth(self):
        return self._target_depth

    def build(self,
              input_ph,
              target_ph,
              keep_prob,
              sampling_prob):
        """Define encoder-decoder architecture.
        """
        # build encoder graph using placeholder (input) nodes
        _, encoder_states, attention_mechanism = self._build_encoder(
            input_sequence=input_ph,
            keep_prob=keep_prob)

        # build decoder graph with encoder graph outputs
        outputs, _decoder_state = self._build_decoder(
            encoder_states=encoder_states,
            target_sequence=target_ph,
            keep_prob=keep_prob,
            sampling_prob=sampling_prob,
            attention_mechanism=attention_mechanism)

        # output of the final model is the prediction
        return outputs.rnn_output

    def _single_cell(self, num_units, keep_prob, layer_index=0):
        """Create an RNN cell.
        """
#         single_cell = tf.nn.rnn_cell.LSTMCell(num_units)
#         single_cell = rnn_cell.NASCell(num_units)

#         if layer_index % 2 == 0:
#             single_cell = LayerNormLSTMCell(num_units, layer_norm=True)
#         else:
#             single_cell = tf.nn.rnn_cell.GRUCell(num_units)

        single_cell = tf.nn.rnn_cell.GRUCell(num_units)

        single_cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=single_cell, input_keep_prob=keep_prob)

        if layer_index > 0:
            single_cell = tf.nn.rnn_cell.ResidualWrapper(single_cell)

        return single_cell

    def _multi_cell(self, num_units, num_layers, keep_prob):
        """Create multiple layers of RNN cells.
        """
        cell_list = []

        for i in range(num_layers):
            cell_list.append(
                self._single_cell(
                    num_units=num_units,
                    keep_prob=keep_prob,
                    layer_index=i))

        return tf.nn.rnn_cell.MultiRNNCell(cell_list)

    def _build_encoder(self, input_sequence, keep_prob):
        """Define encoder architecture.
        """
        # connect each layer sequentially, building a graph that resembles a
        # feed-forward network made of recurrent units
        encoder_cell = self._multi_cell(
            num_units=self.num_units,
            num_layers=self.num_layers,
            keep_prob=keep_prob)

        # the model is using fixed lengths of input sequences so tile the defined
        # length in the batch dimension
        sequence_lengths = tf.tile(
            [self.input_length], [tf.shape(input_sequence)[0]])

        # build the unrolled graph of the recurrent neural network
        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(
            cell=encoder_cell,
            inputs=input_sequence,
            sequence_length=sequence_lengths,
            dtype=tf.float32)

        # attention provides a direct connection between the encoder and decoder
        # so that long-range connections are not limited by the fixed size of the
        # thought vector
        attention_layer_size = self.num_units
        attention_mechanism = seq2seq.BahdanauAttention(
            num_units=attention_layer_size,
            memory=encoder_outputs,
            memory_sequence_length=sequence_lengths,
            normalize=True)

        return (encoder_outputs, encoder_states, attention_mechanism)

    def _build_decoder(self,
                       encoder_states,
                       target_sequence,
                       keep_prob,
                       sampling_prob,
                       attention_mechanism):
        """Define decoder architecture.
        """
        # connect each layer sequentially, building a graph that resembles a
        # feed-forward network made of recurrent units
        decoder_cell = self._multi_cell(
            num_units=self.num_units,
            num_layers=self.num_layers,
            keep_prob=keep_prob)

        # connect attention to decoder
        attention_layer_size = self.num_units
        decoder = seq2seq.AttentionWrapper(
            cell=decoder_cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=attention_layer_size)

        # decoder start symbol
        decoder_raw_seq = target_sequence[:, :-1]
        prefix = tf.fill(
            [tf.shape(target_sequence)[0], 1, self.target_depth], 0.0)
        decoder_input_seq = tf.concat([prefix, decoder_raw_seq], axis=1)

        # the model is using fixed lengths of target sequences so tile the defined
        # length in the batch dimension
        decoder_sequence_length = tf.tile(
            [self.target_length],
            [tf.shape(target_sequence)[0]])

        # decoder sampling scheduler feeds decoder output to next time input
        # instead of using ground-truth target vals during training
        helper = seq2seq.ScheduledOutputTrainingHelper(
            inputs=decoder_input_seq,
            sequence_length=decoder_sequence_length,
            sampling_probability=sampling_prob)

        # output layer
        projection_layer = Dense(units=self.target_depth, use_bias=True)

        # clone encoder state
        initial_state = decoder.zero_state(
            tf.shape(target_sequence)[0], tf.float32)
        initial_state = initial_state.clone(cell_state=encoder_states)

        # wrapper for decoder
        decoder = seq2seq.BasicDecoder(
            cell=decoder,
            helper=helper,
            initial_state=initial_state,
            output_layer=projection_layer)

        # build the unrolled graph of the recurrent neural network
        outputs, decoder_state, _sequence_lengths = seq2seq.dynamic_decode(
            decoder=decoder,
            maximum_iterations=self.target_length)

        return (outputs, decoder_state)

#     def _build_encoder(self, input_sequence):
#         """Define encoder architecture.
#         """
#         # type of RNN cell to use as building block
#         cell = rnn_cell.CoupledInputForgetGateLSTMCell
# #         cell = rnn_cell.WeightNormLSTMCell
#
#         encoder_outputs = None
#         encoder_states = None
#         attention_layer_size = None
#
#         if not self.bidirectional_rnn:
#             # a list layers, each n defining the number of cells in that layer
#             cells = [cell(n, layer_norm=True) for n in self.encoder_layers]
#
#             # connect each layer sequentially, building a graph that resembles a
#             # feed-forward network made of recurrent units
#             encoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
#
#             # the model is using fixed lengths of input sequences so tile the defined
#             # length in the batch dimension
#             sequence_lengths = tf.tile(
#                 [self.input_length], [tf.shape(input_sequence)[0]])
#
#             # build the unrolled graph of the recurrent neural network
#             encoder_outputs, encoder_states = tf.nn.dynamic_rnn(
#                 cell=encoder_cell,
#                 inputs=input_sequence,
#                 sequence_length=sequence_lengths,
#                 dtype=tf.float32)
#
#             attention_layer_size = self.encoder_layers[-1]
#
#         else:
#             # a list layers, each n defining the number of cells in that layer
#             fw_cells = [cell(n, layer_norm=True) for n in self.encoder_layers]
#             bw_cells = [cell(n, layer_norm=True) for n in self.encoder_layers]
#
#             # connect each layer sequentially, building a graph that resembles a
#             # feed-forward network made of recurrent units
#             fw_encoder = tf.nn.rnn_cell.MultiRNNCell(fw_cells)
#             bw_encoder = tf.nn.rnn_cell.MultiRNNCell(bw_cells)
#
#             # the model is using fixed lengths of input sequences so tile the defined
#             # length in the batch dimension
#             sequence_lengths = tf.tile(
#                 [self.input_length], [tf.shape(input_sequence)[0]])
#
#             # build the unrolled graph of the recurrent neural network
#             ((outputs_fw, outputs_bw),
#              (state_fw, state_bw)) = tf.nn.bidirectional_dynamic_rnn(
#                 cell_fw=fw_encoder,
#                 cell_bw=bw_encoder,
#                 inputs=input_sequence,
#                 sequence_length=sequence_lengths,
#                 dtype=tf.float32)
#
#             # concatenate states and outputs of bidirectional encoder
#             encoder_outputs = tf.concat((outputs_fw, outputs_bw), 2)
#             encoder_states = []
#             for i in range(len(self.encoder_layers)):
#                 encoder_states_c = tf.concat((state_fw[i].c, state_bw[i].c), 1)
#                 encoder_states_h = tf.concat((state_fw[i].h, state_bw[i].h), 1)
#                 encoder_states.append(tf.nn.rnn_cell.LSTMStateTuple(
#                     c=encoder_states_c, h=encoder_states_h))
#             encoder_states = tuple(encoder_states)
#
#             attention_layer_size = self.encoder_layers[-1] * 2
#
#         # attention provides a direct connection between the encoder and decoder
#         # so that long-range connections are not limited by the fixed size of the
#         # thought vector
#         attention_mechanism = seq2seq.LuongAttention(
#             attention_layer_size,
#             encoder_outputs,
#             memory_sequence_length=sequence_lengths,
#             scale=True)
# #         attention_mechanism = seq2seq.BahdanauAttention(
# #             attention_layer_size,
# #             encoder_outputs,
# #             memory_sequence_length=sequence_lengths)
#
#         return (encoder_outputs, encoder_states, attention_mechanism)
#
#
#     def _build_decoder(self,
#                        encoder_states,
#                        target_sequence,
#                        output_keep_prob,
#                        sampling_prob,
#                        attention_mechanism):
#         """Define decoder architecture.
#         """
#         # type of RNN cell to use as building block
#         cell = rnn_cell.CoupledInputForgetGateLSTMCell
# #         cell = rnn_cell.WeightNormLSTMCell
#
#         cells = None
#         attention_layer_size = None
#
#         if not self.bidirectional_rnn:
#             # a list layers, each n defining the number of cells in that layer
#             cells = [
#                 tf.nn.rnn_cell.DropoutWrapper(
#                     cell(n, layer_norm=True),
#                     output_keep_prob=output_keep_prob) for n in self.decoder_layers]
#             attention_layer_size = self.encoder_layers[-1]
#
#         else:
#             # a list layers, each n defining the number of cells in that layer
#             cells = [
#                 tf.nn.rnn_cell.DropoutWrapper(
#                     cell(n * 2, layer_norm=True),
#                     output_keep_prob=output_keep_prob) for n in self.decoder_layers]
#             attention_layer_size = self.encoder_layers[-1] * 2
#
#         # connect each layer sequentially, building a graph that resembles a
#         # feed-forward network made of recurrent units
#         decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
#
#         # connect attention to decoder
#         decoder = seq2seq.AttentionWrapper(
#             cell=decoder_cell,
#             attention_mechanism=attention_mechanism,
#             attention_layer_size=attention_layer_size)
#
#         # decoder start symbol
#         decoder_raw_seq = target_sequence[:, :-1]
#         prefix = tf.fill(
#             [tf.shape(target_sequence)[0], 1, self.target_depth], 0.0)
#         decoder_input_seq = tf.concat([prefix, decoder_raw_seq], axis=1)
#
#         # the model is using fixed lengths of target sequences so tile the defined
#         # length in the batch dimension
#         decoder_sequence_length = tf.tile(
#             [self.target_length],
#             [tf.shape(target_sequence)[0]])
#
#         # decoder sampling scheduler feeds decoder output to next time input
#         # instead of using ground-truth target vals during training
#         helper = seq2seq.ScheduledOutputTrainingHelper(
#             inputs=decoder_input_seq,
#             sequence_length=decoder_sequence_length,
#             sampling_probability=sampling_prob)
#
#         # output layer
#         projection_layer = Dense(units=self.target_depth, use_bias=True)
#
#         # clone encoder state
#         initial_state = decoder.zero_state(
#             tf.shape(target_sequence)[0], tf.float32)
#         initial_state = initial_state.clone(cell_state=encoder_states)
#
#         # wrapper for decoder
#         decoder = seq2seq.BasicDecoder(
#             cell=decoder,
#             helper=helper,
#             initial_state=initial_state,
#             output_layer=projection_layer)
#
#         # build the unrolled graph of the recurrent neural network
#         outputs, decoder_state, _sequence_lengths = seq2seq.dynamic_decode(
#             decoder=decoder,
#             maximum_iterations=self.target_length)
#
#         return (outputs, decoder_state)
