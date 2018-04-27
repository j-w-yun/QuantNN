from tensorflow.contrib.rnn.python.ops import rnn_cell
from tensorflow.python.layers.core import Dense

import tensorflow as tf


class EncoderDecoderNetwork:

    def __init__(self,
                 encoder_layers,
                 decoder_layers,
                 input_length,
                 input_depth,
                 target_length,
                 target_depth):
        """Create an encoder-decoder model.

        Args:
            encoder_layers: A list of encoder layer size.
            decoder_layers: A list of decoder layer size.
            input_length: Length of input sequence.
            input_depth: Number of input features.
            target_length: Length of target sequence.
            target_depth: Number of target features.
        """
        self._encoder_layers = encoder_layers
        self._decoder_layers = decoder_layers
        self._input_length = input_length
        self._input_depth = input_depth
        self._target_length = target_length
        self._target_depth = target_depth

    @property
    def encoder_layers(self):
        return self._encoder_layers

    @property
    def decoder_layers(self):
        return self._decoder_layers

    @property
    def num_attention_units(self):
        return self.encoder_layers[-1]

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
              output_keep_prob,
              sampling_prob):
        """Define sequence to sequence architecture.
        """
        # build encoder graph using placeholder (input) nodes
        _encoder_outputs, encoder_states, attention_mechanism = self._build_encoder(
            input_sequence=input_ph)

        # build decoder graph with encoder graph outputs
        outputs, _decoder_state = self._build_decoder(
            encoder_states=encoder_states,
            target_sequence=target_ph,
            output_keep_prob=output_keep_prob,
            sampling_prob=sampling_prob,
            attention_mechanism=attention_mechanism)

        # output of the final model is the prediction
        return outputs.rnn_output

    def _build_encoder(self, input_sequence):
        """Define encoder architecture.
        """
        # a list layers, each layer defining the number of cells in that layer
        cell = rnn_cell.LayerNormLSTMCell
        encoder_cells = [cell(n, layer_norm=True) for n in self.encoder_layers]

        # connect each layer sequentially, building a graph that resembles a
        # feed-forward network made of recurrent units
        encoder = tf.nn.rnn_cell.MultiRNNCell(encoder_cells)

        # the model is using fixed lengths of input sequences so tile the defined
        # length in the batch dimension
        encoder_sequence_lengths = tf.tile(
            [self.input_length], [tf.shape(input_sequence)[0]])

        # build the unrolled graph of the recurrent neural network
        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(
            cell=encoder,
            inputs=input_sequence,
            sequence_length=encoder_sequence_lengths,
            dtype=tf.float32)

        # attention provides a direct connection between the encoder and decoder
        # so that long-range connections are not limited by the fixed size of the
        # layers' hidden states
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            self.num_attention_units,
            encoder_outputs,
            memory_sequence_length=encoder_sequence_lengths,
            scale=True)

        return (encoder_outputs, encoder_states, attention_mechanism)

    def _build_decoder(self,
                       encoder_states,
                       target_sequence,
                       output_keep_prob,
                       sampling_prob,
                       attention_mechanism):
        """Define decoder architecture.
        """
        # a list layers, each layer defining the number of cells in that layer
        cell = rnn_cell.LayerNormLSTMCell
        decoder_cells = [
            tf.nn.rnn_cell.DropoutWrapper(
                cell(n, layer_norm=True),
                output_keep_prob=output_keep_prob) for n in self.decoder_layers]

        # connect each layer sequentially, building a graph that resembles a
        # feed-forward network made of recurrent units
        decoder = tf.nn.rnn_cell.MultiRNNCell(decoder_cells)

        # connect attention to decoder
        decoder = tf.contrib.seq2seq.AttentionWrapper(
            cell=decoder,
            attention_mechanism=attention_mechanism,
            attention_layer_size=self.num_attention_units)

        # Tensorflow bug temporary fix
        decoder_initial_state = decoder.zero_state(
            tf.shape(target_sequence)[0], tf.float32).clone(  # batch size
            cell_state=encoder_states)

        # decoder start symbol
        decoder_raw_seq = target_sequence[:, :-1]
        prefix = tf.fill([tf.shape(target_sequence)[0],
                          1, self.target_depth], 0.0)
        decoder_input_seq = tf.concat([prefix, decoder_raw_seq], axis=1)

        # the model is using fixed lengths of target sequences so tile the defined
        # length in the batch dimension
        decoder_sequence_length = tf.tile(
            [self.target_length],
            [tf.shape(target_sequence)[0]])

        # decoder sampling scheduler feeds decoder output to next time input
        # instead of using ground-truth target vals during training
        helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(
            inputs=decoder_input_seq,
            sequence_length=decoder_sequence_length,
            sampling_probability=sampling_prob)

        # output layer
        projection_layer = Dense(
            units=self.target_depth,
            use_bias=True)

        # wrapper for decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder,
            helper=helper,
            initial_state=decoder_initial_state,  # encoder_states
            output_layer=projection_layer)

        # build the unrolled graph of the recurrent neural network
        outputs, decoder_state, _sequence_length = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            maximum_iterations=self.target_length)

        return (outputs, decoder_state)
