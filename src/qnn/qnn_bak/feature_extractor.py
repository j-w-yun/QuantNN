import tensorflow as tf

"""
CNN keywords used as defined in:
http://cs231n.github.io/convolutional-networks/
"""


def _get_kernel(shape, name):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init, name=name)


def _get_bias(shape, name):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init, name=name)


class CryptoFeatureExtractorController:

    def __init__(self,
                 input_length,
                 input_labels,
                 kernel_sizes,
                 kernel_filters,
                 output_size):
        """This class uses convolution layers to extract features from an input
        sequence, with unique kernels trained for each 'set' of input data.

        A 'set' of data is specified by the number of unique label[0] for label
        in input_labels, which can be found by:

            number_of_data_sets = len(set(input_labels[:, 0]))

        where input_labels is a list of labels for the columns in data:

            column_1 = input_labels[0]

        where input_labels[0] is a list containing String ID from dataset_list:

            [exchange_id, product_id, info_type];

        for example:

            column 1:
                input_labels[0] = ['gdax_chart', 'ETH-USD', 'low']
            column 2:
                input_labels[1] = ['gdax_chart', 'ETH-USD', 'high']
            column 3:
                input_labels[2] = ['gdax_chart', 'ETH-USD', 'open']
            column 4:
                input_labels[3] = ['gdax_chart', 'ETH-USD', 'close']

        A new matrix is split from a single Tensorflow object that is passed in
        during self.build(). Each new matrix contains only columns with the
        same ID found in input_labels[0], such that each exchange is given a
        unique convolution kernel that shares parameters across all products
        offered by that exchange.

        Args:
            input_length (int): The length of the input sequence; in other
                words, the number of timesteps in the input sequence.
            input_labels (:obj:'list' of :obj:'list'): Labels for each column.
                It is assumed that the first String in the list of lists is
                the exchange ID specified in dataset_list.
                It is assumed that all features in each exchange is grouped in
                contiguous indices.
            kernel_sizes (:obj:'list' of int): The number of rows in
                convolution kernels.
            kernel_filters (:obj:'list' of int): The number of output channels
                in convolution kernels.
            output_size (int): The number of features in the output sequence.
        """
        self._input_length = input_length
        self._input_labels = input_labels
        self._kernel_sizes = kernel_sizes
        self._kernel_filters = kernel_filters
        self._output_size = output_size

    @property
    def kernel_sizes(self):
        return self._kernel_sizes

    @property
    def kernel_filters(self):
        return self._kernel_filters

    @property
    def input_length(self):
        return self._input_length

    @property
    def input_labels(self):
        return self._input_labels

    @property
    def output_size(self):
        return self._output_size

    def build(self, input_ph):
        """Builds a tensorflow graph with convolution applied to the input,
        where each exchange in the input is given one unique convolution kernel.

        Args:
            input_ph (:obj:'tf.Tensor'): The input tensor from which to extract
                features.

        Returns:
            Tuple of convolution output and output length.
        """
        # split the input data accordingly, to use a unique convolution kernel
        # for each exchange
        input_slices = []
        exchange_ordered_set = []
        slice_start = 0
        # loop the label and identify
        for i, label in enumerate(self.input_labels):
            if not label[0] in exchange_ordered_set:
                exchange_ordered_set.append(label[0])
                # collect each slice
                if i != 0:
                    input_slice = input_ph[:, :, slice_start:i]
                    input_slices.append(input_slice)
                slice_start = i
        # add last slice, which is skipped in the loop
        last_slice = input_ph[:, :, slice_start:]
        input_slices.append(last_slice)

        extracted_feature_list = []
        output_sequence_lengths = []
        total_output_depth = 0

        for exchange, input_slice in zip(exchange_ordered_set, input_slices):
            product_list = []
            for label in self.input_labels:
                if label[0] == exchange:
                    product_list.append(label[1])
            num_features = len(product_list)
            num_products = len(set(product_list))

            # check that features were correctly split
            print('Exchange {} has {} products, with a total of {} features'.format(
                exchange, num_products, num_features))

            # share parameters for all products per exchange
            exchange_fe = FeatureExtractor(
                input_length=self.input_length,
                input_depth=num_features,
                input_sets=num_products,
                kernel_sizes=self.kernel_sizes,
                kernel_filters=self.kernel_filters,
                name='{}_extracted_features'.format(exchange))

            extracted_feature_list.append(exchange_fe.build(input_slice))
            output_sequence_lengths.append(exchange_fe.output_length)
            total_output_depth += exchange_fe.output_depth

        # check that all output sequences are equal
        if not all(x == output_sequence_lengths[0] for x in output_sequence_lengths):
            raise ValueError('All output sequence lengths should be equal')

        # concatenate along the input depth axis
        extracted_features = tf.concat(extracted_feature_list, axis=2)

        # fully connected layer to conform convolution output to output size
        with tf.name_scope('dense'):
            dense_weight = _get_kernel(
                shape=[1, total_output_depth, self.output_size],
                name='dense_weight')
            dense_weight = tf.tile(
                dense_weight,
                [tf.shape(extracted_features)[0], 1, 1])
            dense_bias = _get_bias(
                shape=[self.output_size],
                name='dense_bias')
            dense_activation = tf.nn.leaky_relu(
                tf.matmul(extracted_features, dense_weight) + dense_bias,
                alpha=0.1)

        # check final output shape
        print('Feature-extractor output shape : {}\n'.format(
            dense_activation.shape))

        return dense_activation, output_sequence_lengths[0]


class FeatureExtractor:

    def __init__(self,
                 input_length,
                 input_depth,
                 input_sets,
                 kernel_sizes,
                 kernel_filters,
                 name=None):
        # check that number of features is evenly divisible by the number of
        # sets, since each set needs to have the same number of features
        if input_depth % input_sets != 0:
            raise ValueError('Invalid number of features {} for number of sets {}'.format(
                input_depth, input_sets))

        # number of columns for each set
        size_of_set = input_depth // input_sets

        # (input_length - kernel_size + 2 * pad_size) / stride_length + 1
        output_length = input_length - sum(kernel_sizes) + len(kernel_sizes)

        # check that convolutions will result in sequence length of at least 1
        if output_length <= 0:
            raise ValueError('Input length {} and kernel sizes {} are not compatible. The resulting output_length would be {}'.format(
                input_length, kernel_sizes, output_length))

        self._input_length = input_length
        self._input_depth = input_depth
        self._input_sets = input_sets
        self._kernel_sizes = kernel_sizes
        self._kernel_filters = kernel_filters
        self._output_length = output_length
        self._output_depth = input_sets * kernel_filters[-1]
        self._size_of_set = size_of_set
        self._name = name

    @property
    def kernel_sizes(self):
        return self._kernel_sizes

    @property
    def kernel_filters(self):
        return self._kernel_filters

    @property
    def input_length(self):
        return self._input_length

    @property
    def input_depth(self):
        return self._input_depth

    @property
    def input_sets(self):
        return self._input_sets

    @property
    def output_length(self):
        return self._output_length

    @property
    def output_depth(self):
        return self._output_depth

    @property
    def size_of_set(self):
        return self._size_of_set

    def build(self, input_ph):
        # reshape input for convolution
        x = tf.reshape(input_ph, [-1, self.input_length, self.input_depth, 1])

        last_filter = None
        last_activation = None
        layer_index = 1
        for k_size, k_filter in zip(self.kernel_sizes, self.kernel_filters):
            if layer_index == 1:
                # first convolutional layer
                with tf.name_scope('conv1'):
                    kernel = _get_kernel(
                        shape=[k_size, self.size_of_set, 1, k_filter],
                        name='kernel_{}'.format(layer_index))
                    bias = _get_bias(
                        shape=[k_filter],
                        name='bias_{}'.format(layer_index))
                    conv = tf.nn.conv2d(
                        input=x,
                        filter=kernel,
                        strides=[1, 1, self.size_of_set, 1],
                        padding='VALID')  # no padding
                    last_activation = tf.nn.leaky_relu(conv + bias, alpha=0.1)
            else:
                # subsequent convolutional layers
                with tf.name_scope('conv2'):
                    kernel = _get_kernel(
                        shape=[k_size, 1, last_filter, k_filter],
                        name='kernel_{}'.format(layer_index))
                    bias = _get_bias(
                        shape=[k_filter],
                        name='bias_{}'.format(layer_index))
                    conv = tf.nn.conv2d(
                        input=last_activation,
                        filter=kernel,
                        strides=[1, 1, 1, 1],
                        padding='VALID')  # no padding
                    last_activation = tf.nn.leaky_relu(conv + bias, alpha=0.1)

            # check shapes
            print('{} conv_{} shape : {}'.format(
                self._name, layer_index, conv.shape))

            last_filter = k_filter
            layer_index += 1

        # flatten into shape [batch_size, new_sequence_length, new_features]
        output = tf.reshape(
            last_activation,
            shape=[-1, self.output_length, self.output_depth],
            name=self._name)

        # check shapes
        print('{} output shape : {}'.format(self._name, output.shape))

        return output
