import numpy as np
import tensorflow as tf


def get_num_params():
    """Returns the number of trainable parameters in the current graph.
    """
    total_parameters = np.sum([np.prod(v.get_shape().as_list())
                               for v in tf.trainable_variables()])
    return total_parameters
