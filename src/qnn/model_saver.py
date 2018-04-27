from tensorflow.python.training.saver import Saver as TFSaver


class Saver:
    """Manages saving and loading model variables and model settings.
    """

    def __init__(self):
        self.saver = TFSaver(max_to_keep=None)

    def restore_model(self, sess, model_filename):
        """Restores the model trainable variables from a checkpoint file.
        """
        self.saver.restore(sess, model_filename)

    def save_model(self, sess, directory, filename):
        """Save the model trainable variables as a checkpoint file.
        """
        save_path = self.saver.save(
            sess, '{}/{}.ckpt'.format(directory, filename))
        return save_path

    def save_hparam(self,
                    directory,
                    filename,
                    current_global_step,
                    current_learning_rate,
                    hparam):
        """Save the model hyperparameter as a text file.
        """
        text_filename = '{}/{}'.format(directory, filename)

        with open(text_filename, 'w') as text_file:
            print('Percent Train: {}'.format(
                hparam['percent train']), file=text_file)
            print('\n', end='', file=text_file)

            print('Encoder Layers: {}'.format(
                hparam['encoder layers']), file=text_file)
            print('Decoder Layers: {}'.format(
                hparam['decoder layers']), file=text_file)
            print('Attention Units: {}'.format(
                hparam['attention units']), file=text_file)
            print('\n', end='', file=text_file)

            print('Input Sequence Length: {}'.format(
                hparam['input sequence length']), file=text_file)
            print('Target Sequence Length: {}'.format(
                hparam['target sequence length']), file=text_file)
            print('\n', end='', file=text_file)

            print('Maximum Gradient Norm: {}'.format(
                hparam['maximum gradient norm']), file=text_file)
            print('Initial Learning Rate: {}'.format(
                hparam['initial learning rate']), file=text_file)
            print('Learning Rate Decay Steps: {}'.format(
                hparam['learning rate decay steps']), file=text_file)
            print('Learning Rate Decay Rate: {}'.format(
                hparam['learning rate decay rate']), file=text_file)
            print('\n', end='', file=text_file)

            print('Dropout Keep Probability: {}'.format(
                hparam['dropout keep probability']), file=text_file)
            print('Decoder Sampling Probability: {}'.format(
                hparam['decoder sampling probability']), file=text_file)
            print('\n', end='', file=text_file)

            print('Batch Size: {}'.format(
                hparam['batch size']), file=text_file)
            print('Number of Training Batches: {}'.format(
                hparam['number of training batches']), file=text_file)
            print('\n', end='', file=text_file)

            print('Global Step: {}'.format(
                current_global_step), file=text_file)
            print('Last Learning Rate: {}'.format(
                current_learning_rate), file=text_file)
