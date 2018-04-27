from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt
import numpy as np


class FigureManager:
    """Manages figures to visualize training progress.
    """

    def __init__(self,
                 fig_1_train_sample_rows,
                 fig_1_rows,
                 fig_1_cols,
                 fig_2_rows,
                 fig_2_cols):
        """Initialize figures.
        """
        self.__init_figs(fig_1_train_sample_rows,
                         fig_1_rows,
                         fig_1_cols,
                         fig_2_rows,
                         fig_2_cols)

    def __init_figs(self,
                    fig_1_train_sample_rows,
                    fig_1_rows,
                    fig_1_cols,
                    fig_2_rows,
                    fig_2_cols):
        self.fig_1_train_sample_rows = fig_1_train_sample_rows
        self.fig_1_rows = fig_1_rows
        self.fig_1_cols = fig_1_cols
        self.fig_2_rows = fig_2_rows
        self.fig_2_cols = fig_2_cols
        # pyplot interactive session
        plt.ion()
        # update dynamic lines without refreshing static lines
        self.modifiable_fig_1_lines = {}
        # plots sample of inference outputs
        fig_1, axarr_1 = plt.subplots(fig_1_rows, fig_1_cols)
        axarr_1 = axarr_1.flatten()
        # remove axis labels
        for ax in axarr_1:
            ax.yaxis.set_visible(False)
            ax.xaxis.set_visible(False)
        fig_1.tight_layout()
        fig_1.subplots_adjust(hspace=0, wspace=0,
                              left=0.01, right=0.99,
                              top=0.99, bottom=0.01)
        self.fig_1 = fig_1
        self.axarr_1_train = axarr_1[:self.get_num_train_figs()]
        self.axarr_1_test = axarr_1[self.get_num_train_figs():]
        # # set fig1 window position
        # mngr = plt.get_current_fig_manager()
        # mngr.window.wm_geometry('1063x1847+-2159+5')
        # plt.pause(0.5)
        self.draw_predictions()
        # plots cost
        fig_2, axarr_2 = plt.subplots(fig_2_rows, fig_2_cols)
        axarr_2 = axarr_2.flatten()
        fig_2.tight_layout()
        fig_2.subplots_adjust(hspace=0.2, wspace=0.2)
        self.fig_2 = fig_2
        self.axarr_2 = axarr_2
        # set fig2 window position
        # mngr = plt.get_current_fig_manager()
        # mngr.window.wm_geometry('1063x1847+-1079+5')
        # plt.pause(0.5)
        self.draw_stats()

    def plot_stats(self,
                   train_data,
                   test_data,
                   train_dataset,
                   test_dataset,
                   train_losses,
                   avg_train_losses,
                   inference_losses,
                   sample_inference_losses,
                   learning_rates,
                   distances):
        """Plots all statistic figures.
        """
        # clear old data
        for ax in self.axarr_2:
            # do not refresh figs with static data
            if 'Data' not in ax.get_title():
                ax.clear()
        LINEWIDTH = 0.5
        # plot logarithmic training performance
        self.axarr_2[0].set_title('Logarithmic Total Training Loss')
        self.axarr_2[0].set_yscale('log')
        self.axarr_2[0].plot(train_losses,
                             linewidth=LINEWIDTH,
                             color='blue')
        self.axarr_2[0].text(0.99, 0.99,
                             'Min : {}'.format(min(train_losses)),
                             horizontalalignment='right',
                             verticalalignment='top',
                             fontsize=10, color='black',
                             transform=self.axarr_2[0].transAxes)
        # plot training performance
        self.axarr_2[1].set_title('Total Sample Inference Loss')
        self.axarr_2[1].plot(sample_inference_losses,
                             linewidth=LINEWIDTH,
                             color='green')
        self.axarr_2[1].text(0.99, 0.99,
                             'Last : {}'.format(sample_inference_losses[-1]),
                             horizontalalignment='right',
                             verticalalignment='top',
                             fontsize=10, color='black',
                             transform=self.axarr_2[1].transAxes)
        # plot logarithmic training performance
        self.axarr_2[2].set_title('Logarithmic Average Training Loss')
        self.axarr_2[2].set_yscale('log')
        self.axarr_2[2].plot(avg_train_losses,
                             linewidth=LINEWIDTH,
                             color='blue')
        self.axarr_2[2].text(0.99, 0.99,
                             'Min : {}'.format(min(avg_train_losses)),
                             horizontalalignment='right',
                             verticalalignment='top',
                             fontsize=10, color='black',
                             transform=self.axarr_2[2].transAxes)
        # plot training performance
        self.axarr_2[3].set_title('Recent Average Training Loss')
        self.axarr_2[3].plot(avg_train_losses[-min(len(avg_train_losses), 25):],
                             linewidth=LINEWIDTH,
                             color='blue')
        self.axarr_2[3].text(0.99, 0.99,
                             'Last : {}'.format(avg_train_losses[-1]),
                             horizontalalignment='right',
                             verticalalignment='top',
                             fontsize=10, color='black',
                             transform=self.axarr_2[3].transAxes)
        # plot logarithmic inference performance
        self.axarr_2[4].set_title('Logarithmic Inference Loss')
        self.axarr_2[4].set_yscale('log')
        self.axarr_2[4].plot(inference_losses,
                             linewidth=LINEWIDTH,
                             color='green')
        self.axarr_2[4].text(0.99, 0.99,
                             'Min : {}'.format(min(inference_losses)),
                             horizontalalignment='right',
                             verticalalignment='top',
                             fontsize=10, color='black',
                             transform=self.axarr_2[4].transAxes)
        # plot inference performance
        self.axarr_2[5].set_title('Recent Inference Loss')
        self.axarr_2[5].plot(inference_losses[-min(len(inference_losses), 25):],
                             linewidth=LINEWIDTH,
                             color='green')
        self.axarr_2[5].text(0.99, 0.99,
                             'Last : {}'.format(inference_losses[-1]),
                             horizontalalignment='right',
                             verticalalignment='top',
                             fontsize=10, color='black',
                             transform=self.axarr_2[5].transAxes)
        # learning rate of optimizer
        self.axarr_2[6].set_title('Learning Rate')
        self.axarr_2[6].plot(learning_rates,
                             linewidth=LINEWIDTH,
                             color='orange')
        self.axarr_2[6].text(0.99, 0.99,
                             'Last : {}'.format(learning_rates[-1]),
                             horizontalalignment='right',
                             verticalalignment='top',
                             fontsize=10, color='black',
                             transform=self.axarr_2[6].transAxes)
        # plot L2 distance of trainable variables
        self.axarr_2[7].set_title('L2 Distance of Trainable Variables')
        self.axarr_2[7].plot(distances,
                             linewidth=LINEWIDTH,
                             color='red')
        self.axarr_2[7].text(0.99, 0.99,
                             'Last : {}'.format(distances[-1]),
                             horizontalalignment='right',
                             verticalalignment='top',
                             fontsize=10, color='black',
                             transform=self.axarr_2[7].transAxes)

        # plot all raw data
        title = 'Raw Target Data'
        if self.axarr_2[8].get_title() != title:
            train_target = np.mean(train_data[:, :4], axis=1)
            test_target = np.mean(test_data[:, :4], axis=1)

            self.axarr_2[8].set_title(title)
            self.axarr_2[8].plot(
                np.concatenate([train_target, test_target], axis=0),
                linewidth=LINEWIDTH,
                color='green')
            self.axarr_2[8].plot(train_target,
                                 linewidth=LINEWIDTH,
                                 color='blue')
        # plot all processed data
        title = 'Log Ratio Data'
        if self.axarr_2[9].get_title() != title:
            self.axarr_2[9].set_title(title)
            self.axarr_2[9].plot(
                np.concatenate([train_dataset.targets,
                                test_dataset.targets], axis=0),
                linewidth=LINEWIDTH,
                color='green')
            self.axarr_2[9].plot(train_dataset.targets,
                                 linewidth=LINEWIDTH,
                                 color='blue')

        show = 2000
        scale = 100.0

        # plot sample perceived data
        title = 'Sample Perceived Target Data'
        if self.axarr_2[10].get_title() != title:
            train_target = np.mean(train_data[-show:, :4], axis=1)
            test_target = np.mean(test_data[:show, :4], axis=1)

            train_ratios = np.exp(train_dataset.targets[-show:, 0] / scale)
            test_ratios = np.exp(test_dataset.targets[:show, 0] / scale)

            train_perc = train_target * train_ratios
            test_perc = test_target * test_ratios

            self.axarr_2[10].set_title(title)
            self.axarr_2[10].plot(
                np.concatenate([train_perc,
                                test_perc], axis=0),
                linewidth=LINEWIDTH,
                color='green')
            self.axarr_2[10].plot(train_perc,
                                  linewidth=LINEWIDTH,
                                  color='blue')
        # plot sample ratio data
        title = 'Sample Ratio Data'
        if self.axarr_2[11].get_title() != title:
            train_ratios = np.exp(train_dataset.targets[-show:, 0] / scale)
            test_ratios = np.exp(test_dataset.targets[:show, 0] / scale)

            self.axarr_2[11].set_title(title)
            self.axarr_2[11].plot(
                np.concatenate([train_ratios,
                                test_ratios], axis=0),
                linewidth=LINEWIDTH,
                color='green')
            self.axarr_2[11].plot(train_ratios,
                                  linewidth=LINEWIDTH,
                                  color='blue')
        self.draw_stats()

    def _plot_prediction(self,
                         previous,
                         future,
                         actual,
                         predicted,
                         axis_index,
                         is_training_data):
        """Plots one prediction figure.
        """
        LINEWIDTH = 0.7
        axarr_1 = None
        modifiable_line_key = str(axis_index)
        if is_training_data:
            axarr_1 = self.axarr_1_train
            modifiable_line_key += 'train'
        else:
            axarr_1 = self.axarr_1_test
            modifiable_line_key += 'test'
        if axis_index < len(axarr_1):
            # concatenate to shift to the correct x position
            predicted_plot = np.concatenate([previous, predicted], axis=0)
            future_plot = np.concatenate([previous, future], axis=0)
            actual_plot = np.concatenate([previous, actual], axis=0)
            if modifiable_line_key in self.modifiable_fig_1_lines:
                # alter the y value of prediction
                self.modifiable_fig_1_lines[modifiable_line_key].set_ydata(
                    predicted_plot)
                axarr_1[axis_index].relim()
                axarr_1[axis_index].autoscale_view()
            else:
                # prediction values
                self.modifiable_fig_1_lines[modifiable_line_key], = axarr_1[axis_index].plot(
                    predicted_plot,
                    linewidth=LINEWIDTH,
                    color='red')
                # far future values
                axarr_1[axis_index].plot(future_plot,
                                         linewidth=LINEWIDTH,
                                         color='orange')
                # target values
                axarr_1[axis_index].plot(actual_plot,
                                         linewidth=LINEWIDTH,
                                         color='green')
                # previous values
                axarr_1[axis_index].plot(previous,
                                         linewidth=LINEWIDTH,
                                         color='blue')
                # minimum value that was plotted, excluding prediction values
                axarr_1[axis_index].text(0.99, 0.01,
                                         '{:0.2f}'.format(np.min(future)),
                                         horizontalalignment='right',
                                         verticalalignment='bottom',
                                         fontsize=6, color='black',
                                         transform=axarr_1[axis_index].transAxes)
                # maximum value that was plotted, excluding prediction values
                axarr_1[axis_index].text(0.99, 0.99,
                                         '{:0.2f}'.format(np.max(future)),
                                         horizontalalignment='right',
                                         verticalalignment='top',
                                         fontsize=6, color='black',
                                         transform=axarr_1[axis_index].transAxes)
                if is_training_data:
                    axarr_1[axis_index].set_facecolor('#ecffe6')  # light green
                else:
                    axarr_1[axis_index].set_facecolor('#e6f9ff')  # light blue

    def plot_predictions(self,
                         predictions,
                         data,
                         indices,
                         input_seq_length,
                         target_seq_length,
                         is_training_data):
        """Plots prediction figures for training and inference.
        """
        # log ratio to ratio
        scale = 100.0
        predictions = np.exp(predictions / scale)

        # plot sample outputs of train input data
        for i, index in enumerate(indices):
            # collect sample contexts
            input_start = index
            input_end = input_start + input_seq_length
            output_start = input_end - 1
            output_end = output_start + target_seq_length

            # historical ground-truth values
            show = int(target_seq_length * 1.5)
            cols = data[input_end + 1 - show:input_end + 1, :4]
            previous = (np.sum(cols, axis=1)) / 4.0

            # future ground-truth values
            show = int(target_seq_length * 1.5)
            cols = data[input_end + 1:output_end + show, :4]
            future = (np.sum(cols, axis=1)) / 4.0

            # prediction values
            prediction_vals = []
            current_val = previous[-1]
            for percent_change in predictions[i]:
                current_val = percent_change * current_val
                prediction_vals.append(current_val)
            prediction_vals = np.squeeze(prediction_vals)

            # plot each prediction figure
            self._plot_prediction(previous=previous,
                                  future=future,
                                  actual=future[:target_seq_length],
                                  predicted=prediction_vals,
                                  axis_index=i,
                                  is_training_data=is_training_data)
        self.draw_predictions()

    def get_num_train_figs(self):
        """Get the number of training sample prediction figures.
        """
        return self.fig_1_train_sample_rows * self.fig_1_cols

    def get_num_test_figs(self):
        """Get the number of test sample prediction figures.
        """
        return self.fig_1_rows * self.fig_1_cols - self.get_num_train_figs()

    def draw_predictions(self):
        """Refreshes predictions figure.
        """
        self.fig_1.canvas.draw()
        plt.pause(0.01)

    def draw_stats(self):
        """Refreshes stats figure.
        """
        self.fig_2.canvas.draw()
        plt.pause(0.01)

    def save(self, directory, filename):
        """Saves all figures as one PDF.
        """
        save_path = '{}/{}.pdf'.format(directory, filename)
        pdf = PdfPages(save_path)
        pdf.savefig(self.fig_1)
        pdf.savefig(self.fig_2)
        pdf.close()
        return save_path
