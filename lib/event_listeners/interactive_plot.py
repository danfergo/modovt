import matplotlib.pyplot as plt
import torch

from experimenter import e


class InteractivePlotter:
    """
        Class used to save the plots graphs during training
    """

    def __init__(self):
        self.plots = {}

    def on_plot(self, ev):
        plot_name = ev['name']
        value = ev['value']
        if plot_name not in self.plots:
            self.plots[plot_name] = []

        self.plots[plot_name].append(value)

        plt.clf()

        # plt.subplot(1 + len(self.metrics_names), 1, i + 1)
        plt.plot(self.plots[plot_name])
        plt.title('Min: {:.5f} Max: {:.5f}'.format(min(self.plots[plot_name]), max(self.plots[plot_name])))

        # plt.ylabel(description)
        # plt.legend()

        plt.savefig(e.out('ip_' + plot_name + '.png'), dpi=150)
