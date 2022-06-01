import torch

from experimenter import e


class ModelsSaver:

    def __init__(self):
        self.model = e.model
        self.latest_loss = None

    def save_model(self, name):
        torch.save(self.model.state_dict(), e.out(name))

    def on_validation_end(self, ev):
        loss = ev.loss
        if loss < self.latest_loss or self.latest_loss is None:
            self.save_model('best_model')

    def on_epoch_end(self):
        self.save_model('latest_model')
