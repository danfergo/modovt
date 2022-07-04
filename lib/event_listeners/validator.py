from experimenter.stats import Stats

from experimenter import e


class Validator:

    def __init__(self):
        self.stats = Stats(len(e.metrics))
        self.metrics = e.metrics
        self.loss_fn = e.loss
        self.loader = e.val_loader
        self.n_val_batches = e.n_val_batches
        self.model = e.model

    def on_train_epoch_start(self, ev):
        self.stats.set_current_epoch(ev['epoch'])
        self.stats.reset_running_stats('train')

    def on_train_batch_end(self, ev):
        x, y_true = ev['batch']
        y_pred = ev['y_pred']
        self.stats.update_running_stats('train', ev['loss'], [m(y_pred, y_true).item() for m in self.metrics])

    def on_train_epoch_end(self, ev):
        self.stats.normalize_running_stats('train', ev['n_used_batches'])

        self.stats.reset_running_stats('val')

        b = iter(self.loader)
        for i in range(self.n_val_batches):
            x, y_true = next(b)

            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y_true)

            self.stats.update_running_stats('val', loss.item(), [m(y_pred, y_true).item() for m in self.metrics])

        self.stats.normalize_running_stats('val', self.n_val_batches)

        e.emit('validation_end', {'history': self.stats})
