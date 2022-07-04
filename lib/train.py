"""
Main class that is used to train the NN i.e. run the optimization loop. Samples batches from the train data,
feeds to the NN, computes gradients and updates the network weights. Then computes training metrics. Per epoch,
after training, runs the validation loop by sampling batches from the validation data, and computes validation
metrics. At each relevant moment, calls the corresponding method of the History object, then calls the callbacks
passing them the history.
"""


class SupervisedTrainer:

    def __init__(self, model, opt, loss):
        self.model, self.optimizer, self.loss_fn = model, opt, loss

    def train_on_batch(self, batch, zero_grad=True, step=True):

        x, y_true = batch

        # zero the parameter gradients
        if zero_grad:
            self.optimizer.zero_grad()

        # get predictions and computes loss
        y_pred = self.model(x)
        # print(y_pred.cpu().detach().numpy().sum())
        loss = self.loss_fn(y_pred, y_true)

        # compute gradients and
        loss.backward()

        # perform optimization step

        if step:
            self.optimizer.step()

        return loss, y_pred

    def train(self, e):
        """
        The optimization loop per se
        :return:
        """
        epochs, batches_per_epoch, accumulate, data_loader, train_device, model = e

        model.to(train_device)

        for epoch in range(epochs):
            # e.emit('train_epoch_start', {'epoch': epoch})

            # Train some batches
            x = iter(data_loader)
            for i in range(batches_per_epoch):
                # e.emit('train_batch_start')

                batch = next(x)
                loss, y_pred = self.train_on_batch(batch, zero_grad=(i % accumulate) == 0,
                                                   step=((i + 1) % accumulate) == 0)
                batch_loss = loss.item()

                # save running stats
                # e.emit('train_batch_end',
                #        {'batch': batch, 'y_pred': y_pred, 'loss': batch_loss})

            # e.emit('train_epoch_end', {'n_used_batches': batches_per_epoch})
