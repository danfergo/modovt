"""
Main class that is used to train the NN i.e. run the optimization loop. Samples batches from the train data,
feeds to the NN, computes gradients and updates the network weights. Then computes training metrics. Per epoch,
after training, runs the validation loop by sampling batches from the validation data, and computes validation
metrics. At each relevant moment, calls the corresponding method of the History object, then calls the callbacks
passing them the history.
"""

from experimenter import e


def train_on_batch(batch, zero_grad=True, step=True):
    optimizer, model, loss_fn = e['optimizer', 'model', 'loss']

    x, y_true = batch

    # zero the parameter gradients
    if zero_grad:
        optimizer.zero_grad()

    # get predictions and computes loss
    y_pred = model(x)
    # print(y_pred.cpu().detach().numpy().sum())
    loss = loss_fn(y_pred, y_true)

    # compute gradients and
    loss.backward()

    # perform optimization step

    if step:
        optimizer.step()

    return loss, y_pred


def train():
    """
    The optimization loop per se
    :return:
    """
    epochs, batches_per_epoch, accumulate, data_loader, train_device, model = e[
        'epochs', 'batches_per_epoch', 'batches_per_grad_update', 'data_loader', 'train_device', 'model']

    model.to(train_device)

    for epoch in range(epochs):
        e.emit('train_epoch_start', {'epoch': epoch})

        # Train some batches
        x = iter(data_loader)
        for i in range(batches_per_epoch):
            e.emit('train_batch_start')

            batch = next(x)
            loss, y_pred = train_on_batch(batch, zero_grad=(i % accumulate) == 0, step=((i + 1) % accumulate) == 0)
            batch_loss = loss.item()  # / (accumulate if ((i + 1) % accumulate == 0) else (i + 1) % accumulate)
            # print('batch loss', loss.item(), batch_loss)

            # save running stats
            e.emit('train_batch_end',
                   {'batch': batch, 'y_pred': y_pred, 'loss': batch_loss})

        e.emit('train_epoch_end', {'n_used_batches': batches_per_epoch})
