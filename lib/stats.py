class Stats:
    def __init__(self, n_metrics):
        self.n_metrics = n_metrics
        self.data = {
            'epoch': 0,
            'train': {
                'loss': 0.0,
                'metrics': [],
                'all_losses': [],
                'all_metrics': [[] for _ in range(n_metrics)]
            },
            'val': {
                'loss': 0.0,
                'metrics': [],
                'all_losses': [],
                'all_metrics': [[] for _ in range(n_metrics)]
            }
        }

    def set_current_epoch(self, epoch):
        self['epoch'] = epoch

    def reset_running_stats(self, phase):
        self[phase]['loss'] = 0.0
        self[phase]['metrics'] = [0.0 for _ in range(self.n_metrics)]

    def update_running_stats(self, phase, loss, metrics):
        self[phase]['loss'] += loss
        self[phase]['metrics'] = [self[phase]['metrics'][m] + metrics[m] for m in range(len(metrics))]

    def normalize_running_stats(self, phase, n_batches):
        self[phase]['loss'] /= n_batches

        def normalize(i, n):
            self[phase]['metrics'][i] /= n

        [normalize(i, n_batches) for i in range(self.n_metrics)]

        self[phase]['all_losses'].append(self[phase]['loss'])
        [self[phase]['all_metrics'][i].append(self[phase]['metrics'][i]) for i in range(self.n_metrics)]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, item):
        return self.data[item]
