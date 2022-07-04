import yaml

from experimenter import e


class InteractiveLogger:
    """
        Class used to save the plots graphs during training
    """

    def __init__(self):
        self.logs = {}

    def on_il_patch(self, ev):
        name = ev['name']

        if name not in self.logs:
            self.logs[name] = {}

        self.logs[name] = {**self.logs[name], **ev['data']}

        # File log
        yaml.dump(self.logs[name], open(e.out('il_' + name + '.yaml'), 'w'))
