import datetime
import math

import os
import shutil
import sys, subprocess
import yaml


# helpful info: how to inspect function name/location
# print('fn, ', fn.__name__)
# print('fn, ', fn.__module__)
# print('fn, ', inspect.getmodule(fn))
class EManager:

    def __init__(self):
        self.configs = {}
        self.stack = []
        self.meta = {}

    def key(self):
        return self.meta['key']

    def ws(self, path=None, *args):
        path = '' if path is None else (self[path] if path in self else path)
        return os.path.join(self.meta['ws'] + '/' + path, *args)

    def out(self, path=None, *args):
        path = '' if path is None else (self[path] if path in self else path)
        return os.path.join(self.meta['out'] + '/' + path, *args)

    def __getitem_recursive__(self, item, stack_idx=0):
        try:
            return self.stack[len(self.stack) - 1 - stack_idx][item]
        except KeyError as e:
            if stack_idx + 1 > len(self.stack) - 1:
                return self.__getitem_recursive__(item, stack_idx + 1)
            else:
                raise e

    def __getitem__(self, item):
        try:
            return self.__getitem_recursive__(item)
        except KeyError as e:
            raise KeyError('Configuration "' + str(e.args[0]) + '" is missing. Check the experimenter configurations.')

    def __getattr__(self, item):
        return self[item]

    def emit(self, event_name, args={}):
        [self.stack[len(self.stack) - 1 - i].emit(event_name, args) for i in range(len(self.stack))]


e = EManager()


class EConfigBlock:

    def __init__(self, description=None, config=None, event_listeners=None):
        self.description = description.strip() or ''
        self.config = config or {}
        self.event_listeners = event_listeners
        self.running_event_listeners = []
        self.running_parent = None

        self.title = self.description.split('\n')[0][:100]

    def start(self):
        if self.event_listeners is not None:
            self.running_event_listeners = self.event_listeners()

    def stop(self):
        self.running_event_listeners = []

    def emit(self, event_name, args=None):
        args = args or {}
        [getattr(el, 'on_' + event_name)(args)
         for el in self.running_event_listeners
         if hasattr(el, 'on_' + event_name)]
        [getattr(el, 'on__all')(args)
         for el in self.running_event_listeners
         if hasattr(el, 'on__all')]

    def __getitem__(self, item):
        if type(item) is tuple:
            return tuple([self.config[item_] for item_ in item])
        return self.config[item]

    # Mapping
    #
    # def __len__(self):
    #     return len(self.config)
    #
    # def __iter__(self):
    #     return iter(self.config)
    # try:
    # except KeyError as e:
    #     raise KeyError('Configuration "' + e.args[0] + '" is missing. Check the experimenter configurations.')
    # def push_config(self, config, event_listeners):
    #     def parse(key):
    #         self.config[key[1:-1]] = self.config[key]()
    #         self.config.pop(key)
    #
    #     self.config = config
    #     # {parse(key) for key in list(self.config) if key.startswith('{') and key.endswith('}')}
    #
    #     # self.event_listeners = event_listeners()


def experiment(description, config, event_listeners=None):
    """
        The decorator
    """

    def _(fn):
        # runs with the decorator, i.e. when/where the function is declared.

        # add config to the configs list, for documentation
        e.configs[fn] = config

        e_config = EConfigBlock(
            description=description,
            config=config,
            event_listeners=event_listeners
        )


        def _fn(*args, **kwargs):
            # wraps the function. executed when the function is executed
            # print('START ! push stack', e.stack)

            # push config into the stack
            e_config.start()
            e.stack.append(e_config)

            # print('START ! push stack', e.stack)
            # run function
            r = fn(*args, **kwargs)

            # print('STOP ! push stack', e.stack)

            # pop config from the stack
            e_config.stop()
            e.stack.pop(len(e.stack) - 1)

            return r

        _fn.__fn__ = fn
        _fn.__e_config__ = e_config
        # _fn.__meta__ = {
        #     'description': description,
        #     'config': config,
        #     'event_listeners': event_listeners
        # }

        return _fn

    return _


def check_experiment_exists(title, append):
    workspace_path = os.getcwd()
    outputs_path = workspace_path + '/outputs/'
    experiment_dirs = os.listdir(outputs_path)
    for e_dir in experiment_dirs:
        if title == e_dir.split(' - ')[1]:
            if append:
                return e_dir
            else:
                print('')
                print('THIS EXPERIMENT ALREADY EXISTS!')
                print('---')
                print(title)
                print('---')
                print('Erase it and continue? [Y/n]')
                if input() == 'Y':
                    shutil.rmtree(outputs_path + e_dir)
                    print('done.')
                else:
                    return False
    return True


def run(fn, append=False, tmp=False, open_e=False):
    e_config = fn.__e_config__

    dt = check_experiment_exists(e_config.title, append)

    if not dt:
        return

    # calculates YYYYMMDDHHMM string
    # currentDT = datetime.datetime.strptime(dt.split(' - ')[1], "%Y-%m-%d %H:%M:%S") \
    workspace_path = '/tmp' if tmp else os.getcwd()
    experiment_key = dt \
        if append and dt is not True else \
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - ' + e_config.title

    outputs_path = workspace_path + '/outputs/'
    experiment_path = outputs_path + experiment_key
    e.meta = {
        'key': experiment_key,
        'ws': workspace_path,
        'out': experiment_path + '/out'
    }

    # ensure experiments outputs folder exists
    if not os.path.isdir(outputs_path):
        raise Exception("The outputs folder for the current working dir does not exist (or isn't a folder).")

    if not (append and dt is not True):
        # create current experiment folder
        os.mkdir(experiment_path)
        os.mkdir(e.out())

        f = open(experiment_path + "/readme.md", "w+")
        f.write(e_config.description)
        f.close()


        # copy src folder
        shutil.copytree(workspace_path + '/src', experiment_path + '/src')

    f = open(experiment_path + "/~running", "w+")
    f.close()

    # dump configs
    with open(experiment_path + "/out/e_config.yaml", "w+", encoding="utf-8") as f:
        yaml.dump(e.configs, f)

    # open folder
    if open_e:
        open_experiment(experiment_path)

    # prepare args
    fn()

    # remove /~running file
    os.remove(experiment_path + "/~running")

    e.emit('e_end')


def open_experiment(experiment_path):
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.call([opener, experiment_path])


def query(tail=5):
    workspace_path = os.getcwd()
    outputs_path = workspace_path + '/outputs/'
    experiment_dirs = os.listdir(outputs_path)

    start = 0 if tail == -1 else len(experiment_dirs) - tail

    cs = [5, 3, 20, 30]
    description_max = 140
    description_idx = 3
    ct = ['IDX', 'STA', 'DATE', 'DESCRIPTION']
    alg = ['rjust', 'ljust', 'ljust', 'ljust']

    def row(cells, sep=' | ', pad=' '):
        print(sep.join([getattr(cells[c], alg[c])(cs[c], pad) for c in range(len(ct))]))

    def line(char='-'):
        row(['' for _ in ct], sep=''.ljust(3, char), pad=char)

    row(ct)
    line()
    for i in range(start, len(experiment_dirs)):
        e_dir = experiment_dirs[i]
        description = open(outputs_path + '/' + e_dir + '/readme.md', "r").read().strip()[:description_max] \
            .replace('\n', '').replace('\t', '')
        description_lines = math.ceil(len(description) / cs[description_idx])
        running_exists = os.path.exists(outputs_path + '/' + e_dir + '/~running')

        for ln in range(description_lines):
            d_start = ln * cs[description_idx]
            row([
                str(i) if ln == 0 else '',
                ('RUN' if running_exists else 'DON') if ln == 0 else '',
                e_dir if ln == 0 else '',
                description[d_start:d_start + cs[description_idx]]
            ])


if __name__ == '__main__':
    query()
