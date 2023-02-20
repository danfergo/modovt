import os
import random
from os import path

import cv2
import numpy as np
import yaml


class ReplayBuffer:

    def __init__(self, m_path=None, max_episodes=None):
        self.m_path = m_path
        self.observation_names = []
        self.data = {}
        self.max_episodes = max_episodes

        self.meta = {
            'data_keys': [],
        }

        # load existing memory.
        if m_path is not None and path.exists(path.join(self.m_path, 'meta.yaml')):
            self.meta = yaml.full_load(open(path.join(self.m_path, 'meta.yaml')))

            self.data = {
                k: (np.load(path.join(self.m_path, f'{k}.npy')).tolist()
                    if path.exists(path.join(self.m_path, f'{k}.npy'))
                    else None)
                for k in (self.meta['data_keys'])
            }

    def __len__(self):
        return len(self.data['_i_episodes']) if '_i_episodes' in self.data else 0

    def start_episode(self):
        self.append_item(f'_episodes_starts', len(self))

        # print('-> ', len(self), self.current_episode(), self.max_episodes)
        if self.current_episode() >= self.max_episodes:
            first_episode_length = self.all('_episodes_starts')[1]

            # print('>> first episode length >> ', first_episode_length)
            # print('DATA > ', self.data)

            episodes_ends = np.array(self.data['_episodes_starts'])
            self.data = {
                k: np.array(v)[first_episode_length:].tolist()
                for k, v in self.data.items()
                if k != '_episodes_starts'
            }

            # print('DATA > ', self.data)
            self.data['t'] = (np.array(self.data['t']) - 1).tolist()
            self.data['_i_episodes'] = (np.array(self.data['_i_episodes']) - first_episode_length).tolist()
            self.data['_episodes_starts'] = (episodes_ends[1:] - first_episode_length).tolist()

    def current_episode(self):
        return (len(self.data['_episodes_starts']) - 1) if '_episodes_starts' in self.data else 0

    def episode_end_of(self, t_start):
        if '_episodes_starts' in self.data:
            episode = self.data['_i_episodes'][t_start]
            if len(self.data['_episodes_starts']) > episode + 1:
                return self.data['_episodes_starts'][episode + 1] - 1
            else:
                return len(self)
        else:
            return len(self)

    def append_item(self, n, v):
        img_data = type(v) == np.array and len(v.shape) == 3 and v.shape[2] == 3

        if n not in self.data:
            self.data[n] = None if img_data else []
            if img_data:
                os.mkdir(self.m_path, n)

        if img_data:
            cv2.imwrite(path.join(self.m_path, n, f'{len(self)}.jpg'), v)
        else:
            self.data[n].append(v)
            if self.m_path is not None:
                np.save(path.join(self.m_path, f'{n}.npy'), np.array(self.data[n]))

    def get_item(self, n, i):
        if n not in self.data:
            raise KeyError(f'Key «{n}» was not found in memory data.')

        if self.data[n] is None:
            return cv2.imread(path.join(self.m_path, n, f'{i}.jpg'))
        else:
            return self.data[n][i]

    def append(self, data):
        """
            records tuples of state, action, reward,
            state, action: is
                - a dict of  ND numpy arrays, or scalars
            reward:
                - a list of numbers
        """
        keys = []
        for k in data:
            if type(data[k]) == dict:
                [self.append_item(f'{k}_{name}', value) for name, value in data[k].items()]
            else:
                self.append_item(k, data[k])
            keys.append(k)

        self.append_item('_i_episodes', self.current_episode())
        self.append_item('t', len(self.data['_i_episodes']) - 1)

        self.meta['data_keys'] = keys

        if self.m_path is not None:
            with open(path.join(self.m_path, 'meta.yaml'), 'w') as f:
                yaml.dump(self.meta, f)

    def __getitem__(self, i):
        return {
            k: self.get_item(k, i)
            for (k, v) in self.data.items() if v is not None
        }

    def all(self, *args):
        if len(args) == 1:
            return self.data[args[0]]

        return list(zip(*tuple([self.data[a] for a in args])))

    def sample(self, batch_size, *args):
        samples = random.sample(self.data['t'], batch_size)

        return list(zip(*tuple([[self.data[a][i] for i in samples] for a in args])))

