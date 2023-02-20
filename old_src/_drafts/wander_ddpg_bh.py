import itertools
import random

import yarok
from yarok import wait, PlatformMJC
from math import pi

from yarok.components.geltip.geltip import GelTip
from yarok.components.robotiq_2f85.robotiq_2f85 import robotiq_2f85
from yarok.components.ur5.ur5 import UR5

import cv2
import numpy as np

import experimenter
from experimenter import experiment
from lib.event_listeners.exp_board.e_board import EBoard
from lib.event_listeners.interactive_logger import InteractiveLogger
from lib.event_listeners.interactive_plot import InteractivePlotter
from old_src.agent.nn import minimal_discrete_actor, minimal_dynamics_model, minimal_continuous_actor, minimal_q_fn
from old_src.agent.rewards import TactilePainPleasureReward
from old_src.rl_algorithms.ddpg import DDPG, ReplayBuffer
from old_src.utils.geometry import distance
from old_src.utils.img import show
from old_src.world.manipulator_world import ManipulatorWorld

from experimenter import e


class WanderBehaviour:

    def __init__(self,
                 arm: UR5,
                 gripper: robotiq_2f85,
                 finger_yellow: GelTip,
                 finger_blue: GelTip):
        self.arm = arm
        self.gripper = gripper
        self.fingerY = finger_yellow
        self.fingerB = finger_blue
        self.finger1_bkg = None
        self.finger2_bkg = None

        self.tactileReward = TactilePainPleasureReward()

        self.initial_q = [pi / 2,
                          -pi / 2 - pi / 4,
                          pi / 2 - pi / 4,
                          0,
                          pi / 2,
                          - pi / 2]
        self.working_space = [
            [pi / 2 - pi / 4, pi / 2 + pi / 4],
            [(-pi / 2 - pi / 4) - pi / 8, (-pi / 2 - pi / 4) + pi / 8],  # shoulder
            [pi / 2 - pi / 4 - pi / 4, pi / 2 - pi / 4 + pi / 4],
            [0 - pi / 4, 0 + pi / 4],
            [pi / 2 - pi / 4, pi / 2 + pi / 4],
            [- pi / 2 - pi / 4, - pi / 2 + pi / 4],
        ]

        self.rl_learner = DDPG(
            minimal_q_fn(6, 6),
            minimal_continuous_actor(6, 6),
        )

    def clip_workspace(self, q):
        return [max(self.working_space[i][0], min(q[i], self.working_space[i][1])) for i in range(len(q))]

    def rnd_rel_qi(self, i):
        delta = 0.05
        q = self.target_q[i]
        # return random.uniform(max(self.working_space[i][0], q - delta), min(self.working_space[i][1], q + delta))
        return random.uniform(max(self.working_space[i][0], q - delta), min(self.working_space[i][1], q + delta))

    def rnd_rel_discrete_q(self, i):
        q = [qi for qi in self.target_q]
        q[i] = self.rnd_rel_qi(i)
        return q

    def rnd_rel_q(self):
        return [
            self.rnd_rel_qi(i)
            for i in range(len(self.target_q))
        ]

    def rnd_q(self):
        return [
            random.uniform(self.working_space[i][0], self.working_space[i][1])
            for i in range(6)
        ]

    """
        Used for testing, random exploration 
    """

    def just_wander(self):
        while True:
            self.target_q = self.rnd_rel_q()
            self.arm.move_q(self.target_q)

            finger1 = self.fingerY.read()
            show('finger1', finger1)

            finger1_diff = self.finger1_bkg - finger1

            ret, mask = cv2.threshold(finger1_diff,
                                      np.max(finger1_diff) / 2,
                                      np.max(finger1_diff), cv2.THRESH_BINARY)

            show('finger1_diff', mask)
            cv2.setWindowTitle('finger1_diff',
                               str(np.min(finger1_diff))
                               + ' --- '
                               + str(np.max(finger1_diff)))

            wait(lambda: self.arm.is_at(self.target_q))

    def wake_up(self):
        self.gripper.move(0)
        # self.arm.move_q(self.initial_q)
        # wait(lambda: self.arm.is_at(self.target_q))  # and self.gripper.is_at(0))

        # store background
        self.finger1_bkg = cv2.resize(self.fingerY.read(), (320, 240))
        self.finger2_bkg = cv2.resize(self.fingerB.read(), (320, 240))

        def get_observation():
            return self.arm.at()

        def get_reward():
            # depthY = cv2.resize(self.fingerY.read(), (320, 240))
            # depthB = cv2.resize(self.fingerB.read(), (320, 240))
            #
            # rel_depthY = self.finger1_bkg - depthY
            # rel_depthB = self.finger2_bkg - depthB
            #
            # rY, _, rY_map = self.tactileReward.reward(rel_depthY, return_maps=True)
            # rB, _, rB_map = self.tactileReward.reward(rel_depthB, return_maps=True)

            # show('depth_mapY', depthY)
            # show('depth_mapB', depthB)
            # cv2.setWindowTitle('tactile_mapY', "Yellow. R: {:.2f}".format(rY))
            # cv2.setWindowTitle('tactile_mapB', "Blue. R: {:.2f}".format(rB))
            # show('tactile_mapY', self.tactileReward.tactile_color_map(rY_map))
            # show('tactile_mapB', self.tactileReward.tactile_color_map(rB_map))

            # r = rY + rB - 0.5
            # print('reward: ', r)

            p = [round(x, 3) for x in [pi / 2,
                                       -pi / 2,
                                       pi / 2 - pi / 4,
                                       0,
                                       pi / 2,
                                       - pi / 2
                                       ]]
            st = [round(x, 3) for x in self.arm.at()]
            d = distance(p, st)
            r = 1000 if d == 0 else 1 / d
            # print(r)
            # print(st, distance([pi / 2, -pi / 2, pi / 2 - pi / 4, 0, pi / 2, - pi / 2], st))
            return r

        # def clip(a):

        def take_action(a):
            self.target_q += np.array(a) * 0.1
            self.target_q = self.clip_workspace(self.target_q)
            self.arm.move_q(self.target_q)
            wait(lambda: self.arm.is_at(self.target_q))

        def reset():
            self.target_q = self.rnd_q()
            self.arm.move_q(self.target_q)
            wait(lambda: self.arm.is_at(self.target_q))  # and self.gripper.is_at(0))

        reset()

        # clip,
        self.rl_learner.run(get_observation,
                            get_reward,
                            take_action,
                            reset)


@experiment(
    """
        Testing DDPG with simple reaching position goal (Tuesday), lr 1.0, after fix r*=-1_over_r - 10000step , 1024 h
        r = (1 / distance( [pi / 2, -pi / 2, pi / 2 - pi / 4, 0, pi / 2, - pi / 2], st))
    """,
    {
        'yarok': {
            'world': ManipulatorWorld,
            'behaviour': WanderBehaviour,
            'defaults': {
                'environment': 'sim',
                'components': {
                    '/finger_yellow': {
                        'label_color': '1.0 1.0 0.0'
                    },
                    '/finger_blue': {
                        'label_color': '0.0 0.0 1.0'
                    }
                }
            },
            'environments': {
                'sim': {
                    'platform': {
                        'class': PlatformMJC,
                        'mode': 'view'
                    },
                    'inspector': False,
                    'behaviour': {
                        'dataset_name': 'sim_depth'
                    }
                }
            },
            # 'callbacks': [
            #     lambda platform: e.emit('mjc_render', {'platform': platform})
            # ]
        }
    },
    lambda: [
        InteractivePlotter(),
        InteractiveLogger(),
        # MJCRenderer(),
        # EBoard(8081)
    ])
def main():
    experiments_datetimes = [
        '2022-06-14 00:39:27',
        '2022-06-14 00:39:28',
        '2022-06-14 00:39:31',
        '2022-06-14 00:39:32',
        '2022-06-14 00:39:35',
        '2022-06-14 00:39:38',
        '2022-06-14 00:39:40',
        '2022-06-14 00:40:03'
    ]
    files = [
        e.ws('outputs',
             e_datetime + ' - Testing DDPG with simple reaching position goal (data collection - day 2)',
             'out',
             str(i+1) + '_replay_buffer.npy')
        for i, e_datetime in enumerate(experiments_datetimes)
    ]
    buffers = ReplayBuffer.load_all(files)
    buff = list(itertools.chain(*[b.buffer for b in buffers]))
    b = ReplayBuffer(buff).save(e.out('0_replay_buffer'))

    yarok.run(e.yarok)


if __name__ == '__main__':
    experimenter.run(main, append=True)
    # experimenter.query()
