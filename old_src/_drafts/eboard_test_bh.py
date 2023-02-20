import random

import torch.nn
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
from lib.event_listeners.mjc_renderer import MJCRenderer
from old_src.agent.nn import minimal_discrete_actor, minimal_dynamics_model
from old_src.agent.rewards import TactilePainPleasureReward
from old_src.rl_algorithms.curiosity_reinforce import CuriosityReinforce
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

        self.target_q = [pi / 2, -pi / 2, pi / 2 - pi / 4, 0, pi / 2, - pi / 2]
        self.working_space = [
            [pi / 2 - pi / 4, pi / 2 + pi / 4],
            [-pi / 2 - pi / 3, -pi / 2 - pi / 4],  # shoulder
            [pi / 2 - pi / 4 - pi / 4, pi / 2 - pi / 4 + pi / 4],
            [0 - pi / 4, 0 + pi / 4],
            [pi / 2 - pi / 4, pi / 2 + pi / 4],
            [- pi / 2 - pi / 4, - pi / 2 + pi / 4],
        ]

        self.rl_learner = CuriosityReinforce(
            minimal_dynamics_model(7, 6),
            minimal_discrete_actor(6, 6)
        )

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
        print('WAK UP 1')
        self.gripper.move(0)
        print('WAK UP 2')
        self.arm.move_q(self.target_q)
        print('WAK UP 3')
        wait(lambda: self.arm.is_at(self.target_q) ) # and self.gripper.is_at(0))
        print('WAK UP 4')
        print('WAK UP')
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

            p = [pi / 2, -pi / 2, pi / 2 - pi / 4, 0, pi / 2, - pi / 2]
            st = self.arm.at()

            return 10 * (1 / distance(p, st))

        def take_action(ai):
            # print('action: ', ai)
            self.target_q = self.rnd_rel_discrete_q(ai)
            # self.target_q[1] = -pi / 2 - pi / 3
            self.arm.move_q(self.target_q)
            wait(lambda: self.arm.is_at(self.target_q))

        self.rl_learner.run(get_observation,
                            get_reward,
                            take_action)


@experiment(
    """
        Exploration with DDPG
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
            'callbacks': [
                lambda platform: e.emit('mjc_render', {'platform': platform})
            ]
        }
    },
    lambda: [
        InteractivePlotter(),
        InteractiveLogger(),
        # MJCRenderer(),
        EBoard()
    ])
def main(e):
    while True:
        cv2.waitKey(1)
    # wait()
    # yarok.run(e.yarok)


if __name__ == '__main__':
    experimenter.run(main,
                     append=True)
    experimenter.query()
