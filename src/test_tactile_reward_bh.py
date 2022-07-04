import math

import cv2
import yarok
from yarok import PlatformMJC, PlatformHW
from yarok.components_manager import component

from yarok.components.robotiq_2f85.robotiq_2f85 import robotiq_2f85
from yarok.components.ur5.ur5 import UR5

from yarok.components.geltip.geltip import GelTip

from src.wander_bh import normalize
import numpy as np


def color_map(size, color):
    image = np.zeros((size[0], size[1], 3), np.uint8)
    image[:] = color
    return image


"""
    THIS WORLD AND BEHAVIOUR IS USED TO TEST DIFFERENT IMPLEMENTATIONS
    FOR THE TACTILE REWARD FUNCTION
"""


@component(
    components=[
        UR5,
        robotiq_2f85,
        GelTip
    ],
    # language=xml
    template="""
        <mujoco>
            <!-- possibly the physics solver -->
            <option timestep="0.01" solver="Newton" iterations="30" tolerance="1e-10" jacobian="auto" cone="pyramidal"/>
            <compiler angle="radian"/>

            <visual>
                <!-- important for the Geltips, to ensure the its camera frustum captures the close-up elastomer -->
                <map znear="0.001" zfar="50"/>
                <!--<quality shadowsize="2048"/> -->
            </visual>

            <asset>
                <!-- empty world -->
                <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2"
                         width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>    
                <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
                
                <material name="red" rgba="1 0 0 1"/>
                <material name="green" rgba="0 1 0 1"/>

            </asset>


            <worldbody>
                <light directional="true" diffuse=".4 .4 .4" specular="0.1 0.1 0.1" pos="0 0 5.0" dir="0 0 -1"/>
                <light directional="true" diffuse=".6 .6 .6" specular="0.2 0.2 0.2" pos="0 0 4" dir="0 0 -1"/>
                <camera name="viewer" pos="-0. -2 0.3" mode="fixed" zaxis="0 -1 0"/>

                <body name="floor">
                    <geom name="ground" type="plane" size="0 0 1" pos="0 0 0" quat="1 0 0 0" material="matplane" condim="1"/>
                </body>
                
                <body pos="0.1 0 0">
                    <geltip name="bkg"/>
                </body>

                <geltip name="finger"/>
                
                <body name='contacts'>
                    <geom type="sphere" material="red" size="0.01" pos="0.01 0.01 0.05"/>
                    <geom type="sphere" material="green" size="0.01" pos="-0.013 -0.013 0.05"/>
                </body>
            </worldbody>        
        </mujoco>
    """
)
class GelTipWorld:
    pass


class TestBehaviour:

    def __init__(self, finger: GelTip, bkg: GelTip):
        self.finger = finger
        self.bkg = bkg

    def in_contact_area(self, rel_depth):
        area = np.array(rel_depth)
        area[area < 1e-05] = 0
        area[area >= 1e-05] = 1
        return area

    def tactile_pixel_reward(self, rel_depth):
        pain_min = 0.005
        pain_range = 0.005
        max_v = (pain_min / 2)
        pain_coeff = 50
        pleasure_coff = 5
        return (-1 * pain_coeff * min(1.0, (rel_depth - pain_min) / pain_range)) \
            if rel_depth > pain_min else (pleasure_coff * math.cos(((rel_depth - max_v) / max_v) * (math.pi / 2)))

    def reward(self, rel_depth, return_maps=False):
        in_contact_map = self.in_contact_area(rel_depth)
        reward_map = np.vectorize(self.tactile_pixel_reward)(rel_depth)

        area_size = np.sum(in_contact_map)
        r = 0 if area_size == 0 else np.sum(reward_map) / area_size

        if return_maps:
            return r, in_contact_map, reward_map
        return r

    def tactile_color_map(self, tactile_reward_map):
        pain_color = (0, 0, 255)
        pleasure_color = (0, 255, 0)
        map_shape = tactile_reward_map.shape

        pain_map = np.array(tactile_reward_map)
        pain_map[tactile_reward_map >= 0] = 0.0
        pain_map *= -1
        pain_map3 = np.stack([pain_map, pain_map, pain_map], axis=2)

        pleasure_map = np.array(tactile_reward_map)
        pleasure_map[tactile_reward_map < 0] = 0.0
        pleasure_map3 = np.stack([pleasure_map, pleasure_map, pleasure_map], axis=2)

        return np.multiply(pain_map3, color_map(map_shape, pain_color)) + \
               np.multiply(pleasure_map3, color_map(map_shape, pleasure_color))

    def wake_up(self):
        while True:
            bkg = self.bkg.read()
            depth = self.finger.read()
            rel_depth = bkg - depth

            reward, in_contact_map, reward_map = self.reward(rel_depth, return_maps=True)

            print(reward)
            cv2.imshow('tactile_reward_map', normalize(reward_map))
            cv2.imshow('tactile_in_contact_area', normalize(in_contact_map))
            cv2.imshow('color_map', normalize(self.tactile_color_map(reward_map)))
            yarok.wait(lambda: True)


if __name__ == '__main__':
    yarok.run({
        'world': GelTipWorld,
        'behaviour': TestBehaviour,
        'defaults': {
            'environment': 'sim',
        },
        'environments': {
            'sim': {
                'platform': {
                    'class': PlatformMJC,
                    'mode': 'view'
                },
                'inspector': False,
            },
            'real': {
                'platform': PlatformHW
            }
        },
    })
