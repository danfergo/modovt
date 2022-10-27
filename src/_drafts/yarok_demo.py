import random

from yarok import wait, PlatformMJC

import yarok
from yarok.components_manager import component

from yarok.components.robotiq_2f85.robotiq_2f85 import robotiq_2f85
from yarok.components.ur5.ur5 import UR5

from yarok.components.geltip.geltip import GelTip
from yarok.components.anet_a30.anet_a30 import AnetA30
from math import pi


@component(
    components=[
        UR5,
        robotiq_2f85,
        GelTip,
        AnetA30
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
            </asset>


            <worldbody>
                <light directional="true" diffuse=".4 .4 .4" specular="0.1 0.1 0.1" pos="0 0 5.0" dir="0 0 -1"/>
                <light directional="true" diffuse=".6 .6 .6" specular="0.2 0.2 0.2" pos="0 0 4" dir="0 0 -1"/>
                <camera name="viewer" pos="-0. -2 0.3" mode="fixed" zaxis="0 -1 0"/>

                <body name="floor">
                    <geom name="ground" type="plane" size="0 0 1" pos="0 0 0" quat="1 0 0 0" material="matplane" condim="1"/>
                </body> 

               <!-- <body name="rope" pos="0.5 -0.1 -0.63" > 
                    <composite type="grid" count="40 1 1" spacing="0.01" offset="0 0 1">
                        <joint kind="main" damping="0.01"/>
                        <tendon kind="main" width="0.01" rgba=".8 .2 .1 1"/>
                        <geom size=".01" rgba=".8 .2 .1 1"/>
                        <pin coord="39"/> 
                    </composite>
                </body> -->
                <!--
                <body name='plate'>
                    <geom type="box" size="0.1 0.1 0.05" pos="0.75 -0.05 0"                           contype="1" 
                          conaffinity="1"/>
                </body> -->

                <body  pos="-0.15 1.5 0" >
                    <anet_a30 name="printer"></anet_a30>
                </body>
    
                <body  pos="1 1 0" >
                  <ur5 name="arm_1">
                       <robotiq_2f85 name="gripper" parent="ee_link">
                            <body pos="0 0.02 0.0395" parent="right_tip">
                                <geltip name="finger_yellow" parent="left_tip"/>
                            </body>
                            <body pos="0 0.02 0.0395" parent="left_tip">
                                <geltip name="finger_blue" parent="right_tip"/>
                            </body>
                        </robotiq_2f85>
                    </ur5> 
                </body>

                <body  pos="-0.5 1 0" >
                  <ur5 name="arm">
                       <robotiq_2f85 name="gripper2" parent="ee_link">
                            <body pos="0 0.02 0.0395" parent="right_tip">
                                <geltip name="finger_yellow" parent="left_tip"/>
                            </body>
                            <body pos="0 0.02 0.0395" parent="left_tip">
                                <geltip name="finger_blue" parent="right_tip"/>
                            </body>
                        </robotiq_2f85>
                    </ur5> 
                </body>
            </worldbody>        
        </mujoco>
    """
)
class DemoWorld:

    def __init__(self):
        pass


class DemoBehaviour:

    def __init__(self,
                 arm: UR5,
                 arm_1: UR5,
                 gripper2: robotiq_2f85
                 ):
        self.arm = arm
        self.arm_1 = arm_1
        self.gripper2 = gripper2

    def on_start(self):
        self.initial_q = [
            pi / 2 + pi / 4 - pi / 3,
            -pi / 2 - pi / 4,
            pi / 2 - pi / 4,
            0,
            pi / 2,
            - pi / 2
        ]

        self.initial_q_1 = [
            -pi / 2 - pi / 4,
            -pi / 2 - pi / 4 + pi / 3,
            pi / 2 - pi / 4,
            0,
            pi / 2,
            - pi / 2
        ]

        self.arm.move_q(self.initial_q)
        self.arm_1.move_q(self.initial_q_1)
        self.gripper2.move(0.5)

    def on_update(self):
        return True


yarok.run({
    'world': DemoWorld,
    'behaviour': DemoBehaviour,
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
})
