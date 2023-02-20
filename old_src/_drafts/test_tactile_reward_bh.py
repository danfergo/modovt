import yarok
from yarok import PlatformMJC, PlatformHW
from yarok.components_manager import component

from yarok.components.robotiq_2f85.robotiq_2f85 import robotiq_2f85
from yarok.components.ur5.ur5 import UR5

from yarok.components.geltip.geltip import GelTip

from old_src.agent.rewards import TactilePainPleasureReward
from old_src.utils.img import show

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
        self.tactileReward = TactilePainPleasureReward()

    def wake_up(self):
        while True:
            bkg = self.bkg.read()
            depth = self.finger.read()
            rel_depth = bkg - depth

            reward, in_contact_map, reward_map = self.tactileReward.reward(rel_depth, return_maps=True)

            print(reward)
            show('tactile_reward_map', reward_map)
            show('tactile_in_contact_area', in_contact_map)
            show('color_map', self.tactileReward.tactile_color_map(reward_map))
            yarok.wait(lambda: True)


if __name__ == '__main__':
    yarok.run({
        'world': GelTipWorld,
        'behaviour': TestBehaviour,
        'defaults': {
            'environment': 'sim',
            'components': {
                '/finger': {
                    'label_color': '1.0 1.0 0.0'
                },
                '/bkg': {
                    'label_color': '1.0 1.0 0.0'
                },
            }
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
        }
    })
