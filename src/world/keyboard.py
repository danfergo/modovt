from time import time

from yarok import Platform, PlatformMJC, PlatformHW, component, ConfigBlock, Injector
import curses

import yarok.comm.plugins.cv2_waitkey as cv2wk


class KeyboardInterface:

    def __init__(self, interface):
        self.last_update = time()
        self.pressed_key = None

    def key(self):
        return self.pressed_key

    def step(self):
        t = time()
        key = cv2wk.__LAST_KEY__

        if key is not None:
            self.last_update = time()
            # if key in range(32, 127)
            self.pressed_key = chr(key)
        elif self.last_update > t - 1.0:
            self.pressed_key = None


@component(
    tag='keyboard',
    defaults={
        'interface_mjc': KeyboardInterface
    },
    # language=xml
    template="""
        <mujoco>
            <default>
            </default>
            <worldbody>
            </worldbody>        
        </mujoco>
    """
)
class Keyboard:
    pass

    def key(self):
        pass
