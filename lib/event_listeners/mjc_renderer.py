import cv2
import mujoco
import numpy as np
from experimenter import e


class MJCRenderer:
    """
        Class used to save the plots graphs during training
    """

    def __init__(self):
        self.rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        self.viewport = mujoco.MjrRect(0, 0, 640, 480)
        self.vopt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()

        self.platform = None
        self.ctx = None
        self.scn = None

        self.cam = mujoco.MjvCamera()
        self.cam.type = 0  # free camera

    def on_mjc_render(self, ev):
        if self.scn is None:
            self.platform = ev['platform']
            self.scn = mujoco.MjvScene(self.platform.model, maxgeom=10000)
            self.ctx = mujoco.MjrContext(self.platform.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        mujoco.mjv_updateScene(
            self.platform.model,
            self.platform.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scn)

        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjr_render(self.viewport, self.scn, self.ctx)
        mujoco.mjr_readPixels(self.rgb, None, self.viewport, self.ctx)

        self.rgb = cv2.flip(self.rgb, 0)
        self.rgb = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2RGB)

        cv2.imwrite(e.out('mjc_view.png'), self.rgb)