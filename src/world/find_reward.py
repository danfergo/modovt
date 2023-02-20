import cv2
import os

import numpy as np
from yarok import Platform
from yarok.comm.components.cam.cam import Cam

from src.world.rope_world import config


class FindHBehaviour:

    def __init__(self, vis_cam: Cam, pl: Platform):
        self.cam = vis_cam
        self.pl = pl

    def on_start(self):
        pass

    def on_update(self):
        self.pl.wait_seconds(5)

        # Gets frame from camera
        rgb = self.cam.read()

        __location__ = os.path.dirname(os.path.abspath(__file__))
        H = np.load(os.path.join(__location__, 'h.npy'))
        bkg = cv2.imread(os.path.join(__location__, 'bkg.jpg'))

        # The size of the workspace area in mm
        MARGIN = 30
        WS_WIDTH = 600
        WS_HEIGHT = 400

        ws_pts = np.array([
            (MARGIN, WS_HEIGHT + MARGIN),
            (WS_WIDTH + MARGIN, MARGIN),
            (MARGIN, MARGIN),
            (WS_WIDTH + MARGIN, WS_HEIGHT + MARGIN),
        ])

        top_view = cv2.warpPerspective(rgb, H, (WS_WIDTH + 2 * MARGIN, WS_HEIGHT + 2 * MARGIN))

        gray_bkg = cv2.cvtColor(bkg, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.cvtColor(top_view, cv2.COLOR_BGR2GRAY)

        diff = np.abs(gray_bkg - gray_current)
        diff_cleaned = cv2.erode(diff, np.ones((6, 6), np.uint8))
        _, diff_bin = cv2.threshold(diff_cleaned, 10, 255, cv2.THRESH_BINARY)  # uint8, 0-255

        st = cv2.resize(diff_bin, (12, 8))

        cv2.imshow('diff', diff_bin)
        cv2.imshow('st', st)


if __name__ == '__main__':
    Platform.create(config(FindHBehaviour)).run()
