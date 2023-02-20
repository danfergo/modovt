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

        # gets frame from camera
        rgb = self.cam.read()

        # converts to grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        # inits & detects keypoints / markers
        params = cv2.SimpleBlobDetector_Params()
        # params.filterByConvexity = True
        # params.minConvexity  = 0.95
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        img_pts = np.array([kp.pt for kp in keypoints])

        assert len(img_pts) == 4

        # shows camera image with the found keypoints.
        kp_img = cv2.drawKeypoints(rgb.copy(), keypoints, np.array([]), (0, 0, 255),
                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('vis', kp_img)

        # the size of the workspace area in mm
        MARGIN = 30
        WS_WIDTH = 600
        WS_HEIGHT = 400

        # the corresponding 2D points on the working surface
        # this are sorted matching the corresponding keypoints extracted from
        # the camera view.
        ws_pts = np.array([
            (MARGIN, WS_HEIGHT + MARGIN),
            (WS_WIDTH + MARGIN, MARGIN),
            (MARGIN, MARGIN),
            (WS_WIDTH + MARGIN, WS_HEIGHT + MARGIN),
        ])

        # finds the homography. i.e. the transformation matrix
        # between points in the image and points in the working surface
        H, status = cv2.findHomography(img_pts, ws_pts)

        # generates & shows a top view of the working surface.
        top_view = cv2.warpPerspective(rgb, H, (WS_WIDTH + 2 * MARGIN, WS_HEIGHT + 2 * MARGIN))
        cv2.imshow('top-view', top_view)

        __location__ = os.path.dirname(os.path.abspath(__file__))
        np.save(os.path.join(__location__, 'h.npy'), H)
        cv2.imwrite(os.path.join(__location__, 'bkg.jpg'), top_view)
        print('Saved homography.')

        self.pl.wait_seconds(100)



if __name__ == '__main__':
    Platform.create(config(FindHBehaviour, include_rope=False)).run()
