import cv2
import numpy as np


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def estimate_pose(obj, img, K):
    success, rvec, tvec, _ = cv2.solvePnPRansac(obj, img, K, None)

    if not success:
        return None

    R, _ = cv2.Rodrigues(rvec)
    return make_T(R, tvec)