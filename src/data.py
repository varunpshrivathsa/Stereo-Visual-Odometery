import os
import glob
import cv2
import numpy as np


def load_image_paths(sequence_path):
    left = sorted(glob.glob(os.path.join(sequence_path, "image_0/*.png")))
    right = sorted(glob.glob(os.path.join(sequence_path, "image_1/*.png")))

    if len(left) == 0:
        left = sorted(glob.glob(os.path.join(sequence_path, "image_0/data/*.png")))
    if len(right) == 0:
        right = sorted(glob.glob(os.path.join(sequence_path, "image_1/data/*.png")))

    return left, right


def read_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def read_calib(calib_path):
    data = {}
    with open(calib_path) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":")
                data[k] = np.array([float(x) for x in v.split()])

    P0 = data["P0"].reshape(3, 4)
    P1 = data["P1"].reshape(3, 4)

    K = P0[:, :3]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    baseline = abs(P1[0, 3] / P1[0, 0] - P0[0, 3] / P0[0, 0])

    return K, fx, fy, cx, cy, baseline


def load_gt(gt_path):
    poses = []
    with open(gt_path) as f:
        for line in f:
            vals = np.fromstring(line, sep=" ")
            T = np.eye(4)
            T[:3, :4] = vals.reshape(3, 4)
            poses.append(T)
    return poses