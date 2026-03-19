import numpy as np


def compute_3d(kpL, kpR, matches, fx, fy, cx, cy, baseline):
    pts3d = {}

    for m in matches:
        iL = m.queryIdx
        iR = m.trainIdx

        uL, vL = kpL[iL].pt
        uR, vR = kpR[iR].pt

        if abs(vL - vR) > 2:
            continue

        disp = uL - uR
        if disp <= 1:
            continue

        Z = fx * baseline / disp
        X = (uL - cx) * Z / fx
        Y = (vL - cy) * Z / fy

        pts3d[iL] = np.array([X, Y, Z])

    return pts3d


def temporal_match(matches, kp1, kp2):
    mapping = {}
    for m in matches:
        mapping[m.queryIdx] = m.trainIdx
    return mapping


def build_correspondences(pts3d, temporal_map, kp_next):
    obj, img = [], []

    for idx, p3d in pts3d.items():
        if idx not in temporal_map:
            continue

        idx2 = temporal_map[idx]
        u, v = kp_next[idx2].pt

        obj.append(p3d)
        img.append([u, v])

    if len(obj) < 6:
        return None, None

    return np.array(obj, dtype=np.float32), np.array(img, dtype=np.float32)