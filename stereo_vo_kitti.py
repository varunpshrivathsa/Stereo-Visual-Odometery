import os
import glob
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Utility functions
# ============================================================

def load_image_paths(sequence_path):
    left_dir = os.path.join(sequence_path, "image_0")
    right_dir = os.path.join(sequence_path, "image_1")

    left_imgs = sorted(glob.glob(os.path.join(left_dir, "*.png")))
    right_imgs = sorted(glob.glob(os.path.join(right_dir, "*.png")))

    # KITTI can also be image_0/data/*.png
    if len(left_imgs) == 0:
        left_imgs = sorted(glob.glob(os.path.join(left_dir, "data", "*.png")))
    if len(right_imgs) == 0:
        right_imgs = sorted(glob.glob(os.path.join(right_dir, "data", "*.png")))

    if len(left_imgs) == 0 or len(right_imgs) == 0:
        raise FileNotFoundError(
            f"Could not find images.\n"
            f"Tried:\n"
            f"  {os.path.join(left_dir, '*.png')}\n"
            f"  {os.path.join(left_dir, 'data', '*.png')}\n"
            f"  {os.path.join(right_dir, '*.png')}\n"
            f"  {os.path.join(right_dir, 'data', '*.png')}"
        )

    if len(left_imgs) != len(right_imgs):
        raise ValueError("Left and right image counts do not match")

    return left_imgs, right_imgs


def read_calib_kitti(calib_path):
    data = {}
    with open(calib_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = line.strip().split(":", 1)
            data[key] = np.array([float(x) for x in val.strip().split()])

    P0 = data["P0"].reshape(3, 4)
    P1 = data["P1"].reshape(3, 4)

    K = P0[:, :3]
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    baseline = abs(P1[0, 3] / P1[0, 0] - P0[0, 3] / P0[0, 0])
    return K, fx, fy, cx, cy, baseline


def load_ground_truth(gt_path):
    poses = []
    with open(gt_path, "r") as f:
        for line in f:
            vals = np.fromstring(line.strip(), sep=" ")
            if vals.size != 12:
                continue
            T = np.eye(4)
            T[:3, :4] = vals.reshape(3, 4)
            poses.append(T)
    return poses


def detect_and_compute(img, detector):
    kp, des = detector.detectAndCompute(img, None)
    return kp, des


# ============================================================
# Stereo VO helper functions
# ============================================================

def stereo_3d_from_features(
    kp_left, des_left,
    kp_right, des_right,
    fx, fy, cx, cy, baseline,
    ratio_thresh=0.75,
    epi_thresh=2.0,
    min_disparity=1.0,
    max_depth=80.0
):
    points3d_dict = {}

    if des_left is None or des_right is None:
        return points3d_dict

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des_left, des_right, k=2)

    for pair in knn:
        if len(pair) < 2:
            continue

        m, n = pair
        if m.distance >= ratio_thresh * n.distance:
            continue

        left_idx = m.queryIdx
        right_idx = m.trainIdx

        uL, vL = kp_left[left_idx].pt
        uR, vR = kp_right[right_idx].pt

        # Rectified stereo: vertical coordinates should match
        if abs(vL - vR) > epi_thresh:
            continue

        disparity = uL - uR
        if disparity <= min_disparity:
            continue

        Z = (fx * baseline) / disparity
        if Z <= 0 or Z > max_depth:
            continue

        X = (uL - cx) * Z / fx
        Y = (vL - cy) * Z / fy

        points3d_dict[left_idx] = np.array([X, Y, Z], dtype=np.float32)

    return points3d_dict


def temporal_matches_left_to_left(
    kp_left_t, des_left_t,
    kp_left_t1, des_left_t1,
    ratio_thresh=0.75
):
    temporal_dict = {}

    if des_left_t is None or des_left_t1 is None:
        return temporal_dict

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des_left_t, des_left_t1, k=2)

    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            temporal_dict[m.queryIdx] = m.trainIdx

    return temporal_dict


def build_3d2d_correspondences(points3d_dict, temporal_dict, kp_left_t1):
    obj_points = []
    img_points = []
    left_t1_indices = []

    for idx_t, point3d in points3d_dict.items():
        if idx_t not in temporal_dict:
            continue

        idx_t1 = temporal_dict[idx_t]
        u1, v1 = kp_left_t1[idx_t1].pt

        obj_points.append(point3d)
        img_points.append([u1, v1])
        left_t1_indices.append(idx_t1)

    if len(obj_points) == 0:
        return None, None, None

    obj_points = np.asarray(obj_points, dtype=np.float32).reshape(-1, 3)
    img_points = np.asarray(img_points, dtype=np.float32).reshape(-1, 2)
    left_t1_indices = np.asarray(left_t1_indices, dtype=np.int32)

    return obj_points, img_points, left_t1_indices


def make_transform(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def draw_inlier_points(image_gray, img_points, inliers, radius=2):
    vis = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    if inliers is not None and len(inliers) > 0:
        for idx in inliers.flatten():
            u, v = img_points[idx]
            cv2.circle(vis, (int(u), int(v)), radius, (0, 255, 0), -1)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# ============================================================
# Live visualization main
# ============================================================

def run_stereo_vo_live(
    sequence_path,
    gt_path,
    sequence_name="07",
    max_frames=None,
    step=1,
    playback_delay=0.001
):
    left_imgs, right_imgs = load_image_paths(sequence_path)
    K, fx, fy, cx, cy, baseline = read_calib_kitti(os.path.join(sequence_path, "calib.txt"))
    gt_poses = load_ground_truth(gt_path)

    n = min(len(left_imgs), len(right_imgs), len(gt_poses))
    if max_frames is not None:
        n = min(n, max_frames)

    detector = cv2.ORB_create(nfeatures=3000)

    pose = np.eye(4, dtype=np.float64)
    est_traj = [pose[:3, 3].copy()]

    # full GT trajectory plotted from start
    gt_all = np.asarray([p[:3, 3].copy() for p in gt_poses[:n]])
    # -------------------------
    # matplotlib live window
    # -------------------------
    plt.ion()
    fig, (ax_img, ax_traj) = plt.subplots(1, 2, figsize=(16, 7))

    # Initial blank image
    sample_img = cv2.imread(left_imgs[0], cv2.IMREAD_GRAYSCALE)
    if sample_img is None:
        raise RuntimeError("Could not read first image for initialization")

    blank_rgb = cv2.cvtColor(sample_img, cv2.COLOR_GRAY2RGB)
    img_artist = ax_img.imshow(blank_rgb)
    ax_img.set_title("Current Left Image + Inlier Tracks")
    ax_img.axis("off")

    gt_line, = ax_traj.plot(gt_all[:, 0], gt_all[:, 2], label="Ground Truth", linewidth=2)
    est_line, = ax_traj.plot([], [], label="Estimated", linewidth=2)
    current_gt_pt, = ax_traj.plot([gt_all[0, 0]], [gt_all[0, 2]], "o", markersize=4)
    current_est_pt, = ax_traj.plot([], [], "o", markersize=4)

    ax_traj.set_title(f"Stereo VO Live Trajectory - KITTI {sequence_name}")
    ax_traj.set_xlabel("X")
    ax_traj.set_ylabel("Z")
    ax_traj.grid(True)
    ax_traj.legend()
    ax_traj.axis("equal")

    plt.tight_layout()

    print(f"Total frames used: {n}")
    print(f"Baseline: {baseline:.6f}")
    print("Press Ctrl+C in terminal to stop.")

    try:
        for i in range(0, n - step, step):
            left_t = cv2.imread(left_imgs[i], cv2.IMREAD_GRAYSCALE)
            right_t = cv2.imread(right_imgs[i], cv2.IMREAD_GRAYSCALE)
            left_t1 = cv2.imread(left_imgs[i + step], cv2.IMREAD_GRAYSCALE)

            if left_t is None or right_t is None or left_t1 is None:
                est_traj.append(pose[:3, 3].copy())
                continue

            kp_left_t, des_left_t = detect_and_compute(left_t, detector)
            kp_right_t, des_right_t = detect_and_compute(right_t, detector)
            kp_left_t1, des_left_t1 = detect_and_compute(left_t1, detector)

            used_img = cv2.cvtColor(left_t1, cv2.COLOR_GRAY2RGB)

            if des_left_t is None or des_right_t is None or des_left_t1 is None:
                est_traj.append(pose[:3, 3].copy())
            else:
                points3d_dict = stereo_3d_from_features(
                    kp_left_t, des_left_t,
                    kp_right_t, des_right_t,
                    fx, fy, cx, cy, baseline,
                    ratio_thresh=0.75,
                    epi_thresh=2.0,
                    min_disparity=1.0,
                    max_depth=80.0
                )

                temporal_dict = temporal_matches_left_to_left(
                    kp_left_t, des_left_t,
                    kp_left_t1, des_left_t1,
                    ratio_thresh=0.75
                )

                obj_points, img_points, _ = build_3d2d_correspondences(
                    points3d_dict, temporal_dict, kp_left_t1
                )

                if obj_points is not None and len(obj_points) >= 6:
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        objectPoints=obj_points,
                        imagePoints=img_points,
                        cameraMatrix=K,
                        distCoeffs=None,
                        iterationsCount=200,
                        reprojectionError=2.0,
                        confidence=0.999,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    if success and inliers is not None and len(inliers) >= 6:
                        R, _ = cv2.Rodrigues(rvec)
                        t = tvec.reshape(3, 1)
                        T_t_to_t1 = make_transform(R, t)

                        # accumulate inverse motion to get camera global pose
                        pose = pose @ np.linalg.inv(T_t_to_t1)

                        used_img = draw_inlier_points(left_t1, img_points, inliers, radius=2)

                        inlier_ratio = len(inliers) / len(obj_points)
                        print(
                            f"[Frame {i:04d}] "
                            f"stereo_3d={len(points3d_dict):4d}, "
                            f"3D-2D={len(obj_points):4d}, "
                            f"inliers={len(inliers):4d}, "
                            f"inlier_ratio={inlier_ratio:.3f}, "
                            f"|t|={np.linalg.norm(t):.3f}"
                        )

                est_traj.append(pose[:3, 3].copy())

            # -------------------------
            # update live image
            # -------------------------
            img_artist.set_data(used_img)
            ax_img.set_title(f"Left Image @ frame {i + step}")

            # -------------------------
            # update live trajectory
            # -------------------------
            est_arr = np.asarray(est_traj)

            # GT already fully drawn, only estimated grows
            est_line.set_data(est_arr[:, 0], est_arr[:, 2])

            # move current markers
            current_gt_pt.set_data([gt_all[i + step, 0]], [gt_all[i + step, 2]])
            current_est_pt.set_data([est_arr[-1, 0]], [est_arr[-1, 2]])

            # auto-rescale using full GT only
            x_min, x_max = gt_all[:, 0].min(), gt_all[:, 0].max()
            z_min, z_max = gt_all[:, 2].min(), gt_all[:, 2].max()

            margin_x = 0.1 * (x_max - x_min)
            margin_z = 0.1 * (z_max - z_min)

            ax_traj.set_xlim(x_min - margin_x, x_max + margin_x)
            ax_traj.set_ylim(z_min - margin_z, z_max + margin_z)

            fig.canvas.draw_idle()
            plt.pause(playback_delay)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    plt.ioff()
    plt.show()


# ============================================================
# Entry
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence",
        type=str,
        default="07",
        help="KITTI sequence id, e.g. 00, 01, 02, ..., 10"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/data/datasets/kitti/sequences",
        help="Root folder containing KITTI sequence folders"
    )
    parser.add_argument(
        "--gt_root",
        type=str,
        default="/data/datasets/kitti/groundtruth/poses",
        help="Root folder containing KITTI ground truth pose txt files"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Frame step size"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.001,
        help="Pause delay per frame for live display"
    )

    args = parser.parse_args()

    sequence_path = os.path.join(args.dataset_root, args.sequence)
    gt_path = os.path.join(args.gt_root, f"{args.sequence}.txt")

    run_stereo_vo_live(
        sequence_path=sequence_path,
        gt_path=gt_path,
        sequence_name=args.sequence,
        max_frames=args.max_frames,
        step=args.step,
        playback_delay=args.delay
    )

if __name__ == "__main__":
    main()