import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .data import load_image_paths, read_image, read_calib, load_gt
from .features import create_orb, detect, match_knn
from .stereo import compute_3d, temporal_match, build_correspondences


def draw_inlier_points(image_gray, img_points, inliers, radius=2):
    vis = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

    if inliers is not None and len(inliers) > 0:
        for idx in inliers.flatten():
            u, v = img_points[idx]
            cv2.circle(vis, (int(u), int(v)), radius, (0, 255, 0), -1)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def draw_depth_points_on_right_image(
    right_gray,
    kp_right,
    stereo_matches,
    pts3d,
    depth_norm,
    cmap_name="turbo",
    radius=2,
):
    vis = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2RGB)
    cmap = plt.get_cmap(cmap_name)

    for m in stereo_matches:
        left_idx = m.queryIdx
        right_idx = m.trainIdx

        if left_idx not in pts3d:
            continue

        z = float(pts3d[left_idx][2])
        u, v = kp_right[right_idx].pt

        norm = float(np.clip(depth_norm(z), 0.0, 1.0))
        rgba = cmap(norm)
        color = tuple(int(255 * c) for c in rgba[:3])

        cv2.circle(vis, (int(u), int(v)), radius, color, -1)

    return vis


def stack_left_right(left_rgb, right_rgb):
    return np.hstack((left_rgb, right_rgb))


def run_vo(
    sequence_path,
    gt_path,
    max_frames=None,
    sequence_name="07",
    step=1,
    playback_delay=0.001,
):
    left_paths, right_paths = load_image_paths(sequence_path)

    calib_path = os.path.join(sequence_path, "calib.txt")
    K, fx, fy, cx, cy, baseline = read_calib(calib_path)

    gt_poses = load_gt(gt_path)

    n = min(len(left_paths), len(right_paths), len(gt_poses))
    if max_frames is not None:
        n = min(n, max_frames)

    if n < 2:
        raise RuntimeError("Not enough frames to run stereo VO")

    orb = create_orb()

    pose = np.eye(4, dtype=np.float64)
    est_traj = [pose[:3, 3].copy()]
    gt_all = np.asarray([p[:3, 3].copy() for p in gt_poses[:n]])

    depth_norm = Normalize(vmin=3.0, vmax=40.0)
    cmap = plt.get_cmap("turbo")
    sm = ScalarMappable(norm=depth_norm, cmap=cmap)
    sm.set_array([])

    plt.ion()
    plt.style.use("dark_background")

    fig = plt.figure(figsize=(18, 10), facecolor="black")
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.6],
        hspace=0.18,
    )

    ax_img = fig.add_subplot(gs[0, 0], facecolor="black")
    ax_traj = fig.add_subplot(gs[1, 0], facecolor="black")

    sample_left = read_image(left_paths[0])
    sample_right = read_image(right_paths[0])

    if sample_left is None or sample_right is None:
        raise RuntimeError("Could not read first stereo pair for visualization")

    sample_left_rgb = cv2.cvtColor(sample_left, cv2.COLOR_GRAY2RGB)
    sample_right_rgb = cv2.cvtColor(sample_right, cv2.COLOR_GRAY2RGB)
    combined0 = stack_left_right(sample_left_rgb, sample_right_rgb)

    img_artist = ax_img.imshow(combined0, aspect="auto")
    ax_img.set_title(
        "Left Image - PnP  |  Right Image - Depth Points",
        color="white",
    )
    ax_img.axis("off")

    cbar = fig.colorbar(sm, ax=ax_img, fraction=0.025, pad=0.012)
    cbar.set_label("Depth (Z)", fontsize=10, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    cbar.outline.set_edgecolor("white")
    cbar.ax.set_facecolor("black")
    cbar.ax.invert_yaxis()

    # 🔥 COLORS UPDATED HERE
    gt_line, = ax_traj.plot(
        gt_all[:, 0],
        gt_all[:, 2],
        color="#66ccff",   # light blue
        label="Ground Truth",
        linewidth=2,
    )
    est_line, = ax_traj.plot(
        [],
        [],
        color="red",
        label="Estimated",
        linewidth=2,
    )
    current_gt_pt, = ax_traj.plot(
        [gt_all[0, 0]],
        [gt_all[0, 2]],
        "o",
        color="#66ccff",
        markersize=6,
        label="Current GT",
    )
    current_est_pt, = ax_traj.plot(
        [],
        [],
        "o",
        color="red",
        markersize=6,
        label="Current Estimate",
    )

    ax_traj.set_title(
        f"Stereo VO Live Trajectory - KITTI {sequence_name}",
        color="white",
    )
    ax_traj.set_xlabel("X", color="white")
    ax_traj.set_ylabel("Z", color="white")

    ax_traj.tick_params(axis="both", colors="white")
    ax_traj.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)

    for spine in ax_traj.spines.values():
        spine.set_color("white")

    legend = ax_traj.legend(loc="upper right", facecolor="black", edgecolor="white")
    for text in legend.get_texts():
        text.set_color("white")

    fig.subplots_adjust(left=0.04, right=0.96, top=0.95, bottom=0.06)

    print(f"Sequence: {sequence_name}")
    print(f"Frames used: {n}")
    print(f"Baseline: {baseline:.6f}")
    print("Press Ctrl+C in terminal to stop.")

    try:
        for i in range(0, n - step, step):
            left_t = read_image(left_paths[i])
            right_t = read_image(right_paths[i])
            left_t1 = read_image(left_paths[i + step])

            if left_t is None or right_t is None or left_t1 is None:
                est_traj.append(pose[:3, 3].copy())
                continue

            kp_left_t, des_left_t = detect(left_t, orb)
            kp_right_t, des_right_t = detect(right_t, orb)
            kp_left_t1, des_left_t1 = detect(left_t1, orb)

            left_vis = cv2.cvtColor(left_t1, cv2.COLOR_GRAY2RGB)
            right_vis = cv2.cvtColor(right_t, cv2.COLOR_GRAY2RGB)
            used_img = stack_left_right(left_vis, right_vis)

            if des_left_t is None or des_right_t is None or des_left_t1 is None:
                est_traj.append(pose[:3, 3].copy())
            else:
                stereo_matches = match_knn(des_left_t, des_right_t, ratio=0.75)
                temporal_matches = match_knn(des_left_t, des_left_t1, ratio=0.75)

                pts3d = compute_3d(
                    kp_left_t,
                    kp_right_t,
                    stereo_matches,
                    fx,
                    fy,
                    cx,
                    cy,
                    baseline,
                )

                right_vis = draw_depth_points_on_right_image(
                    right_gray=right_t,
                    kp_right=kp_right_t,
                    stereo_matches=stereo_matches,
                    pts3d=pts3d,
                    depth_norm=depth_norm,
                    cmap_name="turbo",
                    radius=2,
                )

                temp_map = temporal_match(temporal_matches, kp_left_t, kp_left_t1)
                obj_points, img_points = build_correspondences(pts3d, temp_map, kp_left_t1)

                if obj_points is not None and len(obj_points) >= 6:
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        objectPoints=obj_points,
                        imagePoints=img_points,
                        cameraMatrix=K,
                        distCoeffs=None,
                        iterationsCount=200,
                        reprojectionError=2.0,
                        confidence=0.999,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )

                    if success and inliers is not None and len(inliers) >= 6:
                        R, _ = cv2.Rodrigues(rvec)

                        T = np.eye(4, dtype=np.float64)
                        T[:3, :3] = R
                        T[:3, 3] = tvec.reshape(3)

                        pose = pose @ np.linalg.inv(T)
                        left_vis = draw_inlier_points(left_t1, img_points, inliers, radius=2)
                    else:
                        left_vis = cv2.cvtColor(left_t1, cv2.COLOR_GRAY2RGB)
                else:
                    left_vis = cv2.cvtColor(left_t1, cv2.COLOR_GRAY2RGB)

                used_img = stack_left_right(left_vis, right_vis)
                est_traj.append(pose[:3, 3].copy())

            img_artist.set_data(used_img)

            est_np = np.asarray(est_traj)
            gt_idx = min(i + step, len(gt_all) - 1)

            est_line.set_data(est_np[:, 0], est_np[:, 2])
            current_gt_pt.set_data([gt_all[gt_idx, 0]], [gt_all[gt_idx, 2]])
            current_est_pt.set_data([est_np[-1, 0]], [est_np[-1, 2]])

            ax_traj.relim()
            ax_traj.autoscale_view()

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(playback_delay)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    plt.ioff()
    plt.show()