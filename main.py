import os
import argparse
from src.vo import run_vo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="07")
    parser.add_argument("--dataset_root", type=str, default="/data/datasets/kitti/sequences")
    parser.add_argument("--gt_root", type=str, default="/data/datasets/kitti/groundtruth/poses")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.001)

    args = parser.parse_args()

    sequence_path = os.path.join(args.dataset_root, args.sequence)
    gt_path = os.path.join(args.gt_root, f"{args.sequence}.txt")

    run_vo(
        sequence_path=sequence_path,
        gt_path=gt_path,
        max_frames=args.max_frames,
        sequence_name=args.sequence,
        step=args.step,
        playback_delay=args.delay,
    )


if __name__ == "__main__":
    main()