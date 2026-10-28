"""
Runs TiledPlayerDetector over a full video clip at an adjustable frame stride,
producing bbox time-series data for downstream analysis.

Needs pyarrow for Parquet output:  pip install pyarrow

Output (two tables, so frame sampling and per-box detections don't get mixed):
  <out_prefix>_frames.parquet      frame_idx, timestamp_s      -- one row per sampled frame
  <out_prefix>_detections.parquet  frame_idx, x, y, w, h, confidence  -- one row per detected box
  <out_prefix>_meta.json           video path, fps, frame_stride, detector params -- for reproducibility

Load together for analysis with:
  frames = pd.read_parquet("..._frames.parquet")
  dets = pd.read_parquet("..._detections.parquet")
  df = dets.merge(frames, on="frame_idx")

Parallelism: frame decoding runs on a background thread (cv2 releases the GIL
during decode) so it overlaps with GPU inference in the main thread. Detection
itself is batched across multiple sampled frames per model.predict() call
(TiledPlayerDetector.detect_batch) -- that's what actually uses a GPU's
parallelism; one CUDA context per worker process would just make workers fight
over the same GPU instead of speeding things up.
"""

import argparse
import json
import os
import queue
import threading

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from yolo_detector import TiledPlayerDetector, make_shadow_checker
from player_detection_pipeline import (
    ForegroundDetector, PerspectiveSizeFilter, load_removal_mask, resize_to_match,
)


def sampled_frames(video_path, frame_stride):
    """
    Yields (frame_idx, frame_bgr) for every `frame_stride`-th frame.

    Uses grab()+retrieve() instead of seeking via CAP_PROP_POS_FRAMES: many
    codecs only decode correctly from keyframes, so seeking per-frame on a
    compressed video is both slow and can land on the wrong frame. Sequential
    grab() (cheap, no full decode) skips frames; retrieve() (full decode) only
    runs on the ones we actually keep.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    try:
        frame_idx = 0
        while True:
            if not cap.grab():
                break
            if frame_idx % frame_stride == 0:
                ret, frame = cap.retrieve()
                if not ret:
                    break
                yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()


def _producer(video_path, frame_stride, out_queue):
    try:
        for frame_idx, frame in sampled_frames(video_path, frame_stride):
            out_queue.put((frame_idx, frame))
    finally:
        out_queue.put(None)  # sentinel: no more frames


def process_video(video_path, detector, frame_stride=5, frame_batch_size=8,
                   valid_mask=None, shadow_checker=None, bg_bgr=None, size_filter=None,
                   queue_size=32, show_progress=True):
    """
    Returns (frame_rows, detection_rows, fps) where:
      frame_rows      = [(frame_idx, timestamp_s), ...]
      detection_rows  = [(frame_idx, x, y, w, h, confidence), ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # CAP_PROP_FRAME_COUNT is unreliable for some containers/codecs (can read
    # 0 or negative) -- fall back to an unbounded progress bar rather than a
    # wrong percentage/ETA in that case
    expected_samples = (total_frames + frame_stride - 1) // frame_stride if total_frames > 0 else None

    frame_q = queue.Queue(maxsize=queue_size)
    producer = threading.Thread(target=_producer, args=(video_path, frame_stride, frame_q), daemon=True)
    producer.start()

    frame_rows, detection_rows = [], []

    progress = tqdm(total=expected_samples, unit="frame", disable=not show_progress)

    def flush(batch):
        if not batch:
            return
        idxs = [b[0] for b in batch]
        frames = [b[1] for b in batch]
        per_frame_boxes = detector.detect_batch(
            frames, valid_mask=valid_mask, shadow_checker=shadow_checker,
            bg_bgr=bg_bgr, size_filter=size_filter,
        )
        for frame_idx, boxes in zip(idxs, per_frame_boxes):
            frame_rows.append((frame_idx, frame_idx / fps))
            for (x, y, w, h, conf) in boxes:
                detection_rows.append((frame_idx, x, y, w, h, conf))
        progress.update(len(batch))
        progress.set_postfix(
            t=f"{idxs[-1] / fps:.1f}s", detections=len(detection_rows), refresh=False,
        )

    batch = []
    try:
        while True:
            item = frame_q.get()
            if item is None:
                break
            batch.append(item)
            if len(batch) >= frame_batch_size:
                flush(batch)
                batch = []
        flush(batch)
    finally:
        progress.close()

    producer.join()
    return frame_rows, detection_rows, fps


def save_results(frame_rows, detection_rows, out_prefix, meta):
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    frames_df = pd.DataFrame(frame_rows, columns=["frame_idx", "timestamp_s"])
    dets_df = pd.DataFrame(
        detection_rows, columns=["frame_idx", "x", "y", "w", "h", "confidence"]
    )
    frames_df.to_parquet(f"{out_prefix}_frames.parquet", index=False)
    dets_df.to_parquet(f"{out_prefix}_detections.parquet", index=False)
    with open(f"{out_prefix}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    from config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video", default=None, help="overrides paths.video from config")
    parser.add_argument("--frame-stride", type=int, default=None,
                         help="overrides video_processing.frame_stride")
    parser.add_argument("--frame-batch-size", type=int, default=None,
                         help="overrides video_processing.frame_batch_size")
    parser.add_argument("--out-prefix", default=None, help="overrides video_processing.out_prefix")
    parser.add_argument("--no-progress", action="store_true", help="disable the progress bar")
    args = parser.parse_args()

    cfg = load_config(args.config)

    video_path = args.video or cfg.paths.video
    frame_stride = args.frame_stride or cfg.video_processing.frame_stride
    frame_batch_size = args.frame_batch_size or cfg.video_processing.frame_batch_size
    out_prefix = args.out_prefix or cfg.video_processing.out_prefix

    valid_mask = load_removal_mask(cfg.paths.mask, alpha_threshold=cfg.mask.alpha_threshold)

    shadow_checker = make_shadow_checker(
        ForegroundDetector(
            hue_shadow_tol=cfg.foreground_detector.hue_shadow_tol,
            sat_shadow_tol=cfg.foreground_detector.sat_shadow_tol,
            v_ratio_low=cfg.foreground_detector.v_ratio_low,
            v_ratio_high=cfg.foreground_detector.v_ratio_high,
            min_abs_diff=cfg.foreground_detector.min_abs_diff,
        ),
        shadow_ratio_thresh=cfg.shadow_checker.shadow_ratio_thresh,
        min_changed_fraction=cfg.shadow_checker.min_changed_fraction,
    )

    size_filter = PerspectiveSizeFilter(
        y_far=cfg.size_filter.y_far, h_far=cfg.size_filter.h_far,
        y_near=cfg.size_filter.y_near, h_near=cfg.size_filter.h_near,
        tolerance=cfg.size_filter.tolerance,
        min_aspect=cfg.size_filter.min_aspect, max_aspect=cfg.size_filter.max_aspect,
    )

    detector = TiledPlayerDetector(
        model_path=cfg.paths.yolo_model,
        tile_size=cfg.yolo.tile_size,
        overlap=cfg.yolo.overlap,
        conf_thresh=cfg.yolo.conf_thresh,
        person_class_id=cfg.yolo.person_class_id,
        device=cfg.yolo.device,
        cross_tile_nms_iou=cfg.yolo.cross_tile_nms_iou,
        containment_thresh=cfg.yolo.containment_thresh,
        gpu_batch_size=cfg.yolo.gpu_batch_size,
    )

    # peek the first sampled frame to get the real frame resolution, since
    # empty_field.png / mask.png can be a slightly different export resolution
    cap = cv2.VideoCapture(video_path)
    ret, probe_frame = cap.read()
    cap.release()
    if not ret:
        raise IOError(f"Could not read a frame from video: {video_path}")
    frame_hw = probe_frame.shape[:2]

    ref = cv2.imread(cfg.paths.empty_field, cv2.IMREAD_UNCHANGED)[:, :, :3]
    ref = resize_to_match(ref, frame_hw)
    valid_mask = resize_to_match(valid_mask.astype(np.uint8), frame_hw, cv2.INTER_NEAREST).astype(bool)

    frame_rows, detection_rows, fps = process_video(
        video_path, detector, frame_stride=frame_stride, frame_batch_size=frame_batch_size,
        valid_mask=valid_mask, shadow_checker=shadow_checker, bg_bgr=ref, size_filter=size_filter,
        show_progress=not args.no_progress,
    )

    meta = {
        "video_path": video_path,
        "fps": fps,
        "frame_stride": frame_stride,
        "frame_batch_size": frame_batch_size,
        "model_path": cfg.paths.yolo_model,
        "device": cfg.yolo.device,
        "conf_thresh": cfg.yolo.conf_thresh,
        "tile_size": cfg.yolo.tile_size,
        "overlap": cfg.yolo.overlap,
    }
    save_results(frame_rows, detection_rows, out_prefix, meta)

    print(f"processed {len(frame_rows)} sampled frames, {len(detection_rows)} detections")
    print(f"wrote {out_prefix}_frames.parquet, {out_prefix}_detections.parquet, {out_prefix}_meta.json")


if __name__ == "__main__":
    main()
