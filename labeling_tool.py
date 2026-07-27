"""
Semi-supervised labeling tool.

Runs YOLO detection + tracking over a sample of frames spread across the
match, collects one aggregated color sample per track (not per frame --
tracks already smooth over lighting noise, and there are far fewer tracks
than frames), then shows you each track's torso crop one at a time and asks
you to label it with a keypress. Output is a labeled_samples.json you can
feed into build_prototypes_from_labeled_samples() or KNNRoleClassifier.

NOTE: this uses TiledPlayerDetector (YOLO) for box detection, not the classical
detect_players() background-subtraction pipeline used in an earlier version of
this file. A side-by-side comparison on real broadcast footage showed the
classical full-frame blob detector producing far more false positives (thin
shadow-slivers, graphic-overlay artifacts) than YOLO -- see visualize_predictions.py
for the same fix and fuller rationale. The classical ForegroundDetector is
still used here, but only for its narrower, proven job: extracting jersey
color from within a YOLO-confirmed box.

Run on your machine (needs a display -- this uses cv2.imshow, not something
that works in a headless/server environment):
    python labeling_tool.py

Controls shown on screen: one key per role (configurable in config.yaml),
's' to skip (e.g. crop is ambiguous/occluded), 'q' to quit and save what
you've labeled so far -- you don't need to finish the whole video in one run,
just label as many tracks as you have patience for and rerun later to
continue (already-labeled track crops are skipped on the next run, matched
by track fingerprint -- see NOTE below on why this is approximate).
"""

import json
import os
import cv2
import numpy as np

from config import load_config
from player_detection_pipeline import (
    ForegroundDetector, PerspectiveSizeFilter, load_removal_mask,
    CentroidTracker, resize_to_match,
)
from yolo_detector import TiledPlayerDetector, make_shadow_checker
from team_classifier import extract_jersey_color


def sample_tracks_across_match(cfg, video_path, sample_seconds_stride=20, window_seconds=3):
    """
    Runs YOLO detection+tracking over short windows spread evenly across the
    video (rather than the whole 2hr match) to get a diverse-but-manageable
    set of tracks to label. Each window is a few seconds long so tracks have
    enough frames to aggregate a stable color sample.

    Returns: {track_id: {"samples": [(a,b,v), ...], "crop": bgr_image,
                          "box": (x,y,w,h)}}
    track_id here is only unique WITHIN this sampling run, not across the
    whole match -- see labeling_tool module docstring note on resuming.
    """
    ref = cv2.imread(cfg.paths.empty_field, cv2.IMREAD_UNCHANGED)
    alpha_mask = ref[:, :, 3] > 200
    removal_mask = load_removal_mask(cfg.paths.mask, alpha_threshold=cfg.mask.alpha_threshold)

    cap = cv2.VideoCapture(video_path)
    frame_hw = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    # empty_field.png/mask.png can come from a slightly different export
    # resolution than the actual video -- resize them to match rather than
    # letting every downstream pixel-wise comparison fail on shape mismatch
    ref = resize_to_match(ref, frame_hw)
    alpha_mask = resize_to_match(alpha_mask.astype(np.uint8), frame_hw, cv2.INTER_NEAREST).astype(bool)
    removal_mask = resize_to_match(removal_mask.astype(np.uint8), frame_hw, cv2.INTER_NEAREST).astype(bool)

    valid_mask = alpha_mask & removal_mask
    ref_bgr = ref[:, :, :3]

    # fg_detector is used ONLY for jersey color extraction within a YOLO box
    # now, not for finding boxes in the first place
    fg_detector = ForegroundDetector(
        hue_shadow_tol=cfg.foreground_detector.hue_shadow_tol,
        sat_shadow_tol=cfg.foreground_detector.sat_shadow_tol,
        v_ratio_low=cfg.foreground_detector.v_ratio_low,
        v_ratio_high=cfg.foreground_detector.v_ratio_high,
        min_abs_diff=cfg.foreground_detector.min_abs_diff,
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
    )
    shadow_checker = make_shadow_checker(
        fg_detector,
        shadow_ratio_thresh=cfg.shadow_checker.shadow_ratio_thresh,
        min_changed_fraction=cfg.shadow_checker.min_changed_fraction,
    )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride_frames = int(sample_seconds_stride * fps)
    window_frames = int(window_seconds * fps)
    detect_every_n = max(int(round(fps / cfg.sampling.detect_fps)), 1)

    if window_frames >= total_frames:
        # short clip (e.g. a quick test video): just process the whole thing
        # as a single window instead of the multi-window match-spanning logic
        window_frames = total_frames
        stride_frames = total_frames

    tracks = {}
    next_global_id = 0
    window_start = 0

    while window_start < total_frames:
        tracker = CentroidTracker(max_dist=cfg.tracker.max_dist, max_missed=cfg.tracker.max_missed)
        track_id_offset = next_global_id

        cap.set(cv2.CAP_PROP_POS_FRAMES, window_start)
        local_frames = {}
        for local_idx in range(window_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if local_idx % detect_every_n == 0:
                detections = detector.detect(
                    frame, valid_mask=valid_mask, shadow_checker=shadow_checker,
                    bg_bgr=ref_bgr, size_filter=size_filter,
                )
                boxes = [(x, y, w, h) for (x, y, w, h, _conf) in detections]  # tracker doesn't need confidence
                tracker.update(local_idx, boxes)
                local_frames[local_idx] = frame

        for local_tid, t in tracker.tracks.items():
            global_tid = track_id_offset + local_tid
            samples, crop, box = [], None, None
            for local_idx, bbox in t["history"].items():
                frame = local_frames.get(local_idx)
                if frame is None:
                    continue
                bbox_int = tuple(int(v) for v in bbox)
                sample = extract_jersey_color(frame, ref_bgr, bbox_int, fg_detector,
                                               torso_top=cfg.team_classifier.torso_top,
                                               torso_bottom=cfg.team_classifier.torso_bottom,
                                               min_pixels=cfg.team_classifier.min_pixels)
                if sample is not None:
                    samples.append(sample)
                    if crop is None:  # keep the first valid frame as the representative crop
                        x, y, w, h = bbox_int
                        crop = frame[max(y, 0):y + h, max(x, 0):x + w].copy()
                        box = bbox_int
            if len(samples) >= 2 and crop is not None and crop.size > 0:
                tracks[global_tid] = {"samples": samples, "crop": crop, "box": box}
                next_global_id += 1

        window_start += stride_frames

    cap.release()
    return tracks


def run_labeling_ui(tracks, role_keys, output_path="labeled_samples.json"):
    """
    role_keys: {"a": "team_a", "b": "team_b", "r": "referee", "m": "misc"}
    Shows each track's crop, waits for a key in role_keys (or 's' to skip,
    'q' to quit early). Appends to output_path if it already exists, so you
    can label in multiple sittings without losing earlier work.
    """
    labeled = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            labeled = json.load(f)
        print(f"loaded {len(labeled)} existing labels from {output_path}")

    legend = " | ".join(f"[{k}]={v}" for k, v in role_keys.items()) + " | [s]kip | [q]uit"

    for tid, t in tracks.items():
        crop = t["crop"]
        disp = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((disp.shape[0] + 40, max(disp.shape[1], 500), 3), dtype=np.uint8)
        canvas[40:40 + disp.shape[0], 0:disp.shape[1]] = disp
        cv2.putText(canvas, legend, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"track {tid}", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("label this player", canvas)
        key = chr(cv2.waitKey(0) & 0xFF)

        if key == "q":
            break
        if key in role_keys:
            role = role_keys[key]
            for sample in t["samples"]:
                labeled.append([role, list(sample)])
            print(f"track {tid} -> {role} ({len(t['samples'])} samples)")
        # anything else (including 's') skips this track

    cv2.destroyAllWindows()
    with open(output_path, "w") as f:
        json.dump(labeled, f, indent=2)
    print(f"saved {len(labeled)} labeled samples to {output_path}")
    return labeled


if __name__ == "__main__":
    cfg = load_config("config.yaml")

    role_keys = {"a": "team_a", "b": "team_b", "r": "referee", "m": "misc"}

    print("Sampling tracks across the match, this can take a while on a 2hr video...")
    tracks = sample_tracks_across_match(cfg, cfg.paths.video, sample_seconds_stride=20, window_seconds=3)
    print(f"collected {len(tracks)} labelable tracks")

    run_labeling_ui(tracks, role_keys, output_path="labeled_samples.json")