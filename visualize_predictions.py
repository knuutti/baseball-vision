"""
Visual sanity check: samples a handful of frames spread across a clip, runs
YOLO detection + the labeled-data role classifier on each, and saves annotated
overlay images so you can eyeball whether boxes and predicted roles look right
before scaling up to full-match processing.

NOTE: this uses TiledPlayerDetector (YOLO) for box detection, not the classical
detect_players() background-subtraction pipeline. A side-by-side comparison on
real broadcast footage showed the classical full-frame blob detector producing
far more false positives (thin shadow-slivers, graphic-overlay artifacts) than
YOLO, which recognizes person-shapes semantically rather than diffing pixels
against a reference. The classical ForegroundDetector is still used here, but
only for its narrower, already-proven job: extracting jersey color from within
a YOLO-confirmed box, and (optionally) the shadow cross-check on that box --
both much easier problems than full-frame detection.

Run:
    python visualize_predictions.py
"""

import json
import os
import cv2
import numpy as np

from config import load_config
from player_detection_pipeline import (
    ForegroundDetector, PerspectiveSizeFilter, load_removal_mask, resize_to_match,
)
from yolo_detector import TiledPlayerDetector, make_shadow_checker
from team_classifier import extract_jersey_color, build_prototypes_from_labeled_samples, ColorRoleClassifier


ROLE_COLORS = {
    # BGR, distinct + readable on the reddish clay background
    "team_a": (255, 100, 0),
    "team_b": (0, 200, 0),
    "referee": (0, 0, 255),
    "misc": (0, 200, 255),
}
UNKNOWN_COLOR = (180, 180, 180)


def load_classifier(labeled_samples_path="labeled_samples.json"):
    with open(labeled_samples_path) as f:
        raw = json.load(f)
    labeled_samples = [(role, tuple(feat)) for role, feat in raw]
    prototypes = build_prototypes_from_labeled_samples(labeled_samples)
    print("prototypes built from labeled data:", prototypes)
    return ColorRoleClassifier(prototypes)


def setup_pipeline(cfg, frame_hw):
    """
    Returns everything needed for detection (YOLO) + jersey color extraction
    (classical ForegroundDetector, used only within already-detected boxes).
    """
    ref = cv2.imread(cfg.paths.empty_field, cv2.IMREAD_UNCHANGED)
    ref = resize_to_match(ref, frame_hw)
    alpha_mask = ref[:, :, 3] > 200
    removal_mask = load_removal_mask(cfg.paths.mask, alpha_threshold=cfg.mask.alpha_threshold)
    removal_mask = resize_to_match(removal_mask.astype(np.uint8), frame_hw, cv2.INTER_NEAREST).astype(bool)
    valid_mask = alpha_mask & removal_mask

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
    return ref[:, :, :3], valid_mask, fg_detector, size_filter, detector, shadow_checker


def draw_prediction(vis, box, role, confident, yolo_conf=None):
    x, y, w, h = box
    color = ROLE_COLORS.get(role, UNKNOWN_COLOR)
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
    label = f"{role or 'unknown'}" + ("" if confident else "?")
    if yolo_conf is not None:
        label += f" {yolo_conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = max(y - 4, th + 2)
    cv2.rectangle(vis, (x, ty - th - 4), (x + tw + 4, ty), color, -1)
    cv2.putText(vis, label, (x + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def visualize(cfg, video_path, classifier, n_frames=6, output_dir="prediction_previews"):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_hw = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ref_bgr, valid_mask, fg_detector, size_filter, detector, shadow_checker = setup_pipeline(cfg, frame_hw)

    display_positions = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    for i, pos in enumerate(display_positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ret, frame = cap.read()
        if not ret:
            continue

        boxes = detector.detect(
            frame, valid_mask=valid_mask, shadow_checker=shadow_checker,
            bg_bgr=ref_bgr, size_filter=size_filter,
        )

        vis = frame.copy()
        for (x, y, w, h, yolo_conf) in boxes:
            box = (x, y, w, h)
            sample = extract_jersey_color(frame, ref_bgr, box, fg_detector,
                                           torso_top=cfg.team_classifier.torso_top,
                                           torso_bottom=cfg.team_classifier.torso_bottom,
                                           min_pixels=cfg.team_classifier.min_pixels)
            if sample is None:
                draw_prediction(vis, box, None, False, yolo_conf)
                continue
            role, confident = classifier.classify_track([sample])
            draw_prediction(vis, box, role, confident, yolo_conf)

        out_path = os.path.join(output_dir, f"frame_{i:02d}_pos{int(pos)}.png")
        cv2.imwrite(out_path, vis)
        print(f"saved {out_path} ({len(boxes)} boxes)")

    cap.release()


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    classifier = load_classifier("labeled_samples.json")
    visualize(cfg, cfg.paths.video, classifier, n_frames=6)