"""
YOLOv8-based player detection with tiled inference, for comparison against the
background-subtraction pipeline in player_detection_pipeline.py.

Run this on your GPU machine (needs internet access once, to pull the model):
    pip install ultralytics

Why tiled inference: your far-field players are ~35px tall in a 1920x1080 frame.
YOLO models are typically trained/run at ~640px input, meaning a full-frame
resize alone can shrink a 35px player down to ~12px -- often below reliable
detection size. Slicing the frame into overlapping tiles and running inference
per tile (then merging + de-duplicating boxes) keeps small players at a
detectable relative size. This costs more compute per frame than single-pass
inference, but at 8-10fps sampling on a 2hr match it's very tractable on GPU.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class TiledPlayerDetector:
    def __init__(self, model_path="yolov8n.pt", tile_size=640, overlap=0.25,
                 conf_thresh=0.25, person_class_id=0, device="cuda",
                 cross_tile_nms_iou=0.4, containment_thresh=0.8, gpu_batch_size=16):
        self.model = YOLO(model_path)
        self.tile_size = tile_size
        self.overlap = overlap
        self.conf_thresh = conf_thresh
        self.person_class_id = person_class_id  # COCO class 0 = "person"
        self.device = device
        self.cross_tile_nms_iou = cross_tile_nms_iou
        self.containment_thresh = containment_thresh
        # caps how many tile images go through a single model.predict() forward
        # pass in detect_batch() -- keeps GPU memory bounded regardless of how
        # many frames/tiles are handed to it at once
        self.gpu_batch_size = gpu_batch_size

    def _tiles(self, w, h):
        step = int(self.tile_size * (1 - self.overlap))
        xs = list(range(0, max(w - self.tile_size, 0) + 1, step)) or [0]
        ys = list(range(0, max(h - self.tile_size, 0) + 1, step)) or [0]
        if xs[-1] + self.tile_size < w:
            xs.append(w - self.tile_size)
        if ys[-1] + self.tile_size < h:
            ys.append(h - self.tile_size)
        for y in ys:
            for x in xs:
                yield x, y, min(x + self.tile_size, w), min(y + self.tile_size, h)

    def detect(self, frame_bgr, valid_mask=None, shadow_checker=None, bg_bgr=None, size_filter=None):
        h, w = frame_bgr.shape[:2]
        all_boxes, all_scores, all_tile_ids = [], [], []

        for tile_id, (x0, y0, x1, y1) in enumerate(self._tiles(w, h)):
            tile = frame_bgr[y0:y1, x0:x1]
            results = self.model.predict(
                tile, conf=self.conf_thresh, classes=[self.person_class_id],
                device=self.device, verbose=False,
            )[0]
            for box in results.boxes:
                bx0, by0, bx1, by1 = box.xyxy[0].tolist()
                all_boxes.append([bx0 + x0, by0 + y0, bx1 + x0, by1 + y0])
                all_scores.append(float(box.conf[0]))
                all_tile_ids.append(tile_id)

        if not all_boxes:
            return []

        return self._postprocess(
            np.array(all_boxes), np.array(all_scores), np.array(all_tile_ids),
            frame_bgr, valid_mask, shadow_checker, bg_bgr, size_filter,
        )

    def detect_batch(self, frames_bgr, valid_mask=None, shadow_checker=None, bg_bgr=None, size_filter=None):
        """
        Batched version of detect() for a list of frames sharing the same
        resolution (true for consecutive samples from one video). Tiles from
        every frame are flattened into one list and sent through model.predict()
        together (chunked internally to gpu_batch_size images per forward pass)
        instead of one predict() call per tile per frame -- that's what actually
        lets a GPU parallelize across frames; looping detect() per frame leaves
        it mostly idle between many tiny calls.

        Returns a list of per-frame detection lists, same order and format as
        detect() (each entry: [(x, y, w, h, confidence), ...]).
        """
        if not frames_bgr:
            return []

        h, w = frames_bgr[0].shape[:2]
        tiles_grid = list(self._tiles(w, h))

        flat_tiles, tile_frame_idx, tile_ids, tile_origins = [], [], [], []
        for f_i, frame in enumerate(frames_bgr):
            for tile_id, (x0, y0, x1, y1) in enumerate(tiles_grid):
                flat_tiles.append(frame[y0:y1, x0:x1])
                tile_frame_idx.append(f_i)
                tile_ids.append(tile_id)
                tile_origins.append((x0, y0))

        results = self.model.predict(
            flat_tiles, conf=self.conf_thresh, classes=[self.person_class_id],
            device=self.device, batch=self.gpu_batch_size, verbose=False,
        )

        per_frame_boxes = [[] for _ in frames_bgr]
        per_frame_scores = [[] for _ in frames_bgr]
        per_frame_tile_ids = [[] for _ in frames_bgr]
        for res, f_i, tile_id, (x0, y0) in zip(results, tile_frame_idx, tile_ids, tile_origins):
            for box in res.boxes:
                bx0, by0, bx1, by1 = box.xyxy[0].tolist()
                per_frame_boxes[f_i].append([bx0 + x0, by0 + y0, bx1 + x0, by1 + y0])
                per_frame_scores[f_i].append(float(box.conf[0]))
                per_frame_tile_ids[f_i].append(tile_id)

        out = []
        for f_i, frame in enumerate(frames_bgr):
            if not per_frame_boxes[f_i]:
                out.append([])
                continue
            out.append(self._postprocess(
                np.array(per_frame_boxes[f_i]), np.array(per_frame_scores[f_i]),
                np.array(per_frame_tile_ids[f_i]), frame, valid_mask, shadow_checker, bg_bgr, size_filter,
            ))
        return out

    def _postprocess(self, boxes_xyxy, scores, tile_ids, frame_bgr,
                      valid_mask=None, shadow_checker=None, bg_bgr=None, size_filter=None):
        # only suppress across DIFFERENT tiles (same-tile boxes are already
        # resolved by the model's own internal NMS -- re-merging them here is
        # what causes two genuinely close players to collapse into one box)
        keep = self._cross_tile_nms(boxes_xyxy, scores, tile_ids, iou_thresh=self.cross_tile_nms_iou)
        boxes_xyxy, scores = boxes_xyxy[keep], scores[keep]

        # drop boxes that are near-fully contained inside a higher/equal
        # confidence box (fixes "head detected as its own separate player")
        keep = self._suppress_contained(boxes_xyxy, scores, containment_thresh=self.containment_thresh)
        boxes_xyxy, scores = boxes_xyxy[keep], scores[keep]

        out = []
        for (bx0, by0, bx1, by1), score in zip(boxes_xyxy, scores):
            cx, cy = int((bx0 + bx1) / 2), int(by1)  # feet point
            if valid_mask is not None:
                if not (0 <= cy < valid_mask.shape[0] and 0 <= cx < valid_mask.shape[1]):
                    continue
                if not valid_mask[cy, cx]:
                    continue  # outside field-of-play (bench/sideline) -> drop

            box = (int(bx0), int(by0), int(bx1 - bx0), int(by1 - by0))

            if size_filter is not None and not size_filter.is_plausible(box):
                continue  # wrong shape/size for a person at this image row (e.g. a bat)

            # hybrid shadow check: reuses the classical pipeline's HSV shadow
            # test against the pixels inside this box, to catch cases like a
            # harsh-light shadow silhouette getting misclassified as a person
            if shadow_checker is not None and bg_bgr is not None:
                if shadow_checker(frame_bgr, bg_bgr, box):
                    continue

            out.append((*box, float(score)))  # (x, y, w, h, confidence)
        return out

    @staticmethod
    def _cross_tile_nms(boxes, scores, tile_ids, iou_thresh=0.4):
        idxs = scores.argsort()[::-1]
        keep = []
        suppressed = np.zeros(len(boxes), dtype=bool)
        for i in idxs:
            if suppressed[i]:
                continue
            keep.append(i)
            same_tile = tile_ids == tile_ids[i]
            candidates = np.where(~suppressed & ~same_tile)[0]
            if len(candidates) == 0:
                continue
            ious = TiledPlayerDetector._iou(boxes[i], boxes[candidates])
            suppressed[candidates[ious >= iou_thresh]] = True
        return keep

    @staticmethod
    def _suppress_contained(boxes, scores, containment_thresh=0.8):
        """Drops box j if most of its area is contained within a box i that
        has equal or higher confidence (handles head-inside-torso, hand-inside-
        body, etc. where plain IoU-NMS fails because the areas differ a lot)."""
        n = len(boxes)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = scores.argsort()[::-1]
        drop = np.zeros(n, dtype=bool)
        for a in range(n):
            i = order[a]
            if drop[i]:
                continue
            for b in range(a + 1, n):
                j = order[b]
                if drop[j]:
                    continue
                x0 = max(boxes[i, 0], boxes[j, 0])
                y0 = max(boxes[i, 1], boxes[j, 1])
                x1 = min(boxes[i, 2], boxes[j, 2])
                y1 = min(boxes[i, 3], boxes[j, 3])
                inter = max(0, x1 - x0) * max(0, y1 - y0)
                if areas[j] > 0 and inter / areas[j] >= containment_thresh:
                    drop[j] = True
        return [i for i in range(n) if not drop[i]]

    @staticmethod
    def _iou(box, boxes):
        x0 = np.maximum(box[0], boxes[:, 0])
        y0 = np.maximum(box[1], boxes[:, 1])
        x1 = np.minimum(box[2], boxes[:, 2])
        y1 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
        area_a = (box[2] - box[0]) * (box[3] - box[1])
        area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return inter / (area_a + area_b - inter + 1e-6)


def make_shadow_checker(fg_detector, shadow_ratio_thresh=0.7, min_changed_fraction=0.05):
    """
    Builds a shadow_checker(frame_bgr, bg_bgr, box) -> bool function for use
    with TiledPlayerDetector.detect(). Reuses ForegroundDetector's HSV
    hue/saturation/value shadow test (see player_detection_pipeline.py) against
    the pixels inside a YOLO box.

    IMPORTANT: the ratio is computed among CHANGED pixels only (pixels that
    differ from the background at all), not against the whole box area. A
    bounding box always contains a lot of legitimately-unchanged clay around
    an irregular player silhouette (gaps between limbs, box padding, etc.) --
    comparing against total box area wrongly flags real players as "shadow"
    whenever a large fraction of their box happens to be plain background.
    We only ask: "of the pixels that changed, how many are shadow-colored?"

    shadow_ratio_thresh: box is rejected if this fraction (or more) of its
        CHANGED pixels are shadow-classified.
    min_changed_fraction: if fewer than this fraction of the box changed at
        all, there isn't enough signal to judge -- don't reject (avoids
        false rejections on small/noisy boxes).
    """
    def check(frame_bgr, bg_bgr, box):
        x, y, w, h = box
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, frame_bgr.shape[1]), min(y + h, frame_bgr.shape[0])
        if x1 <= x0 or y1 <= y0:
            return False
        crop_frame = frame_bgr[y0:y1, x0:x1]
        crop_bg = bg_bgr[y0:y1, x0:x1]
        crop_valid = np.ones(crop_frame.shape[:2], dtype=bool)

        _, is_shadow, is_changed = fg_detector.detect_detailed(crop_frame, crop_bg, crop_valid)

        n_changed = is_changed.sum()
        if n_changed < min_changed_fraction * is_changed.size:
            return False  # not enough signal -> don't reject

        shadow_ratio = is_shadow[is_changed].mean()
        return shadow_ratio >= shadow_ratio_thresh
    return check


if __name__ == "__main__":
    from config import load_config
    import time
    from player_detection_pipeline import load_removal_mask, ForegroundDetector, PerspectiveSizeFilter

    cfg = load_config("config.yaml")

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
    )

    from player_detection_pipeline import resize_to_match

    ref = cv2.imread(cfg.paths.empty_field, cv2.IMREAD_UNCHANGED)[:, :, :3]
    frame = cv2.imread("game_frame_3.jpeg", cv2.IMREAD_UNCHANGED)[:, :, :3]  # swap for your actual frame source
    # guards against empty_field.png/mask.png being a slightly different
    # resolution than your actual video frames (happened once already --
    # a few pixels off breaks every pixel-wise comparison in this pipeline)
    ref = resize_to_match(ref, frame.shape[:2])
    valid_mask = resize_to_match(valid_mask.astype(np.uint8), frame.shape[:2], cv2.INTER_NEAREST).astype(bool)
    start_time = time.time()
    boxes = detector.detect(frame, valid_mask=valid_mask, shadow_checker=shadow_checker,
                             bg_bgr=ref, size_filter=size_filter)
    end_time = time.time()
    print(f"detected {len(boxes)} players")
    print(f"Processing time of detection: {end_time - start_time:.2f} seconds")
    vis = frame.copy()
    for (x, y, w, h, conf) in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y - 4, th + 2)  # keep label on-screen if box is near the top edge
        cv2.rectangle(vis, (x, ty - th - 4), (x + tw + 4, ty), (0, 255, 0), -1)
        cv2.putText(vis, label, (x + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite("yolo_detections_overlay.png", vis)