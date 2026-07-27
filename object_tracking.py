"""
Simple object tracking demo.

Pipeline:
  1. Sample N frames from a video, one every `frame_stride` frames (default 12
     -> 5fps at 60fps source; tunable).
  2. Run YOLO (TiledPlayerDetector) on each sampled frame.
  3. Link boxes across consecutive sampled frames using RANSAC to find the
     dominant shared motion, with a nearest-neighbor fallback for detections
     that don't fit it, and a "missed frame" tolerance so tracks survive an
     object briefly leaving the visible field/frame.

Note on the matching approach -- worth understanding before trusting the output:
RANSAC fits a single dominant TRANSLATION (dx, dy) that best explains how most
box centroids moved between two sampled frames -- a reasonable "majority
motion" estimate, not a claim that every player shares one motion (they
don't; players run independently). That translation is used only as a PRIOR:
it shifts each track's predicted position before scoring candidate matches.
The actual assignment decision is then made by solving a combined cost matrix
(motion-compensated centroid distance + IoU/box overlap) with the Hungarian
algorithm (scipy.optimize.linear_sum_assignment) -- the globally optimal
one-to-one matching, not a greedy nearest-first pick. This combination fixes
two specific failure modes: a player's box shrinking/shifting during partial
occlusion (IoU still overlaps substantially even when the centroid moves more
than expected), and two nearby tracks/detections getting swapped (greedy
matching can lock in a locally-plausible but globally-wrong pairing; Hungarian
can't). A player moving against the majority's dominant motion is still
handled correctly -- IoU and distance are still evaluated for every pair, the
translation is only a prior nudging the distance term, not a hard filter.

Treat this as a demo-level heuristic, not a production multi-object tracker --
a real tracker (e.g. SORT/DeepSORT) typically gives each track its own
individual velocity estimate (e.g. a per-track Kalman filter) rather than
leaning on one shared motion prior for everyone. This script is a reasonable
stepping stone toward that, not a replacement for it.

Run:
    python object_tracking.py
"""

import os
import cv2
import numpy as np

from config import load_config
from player_detection_pipeline import ForegroundDetector, PerspectiveSizeFilter, load_removal_mask, resize_to_match
from yolo_detector import TiledPlayerDetector, make_shadow_checker


# ----------------------------------------------------------------------------
# Frame sampling + detection
# ----------------------------------------------------------------------------
def sample_frames(video_path, frame_stride=12, n_frames=10, start_frame=0):
    """
    Returns [(frame_idx, frame_bgr), ...], one every frame_stride frames,
    starting at start_frame, up to n_frames frames (fewer if the video ends
    first).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while len(frames) < n_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((frame_idx, frame))
        frame_idx += frame_stride

    cap.release()
    return frames


def detect_boxes_per_frame(frames, detector, valid_mask, shadow_checker, ref_bgr, size_filter):
    """
    frames: [(frame_idx, frame_bgr), ...]
    Returns [[(x, y, w, h, conf), ...], ...] -- one list of boxes per frame,
    same order as `frames`.
    """
    all_boxes = []
    for frame_idx, frame in frames:
        boxes = detector.detect(
            frame, valid_mask=valid_mask, shadow_checker=shadow_checker,
            bg_bgr=ref_bgr, size_filter=size_filter,
        )
        all_boxes.append(boxes)
        print(f"frame {frame_idx}: {len(boxes)} boxes")
    return all_boxes


# ----------------------------------------------------------------------------
# RANSAC-estimated motion prior + Hungarian-solved combined matching
# ----------------------------------------------------------------------------
def _box_centroids(boxes):
    return np.array([[x + w / 2, y + h] for (x, y, w, h, *_rest) in boxes]) if boxes else np.zeros((0, 2))


def ransac_dominant_translation(prev_pts, curr_pts, max_iters=300, inlier_thresh=25.0, rng=None):
    """
    Finds the (dx, dy) translation that maximizes the number of prev points
    whose nearest curr point (after applying the translation) falls within
    inlier_thresh pixels. Classic RANSAC with a minimal model of size 1 (a
    single point correspondence is enough to hypothesize a translation).

    Used here ONLY to estimate a motion prior for the assignment step below --
    it no longer produces the final matches itself (see match_frame_pair).
    Returns just the translation; inlier bookkeeping isn't needed anymore.
    """
    rng = rng or np.random.default_rng()
    if len(prev_pts) == 0 or len(curr_pts) == 0:
        return np.array([0.0, 0.0])

    best_translation = np.array([0.0, 0.0])
    best_match_count = -1

    for _ in range(max_iters):
        i = rng.integers(0, len(prev_pts))
        j = rng.integers(0, len(curr_pts))
        candidate = curr_pts[j] - prev_pts[i]

        predicted = prev_pts + candidate  # (P, 2)
        dists = np.linalg.norm(curr_pts[None, :, :] - predicted[:, None, :], axis=2)  # (P, C)
        nearest_dist = dists.min(axis=1)
        count = (nearest_dist < inlier_thresh).sum()

        if count > best_match_count:
            best_match_count = count
            best_translation = candidate

    return best_translation


def box_iou(box_a, box_b):
    """IoU between two (x, y, w, h, ...) boxes."""
    ax0, ay0, aw, ah = box_a[:4]
    bx0, by0, bw, bh = box_b[:4]
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _shift_box(box, dx, dy):
    x, y, w, h = box[:4]
    return (x + dx, y + dy, w, h)


def build_cost_matrix(prev_boxes, curr_boxes, translation, w_dist=0.5, w_iou=0.5, dist_norm=150.0):
    """
    Combined cost per (prev, curr) pair: a weighted sum of
      - motion-compensated centroid distance (prev centroid shifted by the
        RANSAC-estimated dominant translation, before comparing to curr) --
        this uses RANSAC's estimate as a PRIOR on where a track should be,
        rather than assuming it stayed still.
      - 1 - IoU, using the prev box similarly shifted by the translation
        before overlap is computed -- this is what actually fixes the
        occlusion-driven ID-swap problem: a box that shrank because a player
        got partly occluded usually still overlaps substantially with its
        shifted-forward previous position, even when its centroid moved more
        than expected from the size change alone.
    Both terms are normalized to roughly [0, 1] so the weights are comparable.
    """
    n, m = len(prev_boxes), len(curr_boxes)
    cost = np.zeros((n, m))
    prev_centroids = _box_centroids(prev_boxes)
    curr_centroids = _box_centroids(curr_boxes)
    predicted_centroids = prev_centroids + translation

    for i in range(n):
        shifted_box = _shift_box(prev_boxes[i], translation[0], translation[1])
        for j in range(m):
            dist = np.linalg.norm(predicted_centroids[i] - curr_centroids[j])
            dist_cost = min(dist / dist_norm, 1.0)
            iou = box_iou(shifted_box, curr_boxes[j])
            iou_cost = 1.0 - iou
            cost[i, j] = w_dist * dist_cost + w_iou * iou_cost

    return cost


def hungarian_match(prev_boxes, curr_boxes, translation, w_dist=0.5, w_iou=0.5,
                     dist_norm=150.0, reject_cost=0.85):
    """
    Solves the assignment problem optimally (globally minimal total cost),
    rather than greedily -- this is what fixes ID swaps when two tracks and
    two detections are all close together: greedy matching can lock in a
    locally-plausible but globally-wrong pairing, Hungarian assignment can't.

    reject_cost: assignments above this cost are discarded (treated as "no
    match" rather than a forced bad pairing) -- Hungarian's scipy
    implementation still returns a full assignment even when some pairs are a
    poor fit, so this threshold is what lets a track legitimately end (no
    curr box is a reasonable match) or a detection start a new track.
    """
    from scipy.optimize import linear_sum_assignment

    if not prev_boxes or not curr_boxes:
        return []

    cost = build_cost_matrix(prev_boxes, curr_boxes, translation, w_dist, w_iou, dist_norm)
    row_ind, col_ind = linear_sum_assignment(cost)

    return [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if cost[r, c] <= reject_cost]


def match_frame_pair(prev_boxes, curr_boxes, ransac_iters=300, ransac_inlier_thresh=25.0,
                      w_dist=0.5, w_iou=0.5, dist_norm=150.0, reject_cost=0.85, rng=None):
    """
    Returns list of (prev_idx, curr_idx) matches between two frames' box
    lists. RANSAC estimates the dominant shared motion (a prior on where
    tracks should be); Hungarian assignment on a combined IoU+distance cost
    matrix does the actual one-to-one matching decision.
    """
    prev_pts = _box_centroids(prev_boxes)
    curr_pts = _box_centroids(curr_boxes)

    translation = ransac_dominant_translation(
        prev_pts, curr_pts, max_iters=ransac_iters, inlier_thresh=ransac_inlier_thresh, rng=rng,
    )

    return hungarian_match(
        prev_boxes, curr_boxes, translation,
        w_dist=w_dist, w_iou=w_iou, dist_norm=dist_norm, reject_cost=reject_cost,
    )


# ----------------------------------------------------------------------------
# Track building across all sampled frames (with disappear/reappear tolerance)
# ----------------------------------------------------------------------------
class SimpleTrackBuilder:
    """
    Maintains active tracks across a sequence of sampled frames' box lists.
    Each active track remembers its last-known box and how many consecutive
    sampled frames it's gone unmatched (missed) -- if a track goes unmatched
    for more than max_missed sampled frames, it's considered ended rather than
    kept alive indefinitely.
    """

    def __init__(self, max_missed=2):
        self.max_missed = max_missed
        self.next_id = 0
        self.active = {}  # track_id -> {"box": (x,y,w,h,conf), "missed": int}
        self.history = {}  # track_id -> {frame_idx: (x,y,w,h,conf)}

    def process(self, frames_boxes, frame_indices, ransac_iters=300, ransac_inlier_thresh=25.0,
                w_dist=0.5, w_iou=0.5, dist_norm=150.0, reject_cost=0.85, rng=None):
        if not frames_boxes:
            return

        # seed tracks from the first frame
        for box in frames_boxes[0]:
            self._start_track(frame_indices[0], box)

        for t in range(1, len(frames_boxes)):
            prev_active_ids = list(self.active.keys())
            prev_boxes = [self.active[tid]["box"] for tid in prev_active_ids]
            curr_boxes = frames_boxes[t]

            matches = match_frame_pair(
                prev_boxes, curr_boxes, ransac_iters=ransac_iters,
                ransac_inlier_thresh=ransac_inlier_thresh,
                w_dist=w_dist, w_iou=w_iou, dist_norm=dist_norm, reject_cost=reject_cost, rng=rng,
            )

            matched_prev_idx = {p for p, _ in matches}
            matched_curr_idx = {c for _, c in matches}

            for prev_idx, curr_idx in matches:
                tid = prev_active_ids[prev_idx]
                box = curr_boxes[curr_idx]
                self.active[tid]["box"] = box
                self.active[tid]["missed"] = 0
                self.history[tid][frame_indices[t]] = box

            # unmatched previously-active tracks: bump missed count, drop if over budget
            for prev_idx, tid in enumerate(prev_active_ids):
                if prev_idx in matched_prev_idx:
                    continue
                self.active[tid]["missed"] += 1
                if self.active[tid]["missed"] > self.max_missed:
                    del self.active[tid]

            # unmatched current boxes: new tracks
            for curr_idx, box in enumerate(curr_boxes):
                if curr_idx not in matched_curr_idx:
                    self._start_track(frame_indices[t], box)

    def _start_track(self, frame_idx, box):
        tid = self.next_id
        self.next_id += 1
        self.active[tid] = {"box": box, "missed": 0}
        self.history[tid] = {frame_idx: box}


# ----------------------------------------------------------------------------
# Recommended alternative: track building via the `trackers` package
# ----------------------------------------------------------------------------
def build_tracks_with_bytetrack(frames_boxes, frame_indices, frame_rate=5,
                                 lost_track_buffer=60, minimum_consecutive_frames=1,
                                 track_activation_threshold=0.25, minimum_iou_threshold=0.8):
    """
    Alternative to SimpleTrackBuilder above: uses Roboflow's `trackers`
    package (pip install trackers) -- a maintained, purpose-built library for
    wiring a custom detector into established MOT algorithms (SORT, ByteTrack,
    OC-SORT, BoT-SORT) via the shared `supervision.Detections` format, rather
    than reaching into ultralytics' internal tracker classes directly (which
    aren't really meant to be driven manually -- people online had to
    reverse-engineer undocumented attribute access to do it).

    Why this is the recommended path over the hand-rolled RANSAC+Hungarian
    tracker above, for anything beyond a demo: battle-tested, benchmarked
    association logic; a per-track Kalman filter (each track predicts its own
    next position from its own motion history, rather than everyone leaning
    on one shared RANSAC-estimated motion prior); and ByteTrack's core idea of
    still trying to associate LOW-confidence detections instead of discarding
    them outright -- directly relevant to your case, since a partially
    occluded player often still produces a box, just a lower-confidence one,
    rather than no box at all.

    Requires: pip install trackers  (needs Python >= 3.10)

    NOTE: constructor parameter names have changed between versions of this
    package (e.g. minimum_matching_threshold -> minimum_iou_threshold at some
    point) -- if you hit a TypeError about an unexpected keyword argument,
    check `help(ByteTrackTracker.__init__)` in your installed version rather
    than assuming this signature is current; the error message itself is
    usually enough (it suggested the fix directly when this happened).

    frame_rate: pass your EFFECTIVE sampling rate here (e.g. source video fps
    / frame_stride), not the source video's raw fps. lost_track_buffer and
    related internals are frame-count budgets against whatever sequence you
    actually feed the tracker -- here, your sparsely sampled frames, not
    the original continuous video.

    Returns the same shape as SimpleTrackBuilder.history:
    {track_id: {frame_idx: (x, y, w, h, conf)}}
    """
    import supervision as sv
    from trackers import ByteTrackTracker

    tracker = ByteTrackTracker(
        lost_track_buffer=lost_track_buffer,
        minimum_consecutive_frames=minimum_consecutive_frames,
        track_activation_threshold=track_activation_threshold,
        minimum_iou_threshold=minimum_iou_threshold,
        frame_rate=frame_rate,
    )

    history = {}
    for frame_idx, boxes in zip(frame_indices, frames_boxes):
        if boxes:
            xyxy = np.array([[x, y, x + w, y + h] for (x, y, w, h, conf) in boxes], dtype=np.float32)
            confidence = np.array([conf for (*_, conf) in boxes], dtype=np.float32)
            class_id = np.zeros(len(boxes), dtype=int)
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            confidence = np.zeros((0,), dtype=np.float32)
            class_id = np.zeros((0,), dtype=int)

        detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracked = tracker.update(detections)

        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            x0, y0, x1, y1 = tracked.xyxy[i]
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
            box = (float(x0), float(y0), float(x1 - x0), float(y1 - y0), conf)
            history.setdefault(tid, {})[frame_idx] = box

    return history


# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------
def _track_color(track_id):
    rng = np.random.default_rng(track_id * 2654435761 % (2**32))  # deterministic per id
    return tuple(int(v) for v in rng.integers(60, 255, size=3))


# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------
def _track_color(track_id):
    rng = np.random.default_rng(track_id * 2654435761 % (2**32))  # deterministic per id
    return tuple(int(v) for v in rng.integers(60, 255, size=3))


def _bottom_left(box):
    x, y, w, h = box[:4]
    return (x, y + h)


def _draw_trail(vis, points, color, max_radius=6, min_radius=2):
    """
    points: oldest-to-newest list of (x, y) bottom-left corners. Older points
    are drawn smaller and dimmer so the trail visually fades with age, newest
    point drawn largest/brightest.
    """
    n = len(points)
    if n == 0:
        return
    for idx, (px, py) in enumerate(points):
        age_frac = idx / max(n - 1, 1)  # 0 = oldest, 1 = newest
        radius = int(round(min_radius + age_frac * (max_radius - min_radius)))
        fade = 0.35 + 0.65 * age_frac  # oldest point at 35% brightness, newest at 100%
        faded_color = tuple(int(c * fade) for c in color)
        cv2.circle(vis, (int(px), int(py)), radius, faded_color, -1, lineType=cv2.LINE_AA)
        if idx > 0:
            prev_px, prev_py = points[idx - 1]
            cv2.line(vis, (int(prev_px), int(prev_py)), (int(px), int(py)),
                     faded_color, max(1, radius // 2), lineType=cv2.LINE_AA)


def draw_tracks(frames, history, output_dir="tracking_preview", trail_length=10):
    """
    Renders each sampled frame with current boxes + a trail of each track's
    last `trail_length` bottom-left corner positions (fewer if the track is
    younger than that, or has gaps from missed frames). A track still shows
    its trail on a frame where it wasn't detected (e.g. briefly occluded) as
    long as it has at least one point within the trail window, so you can see
    the tracker bridging the gap rather than the trail just vanishing.

    Returns the list of rendered frames (BGR arrays, same order as `frames`)
    in addition to saving them to disk, so compile_animation() can use them
    directly without a re-read from disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    frame_idx_list = [idx for idx, _ in frames]
    rendered = []

    for i, (frame_idx, frame) in enumerate(frames):
        vis = frame.copy()
        window = frame_idx_list[max(0, i - trail_length + 1):i + 1]

        for tid, boxes_by_frame in history.items():
            trail_points = [_bottom_left(boxes_by_frame[fidx]) for fidx in window if fidx in boxes_by_frame]
            if not trail_points:
                continue
            color = _track_color(tid)
            _draw_trail(vis, trail_points, color)

            if frame_idx in boxes_by_frame:
                x, y, w, h, conf = boxes_by_frame[frame_idx]
                cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                label = f"id{tid}"
                cv2.putText(vis, label, (int(x), max(int(y) - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        out_path = os.path.join(output_dir, f"track_frame_{i:02d}_pos{frame_idx}.png")
        cv2.imwrite(out_path, vis)
        print(f"saved {out_path}")
        rendered.append(vis)

    return rendered


def compile_animation(rendered_frames, output_path="tracking_animation.gif", fps=3):
    """
    Compiles the annotated frames into a single animation you can actually
    watch/share, rather than flipping through stills one by one. Defaults to
    GIF (via imageio) since it's viewable without a video player; pass an
    output_path ending in .mp4 for a video file instead (via cv2.VideoWriter).

    fps here is a playback rate choice, not the original video's fps -- these
    are sparsely sampled frames (e.g. every 12th frame), so there's no single
    "correct" playback speed; a low fps (2-4) makes the trails easier to
    follow frame by frame.
    """
    if output_path.lower().endswith(".gif"):
        import imageio
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in rendered_frames]
        imageio.mimsave(output_path, rgb_frames, fps=fps)
    else:
        h, w = rendered_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for f in rendered_frames:
            writer.write(f)
        writer.release()

    print(f"saved animation to {output_path}")


# ----------------------------------------------------------------------------
# Setup + entry point
# ----------------------------------------------------------------------------
def setup_pipeline(cfg, frame_hw):
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
    return ref[:, :, :3], valid_mask, detector, shadow_checker, size_filter


if __name__ == "__main__":
    cfg = load_config("config.yaml")

    FRAME_STRIDE = 12   # -> 5fps at a 60fps source; tune to taste
    N_FRAMES = 10
    # "bytetrack" (recommended, needs `pip install trackers`) or "custom"
    # (the hand-rolled RANSAC+Hungarian tracker earlier in this file)
    TRACKING_BACKEND = "bytetrack"

    frames = sample_frames(cfg.paths.video, frame_stride=FRAME_STRIDE, n_frames=N_FRAMES)
    print(f"sampled {len(frames)} frames")

    frame_hw = frames[0][1].shape[:2]
    ref_bgr, valid_mask, detector, shadow_checker, size_filter = setup_pipeline(cfg, frame_hw)

    frames_boxes = detect_boxes_per_frame(frames, detector, valid_mask, shadow_checker, ref_bgr, size_filter)
    frame_indices = [idx for idx, _ in frames]

    if TRACKING_BACKEND == "custom":
        cap = cv2.VideoCapture(cfg.paths.video)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        effective_fps = max(int(round(source_fps / FRAME_STRIDE)), 1)  # your sampled sequence's own rate
        history = build_tracks_with_bytetrack(frames_boxes, frame_indices, frame_rate=effective_fps)
    else:
        builder = SimpleTrackBuilder(max_missed=2)
        builder.process(frames_boxes, frame_indices, ransac_iters=300, ransac_inlier_thresh=25.0,
                         w_dist=0.5, w_iou=0.5, dist_norm=150.0, reject_cost=0.85)
        history = builder.history

    print(f"built {len(history)} tracks across {len(frames)} frames")
    rendered = draw_tracks(frames, history, trail_length=10)
    compile_animation(rendered, output_path="tracking_animation.gif", fps=3)