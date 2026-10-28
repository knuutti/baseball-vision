"""
Maps player bounding boxes from image pixels onto the field plane (bird's-eye
view), using the planar homography from the field calibration.

Calibration: field markings were picked by hand on a frame from the same
fixed camera (see notebooks/field_analysis.ipynb) and paired with their field
coordinates. All of them lie on the ground plane (z = 0), so a 3x3 homography
fully describes the image -> field mapping. cv2.findHomography with method=0
solves the same normalized least-squares DLT as calibrate_planar_norm() in
the notebook.

Each box is reduced to its feet point (bottom-center of the box) before
mapping -- the only point of a standing player that actually lies on the
ground plane. Mapping the box center instead would project the torso onto
the ground and push far-field players further away than they are.

Output: one image with the camera frame (boxes) on the left and the
bird's-eye field (matching colored dots) on the right.

Run (needs the source video and the Parquet output of
video_bbox_extraction.py for the same video):
    python src/field_mapping.py --video path/to/match.mp4 --frame-idx 3535
"""

import argparse
import colorsys

import cv2
import numpy as np
import pandas as pd

from player_detection_pipeline import resize_to_match


def load_homography(image_points_csv, field_points_csv, calib_hw, frame_hw):
    """
    Returns H (3x3) mapping frame pixels -> field (x, y).

    calib_hw: (height, width) of the image the calibration points were
        picked on. Image points are rescaled to frame_hw first, since the
        reference images are a slightly different export resolution than
        the video frames (1917x1078 vs 1920x1080).
    """
    img_pts = pd.read_csv(image_points_csv)[["x", "y"]].values.astype(np.float64)
    field_pts = pd.read_csv(field_points_csv)[["x", "y"]].values.astype(np.float64)
    img_pts[:, 0] *= frame_hw[1] / calib_hw[1]
    img_pts[:, 1] *= frame_hw[0] / calib_hw[0]
    H, _ = cv2.findHomography(img_pts, field_pts, method=0)
    return H


def feet_points(boxes):
    """boxes: [(x, y, w, h, ...), ...] -> (N, 2) array of bottom-center points."""
    return np.array([[x + w / 2, y + h] for (x, y, w, h, *_rest) in boxes], dtype=np.float64).reshape(-1, 2)


def image_to_field(H, points_px):
    """(N, 2) pixel points -> (N, 2) field coordinates."""
    if len(points_px) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(points_px[None], H)[0]


def field_to_canvas_matrix(extent_x, extent_y, pixels_per_unit):
    """
    Affine field -> canvas map. Field y increases away from the camera, so it
    is flipped to put the far end of the field at the top of the canvas.
    """
    return np.array([
        [pixels_per_unit, 0, -extent_x[0] * pixels_per_unit],
        [0, -pixels_per_unit, extent_y[1] * pixels_per_unit],
        [0, 0, 1],
    ], dtype=np.float64)


def render_birds_eye(field_bgr, H, extent_x, extent_y, pixels_per_unit, grid_step=10):
    """Warps the camera image into field coordinates and draws a labelled grid."""
    S = field_to_canvas_matrix(extent_x, extent_y, pixels_per_unit)
    size = (int((extent_x[1] - extent_x[0]) * pixels_per_unit),
            int((extent_y[1] - extent_y[0]) * pixels_per_unit))
    canvas = cv2.warpPerspective(field_bgr, S @ H, size)

    grid_color = (235, 235, 235)
    for gx in range(int(np.ceil(extent_x[0] / grid_step)) * grid_step, int(extent_x[1]) + 1, grid_step):
        cx = int(round((gx - extent_x[0]) * pixels_per_unit))
        cv2.line(canvas, (cx, 0), (cx, size[1]), grid_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(gx), (cx + 3, size[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_color, 1, cv2.LINE_AA)
    for gy in range(int(np.ceil(extent_y[0] / grid_step)) * grid_step, int(extent_y[1]) + 1, grid_step):
        cy = int(round((extent_y[1] - gy) * pixels_per_unit))
        cv2.line(canvas, (0, cy), (size[0], cy), grid_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(gy), (3, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_color, 1, cv2.LINE_AA)
    return canvas, S


def _distinct_colors(n):
    # golden-ratio hue steps -> neighboring indices get well-separated colors
    return [tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.85, 1.0))[::-1]
            for i in range(n)]


def draw_mapping(frame_bgr, field_bgr, boxes, H, extent_x, extent_y, pixels_per_unit):
    """
    Returns (side_by_side_bgr, field_xy): the frame with boxes and feet points
    next to the bird's-eye canvas with one matching dot per box, plus the
    (N, 2) field coordinates of the boxes.
    """
    feet = feet_points(boxes)
    field_xy = image_to_field(H, feet)
    colors = _distinct_colors(len(boxes))

    vis = frame_bgr.copy()
    for (x, y, w, h, *_rest), (fx, fy), color in zip(boxes, feet, colors):
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        cv2.circle(vis, (int(fx), int(fy)), 4, color, -1, cv2.LINE_AA)

    canvas, S = render_birds_eye(field_bgr, H, extent_x, extent_y, pixels_per_unit)
    canvas_pts = image_to_field(S, field_xy)  # same projective transform helper, S is affine
    for (cx, cy), color in zip(canvas_pts, colors):
        if 0 <= cx < canvas.shape[1] and 0 <= cy < canvas.shape[0]:
            cv2.circle(canvas, (int(cx), int(cy)), 7, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(cx), int(cy)), 5, color, -1, cv2.LINE_AA)

    # scale the camera frame to the canvas height and place them side by side
    scale = canvas.shape[0] / vis.shape[0]
    vis = cv2.resize(vis, (int(vis.shape[1] * scale), canvas.shape[0]), interpolation=cv2.INTER_AREA)
    gap = np.full((canvas.shape[0], 12, 3), 255, dtype=np.uint8)
    body = np.hstack([vis, gap, canvas])

    header = np.full((44, body.shape[1], 3), 255, dtype=np.uint8)
    for text, x0 in [("Camera frame: YOLO boxes and feet points", 8),
                     ("Field plane (bird's-eye)", vis.shape[1] + gap.shape[1] + 8)]:
        cv2.putText(header, text, (x0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    return np.vstack([header, body]), field_xy


def read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def main():
    from config import load_config

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video", default=None, help="overrides paths.video from config")
    parser.add_argument("--frame-idx", type=int, required=True,
                        help="a frame_idx present in the detections Parquet file")
    parser.add_argument("--detections", default=None,
                        help="defaults to <video_processing.out_prefix>_detections.parquet")
    parser.add_argument("--out", default="birds_eye_view.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fm = cfg.field_mapping
    video_path = args.video or cfg.paths.video
    detections_path = args.detections or f"{cfg.video_processing.out_prefix}_detections.parquet"

    dets = pd.read_parquet(detections_path)
    dets = dets[dets.frame_idx == args.frame_idx]
    if dets.empty:
        raise ValueError(f"No detections for frame_idx={args.frame_idx} in {detections_path}")
    boxes = list(dets[["x", "y", "w", "h", "confidence"]].itertuples(index=False, name=None))

    frame = read_frame(video_path, args.frame_idx)
    frame_hw = frame.shape[:2]
    # the empty-field reference image (no players) is the bird's-eye background
    field_bgr = resize_to_match(cv2.imread(cfg.paths.empty_field, cv2.IMREAD_COLOR), frame_hw)

    H = load_homography(fm.image_points, fm.field_points, tuple(fm.calib_image_hw), frame_hw)
    out, field_xy = draw_mapping(frame, field_bgr, boxes, H, fm.extent_x, fm.extent_y, fm.pixels_per_unit)
    cv2.imwrite(args.out, out)

    inside = ((field_xy[:, 0] >= fm.extent_x[0]) & (field_xy[:, 0] <= fm.extent_x[1])
              & (field_xy[:, 1] >= fm.extent_y[0]) & (field_xy[:, 1] <= fm.extent_y[1]))
    print(f"mapped {len(boxes)} boxes, {int(inside.sum())} inside the canvas extent")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
