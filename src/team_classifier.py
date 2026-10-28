"""
Team / role classification via jersey color.

Approach:
  1. Sample color from the torso region of each detection, restricted to real
     foreground pixels (reusing ForegroundDetector's shadow/background test)
     so background clay bleeding in at the box edges doesn't wash out the
     result.
  2. Represent color as a (chroma_a, chroma_b, value) triple:
         chroma_a = saturation * cos(hue), chroma_b = saturation * sin(hue)
     This is a polar-to-cartesian re-parameterization of HSV's hue+saturation.
     It is deliberately NOT plain (hue, saturation): raw hue is undefined/
     noisy for near-black or near-white fabric (saturation ~0), and a hard
     "exclude low-saturation pixels" filter (an earlier version of this code)
     ends up discarding almost an entire black jersey, leaving only stray
     background-colored pixels as the "reliable" sample -- verified against
     real data, a black kit came out reading nearly the same hue as this
     field's clay background. The chroma vector fixes this: low-saturation
     pixels naturally sit near the origin (small radius) regardless of their
     noisy hue angle, rather than being excluded and replaced by noise.
     Value is kept as a separate third dimension specifically so "black kit"
     (near-origin chroma, low value) and "white/gray staff shirt" (near-origin
     chroma, high value) remain distinguishable from each other.
  3. Classify by nearest reference "prototype" (one per role: team A, team B,
     referee, staff, ...) in this 3D space, rather than blind k-means
     clustering -- more robust when role classes are very unbalanced in count
     (few referees/staff vs many players).
  4. Aggregate per TRACK (not per frame): median chroma_a/b/value over the
     track's lifetime, classify once. Smooths over single-frame lighting
     anomalies (e.g. one backlit frame reading desaturated/gray).

Usage sketch:
    prototypes = build_prototypes_from_crops({
        "team_a": [crop1, crop2, ...],   # a few sample images/crops per role
        "team_b": [crop1, crop2, ...],
        "referee": [crop1, ...],
        "staff": [crop1, ...],
    })
    classifier = ColorRoleClassifier(prototypes)

    # per frame, per detection:
    sample = extract_jersey_color(frame_bgr, bg_bgr, box, fg_detector)
    if sample is not None:
        track_color_history[track_id].append(sample)

    # once per track (e.g. at the end, or after N samples collected):
    role, confident = classifier.classify_track(track_color_history[track_id])
"""

import cv2
import numpy as np


def _hsv_to_chroma(h_deg_0_180, s, v):
    """(hue 0-180, sat 0-255, val 0-255) -> (chroma_a, chroma_b, value)."""
    theta = h_deg_0_180 * (2 * np.pi / 180.0)
    a = s * np.cos(theta)
    b = s * np.sin(theta)
    return a, b, v


# ----------------------------------------------------------------------------
# Per-detection color extraction
# ----------------------------------------------------------------------------
def extract_jersey_color(frame_bgr, bg_bgr, box, fg_detector,
                          torso_top=0.15, torso_bottom=0.55, min_pixels=20):
    """
    Extracts a single (chroma_a, chroma_b, value) sample representing jersey
    color for one detection box, or None if there aren't enough clean
    foreground pixels to trust.

    torso_top/torso_bottom: fraction of box height defining the torso band
        (skips the head near the top and legs/shorts-color-mixing near the
        bottom -- tune these if the camera angle crops players differently).

    Unlike an earlier version of this function, low-saturation pixels are NOT
    excluded -- they're exactly what should register as "near black/white/gray"
    in chroma space. See module docstring.
    """
    x, y, w, h = box
    ty0 = y + int(h * torso_top)
    ty1 = y + int(h * torso_bottom)
    tx0, tx1 = x, x + w

    ty0, ty1 = max(ty0, 0), min(ty1, frame_bgr.shape[0])
    tx0, tx1 = max(tx0, 0), min(tx1, frame_bgr.shape[1])
    if ty1 <= ty0 or tx1 <= tx0:
        return None

    crop_frame = frame_bgr[ty0:ty1, tx0:tx1]
    crop_bg = bg_bgr[ty0:ty1, tx0:tx1]
    crop_valid = np.ones(crop_frame.shape[:2], dtype=bool)

    foreground, _, _ = fg_detector.detect_detailed(crop_frame, crop_bg, crop_valid)
    if foreground.sum() < min_pixels:
        return None  # not enough clean signal -- caller should skip this sample

    hsv = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2HSV)
    h_vals = hsv[..., 0][foreground].astype(np.float32)
    s_vals = hsv[..., 1][foreground].astype(np.float32)
    v_vals = hsv[..., 2][foreground].astype(np.float32)

    a_vals, b_vals = _hsv_to_chroma(h_vals, s_vals, 0)[0:2]
    return float(np.median(a_vals)), float(np.median(b_vals)), float(np.median(v_vals))


def _chroma_distance(sample1, sample2, chroma_weight=1.0, value_weight=0.5):
    a1, b1, v1 = sample1
    a2, b2, v2 = sample2
    chroma_dist = np.hypot(a1 - a2, b1 - b2)
    value_dist = abs(v1 - v2)
    return chroma_weight * chroma_dist + value_weight * value_dist


# ----------------------------------------------------------------------------
# Prototype construction from a handful of labeled example crops
# ----------------------------------------------------------------------------
def build_prototypes_from_crops(role_to_crops_bgr):
    """
    role_to_crops_bgr: {role_name: [bgr_image_crop, ...]}
        Crop these by hand from a clean frame -- e.g. a tight box around one
        player's torso per role. A handful of crops per role (even 2-3) is
        usually enough since jersey colors are typically very consistent
        within a role. Include a black-kit crop as raw pixels (no need to
        pre-filter) -- the chroma representation handles it correctly now.

    Returns: {role_name: (chroma_a, chroma_b, value)} prototype dict for
    ColorRoleClassifier.
    """
    prototypes = {}
    for role, crops in role_to_crops_bgr.items():
        all_a, all_b, all_v = [], [], []
        for crop in crops:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            h_vals = hsv[..., 0].reshape(-1).astype(np.float32)
            s_vals = hsv[..., 1].reshape(-1).astype(np.float32)
            v_vals = hsv[..., 2].reshape(-1).astype(np.float32)
            a, b = _hsv_to_chroma(h_vals, s_vals, 0)[0:2]
            all_a.append(a)
            all_b.append(b)
            all_v.append(v_vals)
        prototypes[role] = (
            float(np.median(np.concatenate(all_a))),
            float(np.median(np.concatenate(all_b))),
            float(np.median(np.concatenate(all_v))),
        )
    return prototypes


# ----------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------
class ColorRoleClassifier:
    def __init__(self, prototypes, chroma_weight=1.0, value_weight=0.5, min_confidence_margin=15.0):
        """
        prototypes: {role_name: (chroma_a, chroma_b, value)}, from
            build_prototypes_from_crops()
        chroma_weight/value_weight: relative weighting in the distance metric.
        min_confidence_margin: if the best and second-best role are within
            this distance of each other, classify_track() flags low confidence
            rather than silently picking one -- useful for surfacing cases
            worth a manual look (e.g. a very desaturated/shadowed track, or
            two roles with genuinely similar colors).
        """
        self.prototypes = prototypes
        self.chroma_weight = chroma_weight
        self.value_weight = value_weight
        self.min_confidence_margin = min_confidence_margin

    def classify_track(self, samples):
        """
        samples: list of (chroma_a, chroma_b, value) samples collected for one
        track across multiple frames (from extract_jersey_color(), skip Nones).

        Returns: (role_name_or_None, confident_bool)
        role is None if there weren't enough samples to judge.
        """
        if len(samples) == 0:
            return None, False

        arr = np.array(samples)
        agg = tuple(np.median(arr, axis=0))

        distances = {
            role: _chroma_distance(agg, proto, self.chroma_weight, self.value_weight)
            for role, proto in self.prototypes.items()
        }
        ranked = sorted(distances.items(), key=lambda kv: kv[1])
        best_role, best_dist = ranked[0]
        second_dist = ranked[1][1] if len(ranked) > 1 else float("inf")

        confident = (second_dist - best_dist) >= self.min_confidence_margin
        return best_role, confident


# ----------------------------------------------------------------------------
# Semi-supervised workflow: build prototypes (or a k-NN model) from a labeled
# dataset produced by labeling_tool.py, instead of a few hand-picked crops.
# ----------------------------------------------------------------------------
def build_prototypes_from_labeled_samples(labeled_samples):
    """
    labeled_samples: list of (role, (chroma_a, chroma_b, value)) tuples, e.g.
        loaded from labeled_samples.json produced by labeling_tool.py.

    Returns: {role: (chroma_a, chroma_b, value)} prototype dict, same shape as
    build_prototypes_from_crops() -- drop-in replacement for ColorRoleClassifier.
    More labeled samples per role = a more robust centroid, and unlike hand
    picking 2-3 crops, this naturally averages over the lighting variation
    the match actually contains (backlit, side-lit, partial shade, etc.).
    """
    by_role = {}
    for role, features in labeled_samples:
        by_role.setdefault(role, []).append(features)

    prototypes = {}
    for role, feats in by_role.items():
        arr = np.array(feats)
        prototypes[role] = tuple(np.median(arr, axis=0))
    return prototypes


class KNNRoleClassifier:
    """
    Optional upgrade over ColorRoleClassifier's single-centroid-per-role model.
    Worth switching to this if a role's color isn't a single tight cluster --
    e.g. a keeper/libero wearing a different color than the rest of their team,
    or a role whose samples split into two lighting-driven sub-clusters that a
    single centroid distance can't represent well. Needs scikit-learn.

    Requires meaningfully more labeled data than the centroid approach to be
    reliable (rule of thumb: at least ~15-20 labeled samples per role), since
    it has no notion of "role" beyond the labeled examples it is given -- the
    centroid approach can work with just a handful of hand-picked crops.
    """

    def __init__(self, labeled_samples, n_neighbors=5, confidence_margin=0.2):
        from sklearn.neighbors import KNeighborsClassifier

        self.roles = [role for role, _ in labeled_samples]
        X = np.array([features for _, features in labeled_samples])
        self.model = KNeighborsClassifier(n_neighbors=min(n_neighbors, len(labeled_samples)))
        self.model.fit(X, self.roles)
        self.confidence_margin = confidence_margin

    def classify_track(self, samples):
        if len(samples) == 0:
            return None, False
        agg = np.median(np.array(samples), axis=0).reshape(1, -1)
        probs = self.model.predict_proba(agg)[0]
        ranked_idx = np.argsort(probs)[::-1]
        best_role = self.model.classes_[ranked_idx[0]]
        best_p = probs[ranked_idx[0]]
        second_p = probs[ranked_idx[1]] if len(ranked_idx) > 1 else 0.0
        confident = (best_p - second_p) >= self.confidence_margin
        return best_role, confident
