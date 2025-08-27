# --- UNIFIED MACROPHAGE SCRIPT (hybrid ghost: polygon centroid ⊕ intensity-weighted COM) ---
# Exports EXACT CSV schemas:
#   1) {base}_ghosttracks.csv      columns: track_id,frame,x,y
#   2) {base}_ghost_metrics.csv    columns: frame,time_sec,velocity,velocity_sem,displacement,displacement_sem
#   3) {base}_ghost_angles.csv     columns: angle_rad
#
# Also exports:
#   {base}_tracks.csv
#   {base}_velocity.png, {base}_persistence.png, {base}_acceleration.png, {base}_netdisplacement.png
#   {base}_ghost_velocity.png, {base}_ghost_persistence.png, {base}_ghost_acceleration.png, {base}_ghost_displacement.png
#   {base}_roseplot.png, {base}_ghost_roseplot.png
#   {base}_motion_overlay.png, {base}_ghost_overlay.png
#   {base}_napari_direct.<movie_format_tracks> (optional)
#
# NOTE: Column names *and order* in the CSV exports are enforced explicitly. Do not change.

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, filters, measure, morphology, exposure, feature
from skimage.segmentation import clear_border, watershed
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from tqdm import tqdm
import napari
import tkinter as tk
from tkinter import filedialog
import warnings
from matplotlib import cm
import shutil
import matplotlib as mpl
from skimage.measure import regionprops

warnings.filterwarnings('ignore')

# -----------------------------
# DEFAULT PARAMETERS
# -----------------------------
default_method = "seeded_ws"

# Sizes / cleanup
min_object_size       = 50
max_object_size       = 600
strict_min_area       = 110
use_clear_border      = True

# Enhancement (gentle)
seg_smooth_sigma      = 1.0
unsharp_radius        = 8.0
unsharp_amount        = 20.0

# Hysteresis (if selected)
hyst_low_pct          = 30.0
hyst_high_pct         = 90.0

# Edge-fusion (if selected)
sobel_weight          = 0.80
edge_sigma            = 2.0

# Seeded watershed (DEFAULT)
log_sigma             = 6
log_thresh_rel        = 0.08
ws_enable_extra_split = False
ws_min_distance       = 12
ws_footprint          = 9
ws_compactness        = 0.0
ws_erode_before       = 0
ws_opening            = 0
ws_h_minima           = 0.0
ws_dilate_after       = 0

# Ghost relabeling defaults
search_radius         = 80
max_centroid_distance = 15
iou_threshold         = 0.15
max_ghost_gap         = 8

# HYBRID ghost definition (0 = pure polygon centroid, 1 = pure intensity-weighted COM)
ghost_com_weight      = 0.75  # default; adjustable in Dock widget 2

# Tracking defaults
max_tracking_distance = 90
max_frame_gap         = 5
min_track_length      = 10
min_jump_distance     = 10
max_merge_angle_deg   = 180
max_pred_error        = 30

# Other
frame_interval_sec    = 30
make_movie            = True
fps_movie             = 10
movie_format_tracks   = 'avi'  # 'avi', 'mp4', 'mkv', 'gif'

# --- Safe ffmpeg Setup ---
ffmpeg_path = shutil.which('ffmpeg')
if ffmpeg_path is not None:
    mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path
else:
    print("⚠️  ffmpeg not found. Movie will be written with imageio (format dependent).")

# -----------------------------
# File selection
# -----------------------------
root = tk.Tk(); root.withdraw()
input_path = filedialog.askopenfilename(title="Select your TIFF image")
if not input_path:
    raise SystemExit("No file selected.")
output_dir = os.path.dirname(input_path)
base_filename = os.path.splitext(os.path.basename(input_path))[0]
print(f"Selected file base name: {base_filename}")

# -----------------------------
# Load image (T, C, Y, X)
# -----------------------------
print(f"Loading image: {input_path}")
image = io.imread(input_path)
if image.ndim != 4:
    raise ValueError(f"Expected image shape (T, C, Y, X); got {image.shape}")
frames, channels, height, width = image.shape
print(f"Image shape: {image.shape}")
if channels < 2:
    raise ValueError("This script expects at least two channels: [green, red].")
green_channel = image[:, 0, :, :]
red_channel   = image[:, 1, :, :]

# -----------------------------
# GPU availability
# -----------------------------
try:
    import pyclesperanto_prototype as cle
    cle.select_device("auto")
    print(f"⚡ Using GPU for segmentation: {cle.get_device()}")
    CLE_AVAILABLE = True
except Exception:
    print("⚠️  pyclesperanto not found or no compatible GPU, using CPU for segmentation.")
    CLE_AVAILABLE = False

try:
    import cupy as cp
    _ = cp.zeros((1,), dtype=cp.float32)
    CUPY_AVAILABLE = True
    print("⚡ CuPy available for GPU tracking.")
except Exception:
    CUPY_AVAILABLE = False
    print("ℹ️  CuPy not available. Tracking will run on CPU.")

# -----------------------------
# Preprocessing
# -----------------------------
print("Preprocessing frames (GPU/CPU)…")
green_normalized = np.zeros_like(green_channel, dtype=np.float32)  # enhanced
green_minmax     = np.zeros_like(green_channel, dtype=np.float32)  # per-frame min–max
red_normalized   = np.zeros_like(red_channel,   dtype=np.float32)

def _gpu_preprocess(img):
    img_gpu  = cle.push(img.astype(np.float32))
    bg_sub   = cle.top_hat_box(img_gpu, radius_x=15, radius_y=15)
    smoothed = cle.gaussian_blur(bg_sub, sigma_x=1.5, sigma_y=1.5)
    arr      = cle.pull(smoothed)
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr_norm = arr_norm ** 0.8  # gamma boost
    return arr_norm.astype(np.float32)

for i in tqdm(range(frames), desc="Preprocess", colour="white"):
    img = green_channel[i]
    try:
        if CLE_AVAILABLE:
            green_normalized[i] = _gpu_preprocess(img)
        else:
            raise RuntimeError("Force CPU")
    except Exception:
        background          = filters.gaussian(img, sigma=10)
        background_smoothed = morphology.opening(background, morphology.disk(15))
        img_subtracted      = np.clip(img - background_smoothed, 0, None)
        img_subtracted_norm = exposure.rescale_intensity(img_subtracted, in_range='image', out_range=(0, 1))
        img_eq              = exposure.equalize_adapthist(img_subtracted_norm, clip_limit=0.008)
        img_final           = filters.gaussian(img_eq, sigma=0.5)
        img_final           = exposure.rescale_intensity(img_final, in_range='image', out_range=(0, 1))
        green_normalized[i] = img_final

    g = img.astype(np.float32)
    mn, mx = float(g.min()), float(g.max())
    green_minmax[i] = (g - mn) / (mx - mn + 1e-8)
    red_normalized[i] = exposure.rescale_intensity(red_channel[i], in_range='image', out_range=(0, 1))

# -----------------------------
# Helpers
# -----------------------------
def _ensure_odd(v: int) -> int:
    v = int(v)
    return v if v % 2 == 1 else v + 1

def _apply_ws_split(binary, radius_erode=0, opening_disk=0, min_distance=8,
                    footprint_size=5, compactness=0.0, h_min=0.0, dilate_after=0):
    mask = binary.astype(bool)
    if radius_erode > 0:
        mask = morphology.erosion(mask, morphology.disk(int(radius_erode)))
    if opening_disk > 0:
        mask = morphology.opening(mask, morphology.disk(int(opening_disk)))

    dist = ndi.distance_transform_edt(mask)
    if h_min and h_min > 0:
        from skimage.morphology import h_minima
        dist = -dist
        dist = -h_minima(dist, h=h_min)
        dist = -dist

    fp = np.ones((_ensure_odd(footprint_size), _ensure_odd(footprint_size)), dtype=bool)
    coords = feature.peak_local_max(dist, footprint=fp, labels=mask,
                                    exclude_border=False, min_distance=int(min_distance))
    markers = np.zeros_like(mask, dtype=int)
    if coords.size:
        markers[tuple(coords.T)] = np.arange(1, coords.shape[0] + 1)
    markers = measure.label(markers)

    labels = watershed(-dist, markers=markers, mask=mask, compactness=float(ws_compactness))
    if dilate_after > 0:
        labels = morphology.dilation(labels, morphology.disk(int(dilate_after)))
    return labels

def length_filter(tracks, min_len: int):
    m = int(max(1, min_len))
    return [t for t in tracks if len(t) >= m]

def filter_tracks_displacement(
    tracks,
    frame_interval_sec=30,
    max_allowed_velocity=8.0,     # pixels/sec
    min_total_displacement=5.0,   # pixels
    max_stall_duration=6,         # consecutive ~zero steps
    min_total_duration=5          # frames
):
    kept = []
    for tr in tracks:
        if len(tr) < int(min_total_duration):
            continue
        coords = np.array([[x, y] for _, x, y in tr], dtype=float)
        if coords.shape[0] < 2:
            continue
        step_d = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        mean_velocity = float(np.mean(step_d)) / float(frame_interval_sec)
        net_disp = float(np.linalg.norm(coords[-1] - coords[0]))
        stall_run = 0
        stalled = False
        for d in step_d:
            if d < 1e-2:
                stall_run += 1
                if stall_run >= int(max_stall_duration):
                    stalled = True
                    break
            else:
                stall_run = 0
        if (mean_velocity <= float(max_allowed_velocity)
            and net_disp >= float(min_total_displacement)
            and not stalled):
            kept.append(tr)
    return kept

# -----------------------------
# Segmentation core
# -----------------------------
def segment_frames(
    gstack,
    method="seeded_ws",
    # Shared clean-up
    min_object_size=50,
    max_object_size=600,
    strict_min_area=110,
    use_clear_border=True,
    # Enhancement
    seg_smooth_sigma=1.0,
    unsharp_radius=2,
    unsharp_amount=2.0,
    # Hysteresis params
    low_pct=30.0,
    high_pct=85.0,
    # Edge-fusion params
    sobel_weight=0.35,
    edge_sigma=1.0,
    # Seeded watershed params
    log_sigma=1.5,
    log_thresh_rel=0.03,
    ws_min_distance=8,
    ws_footprint=5,
    ws_compactness=0.0,
    ws_erode_before=0,
    ws_opening=0,
    ws_h_minima=0.0,
    ws_dilate_after=0,
    # GPU segmentation toggle
    use_gpu_seg=False,
):
    per_frame_labeled_masks = []
    frame_objects = {}
    frame_objects_com = {}

    def _gpu_seeded_labels(img_f32, spot_sigma, outline_sigma):
        img_gpu = cle.push(img_f32)
        labels_gpu = cle.voronoi_otsu_labeling(
            img_gpu,
            spot_sigma=float(max(0.5, spot_sigma)),
            outline_sigma=float(max(0.5, outline_sigma))
        )
        labels = cle.pull(labels_gpu).astype(np.int32)
        return labels

    for i in tqdm(range(gstack.shape[0]), desc=f"Seg ({method})", colour="magenta"):
        frame = gstack[i]
        work = frame.copy()
        if seg_smooth_sigma > 0:
            work = filters.gaussian(work, sigma=float(seg_smooth_sigma))
        if unsharp_amount > 0 and unsharp_radius > 0:
            work = filters.unsharp_mask(work, radius=int(unsharp_radius), amount=float(unsharp_amount))
        work = np.clip(work, 0, 1)

        labels = None
        if method == "seeded_ws" and use_gpu_seg and CLE_AVAILABLE:
            try:
                labels = _gpu_seeded_labels(
                    work.astype(np.float32),
                    spot_sigma=float(log_sigma * 1.2),
                    outline_sigma=float(max(1.0, seg_smooth_sigma))
                )
                binary = labels > 0
            except Exception:
                print("⚠️  GPU segmentation failed for this frame; fallback to CPU.")
                labels = None

        if labels is None:
            if method == "hysteresis":
                vals = work[work > 0]
                if vals.size == 0:
                    binary = np.zeros_like(work, dtype=bool)
                else:
                    lo = np.percentile(vals, np.clip(low_pct, 0, 100))
                    hi = np.percentile(vals, np.clip(high_pct, 0, 100))
                    if hi <= lo:
                        hi = lo + 1e-6
                    binary = filters.apply_hysteresis_threshold(work, lo, hi)

            elif method == "edgefusion":
                sob = filters.sobel(work)
                if edge_sigma > 0:
                    sob = filters.gaussian(sob, sigma=float(edge_sigma))
                sob = (sob - sob.min()) / (sob.max() - sob.min() + 1e-8)
                fused = (1.0 - float(sobel_weight)) * work + float(sobel_weight) * sob
                thr = filters.threshold_otsu(fused) if fused.max() > 0 else 1.0
                binary = fused > thr

            elif method == "seeded_ws":
                base_thr = filters.threshold_otsu(work) if work.max() > 0 else 1.0
                mask0 = work > base_thr
                if ws_erode_before > 0:
                    mask0 = morphology.erosion(mask0, morphology.disk(int(ws_erode_before)))
                if ws_opening > 0:
                    mask0 = morphology.opening(mask0, morphology.disk(int(ws_opening)))
                dist = ndi.distance_transform_edt(mask0)
                fp = np.ones((_ensure_odd(ws_footprint), _ensure_odd(ws_footprint)), dtype=bool)
                coords = feature.peak_local_max(
                    dist, footprint=fp, labels=mask0, exclude_border=False, min_distance=int(ws_min_distance)
                )
                markers = np.zeros_like(mask0, dtype=int)
                if coords.size:
                    markers[tuple(coords.T)] = np.arange(1, coords.shape[0] + 1)
                markers = measure.label(markers)
                labels = watershed(-dist, markers=markers, mask=mask0, compactness=float(ws_compactness))
                if ws_h_minima and ws_h_minima > 0:
                    from skimage.morphology import h_minima
                    d2 = -dist
                    d2 = -h_minima(d2, h=float(ws_h_minima))
                    labels = watershed(d2, markers=markers, mask=mask0, compactness=float(ws_compactness))
                if ws_dilate_after > 0:
                    labels = morphology.dilation(labels, morphology.disk(int(ws_dilate_after)))
                binary = labels > 0
            else:
                raise ValueError(f"Unknown method: {method}")

        # Clean-up
        if min_object_size > 0:
            binary = morphology.remove_small_objects(binary.astype(bool), min_size=int(min_object_size))
        if use_clear_border:
            binary = clear_border(binary)
        labeled = measure.label(binary)

        if strict_min_area > 0:
            props_tmp = measure.regionprops(labeled)
            keep_labels = [p.label for p in props_tmp if p.area >= int(strict_min_area)]
            filtered_mask = np.isin(labeled, keep_labels)
            labeled = measure.label(filtered_mask)

        if method == "seeded_ws" and ws_enable_extra_split:
            labeled = _apply_ws_split(
                labeled > 0,
                radius_erode=int(ws_erode_before),
                opening_disk=int(ws_opening),
                min_distance=int(ws_min_distance),
                footprint_size=int(ws_footprint),
                compactness=float(ws_compactness),
                h_min=float(ws_h_minima),
                dilate_after=int(ws_dilate_after),
            )

        per_frame_labeled_masks.append(labeled.astype(np.int32))

        # Collect centroids + intensity-weighted centers
        objs_centroid, objs_com = [], []
        props = measure.regionprops(labeled, intensity_image=gstack[i])
        for p in props:
            if min_object_size <= p.area <= max_object_size:
                y, x = p.centroid
                objs_centroid.append((x, y))
                if hasattr(p, "weighted_centroid") and p.weighted_centroid is not None:
                    wy, wx = p.weighted_centroid
                else:
                    wy, wx = y, x
                objs_com.append((float(wx), float(wy)))
        frame_objects[i] = np.array(objs_centroid)
        frame_objects_com[i] = np.array(objs_com)

    raw_label_stack = np.stack(per_frame_labeled_masks, axis=0)
    return per_frame_labeled_masks, frame_objects, raw_label_stack, frame_objects_com

# --- Ghost relabeling (IoU + HYBRID distance) ---
def _iou_boolean_masks(prev_mask, curr_mask, use_gpu=False):
    if not use_gpu or not CUPY_AVAILABLE:
        overlap = np.logical_and(prev_mask, curr_mask).sum()
        union   = np.logical_or(prev_mask,  curr_mask).sum()
        return (overlap / union) if union > 0 else 0.0
    pm = cp.asarray(prev_mask)
    cm = cp.asarray(curr_mask)
    overlap = cp.logical_and(pm, cm).sum()
    union   = cp.logical_or(pm,  cm).sum()
    overlap = float(overlap.get())
    union   = float(union.get())
    return (overlap / union) if union > 0 else 0.0

def ghost_label_tracking(per_frame_labeled_masks,
                         intensity_stack,
                         iou_threshold=0.40,
                         max_centroid_distance=30,
                         max_ghost_gap=8,
                         search_radius=25,
                         use_gpu=False,
                         com_weight=0.60):
    """
    Hybrid-ghost relabeling:
    - IoU on masks
    - Distance term uses a hybrid point H = (1-w)*centroid + w*weighted_centroid
    """
    T = len(per_frame_labeled_masks)
    H, W = per_frame_labeled_masks[0].shape
    segmentation_labels = np.zeros((T, H, W), dtype=np.uint16)

    active_tracks = {}   # tid -> dict(mask, last_frame, centroid, wcentroid, hcentroid)
    next_label_id = 1
    max_dist = float(max_centroid_distance)
    gpu_ok = use_gpu and CUPY_AVAILABLE
    w = float(np.clip(com_weight, 0.0, 1.0))

    for f in tqdm(range(T), desc="Relabel (IoU-hybrid)", colour="cyan"):
        current_mask = per_frame_labeled_masks[f]
        current_props = regionprops(current_mask, intensity_image=intensity_stack[f])

        matched = set()
        used_ids = set()
        label_assignment = {}

        if f == 0:
            for region in current_props:
                m = (current_mask == region.label)
                cy, cx = region.centroid
                if region.weighted_centroid is not None:
                    wcy, wcx = region.weighted_centroid
                else:
                    wcy, wcx = cy, cx
                hcx = (1.0 - w) * cx + w * wcx
                hcy = (1.0 - w) * cy + w * wcy

                segmentation_labels[f][m] = next_label_id
                active_tracks[next_label_id] = {
                    'centroid':   (cy, cx),
                    'wcentroid':  (wcy, wcx),
                    'hcentroid':  (hcy, hcx),
                    'mask': m,
                    'last_frame': f
                }
                next_label_id += 1
            continue

        ghost_candidates = {
            tid: tr for tid, tr in active_tracks.items()
            if f - tr['last_frame'] <= int(max_ghost_gap)
        }
        current_label_masks = {region.label: (current_mask == region.label) for region in current_props}

        for region in current_props:
            cy, cx = region.centroid
            if region.weighted_centroid is not None:
                wcy, wcx = region.weighted_centroid
            else:
                wcy, wcx = cy, cx
            hy = (1.0 - w) * cy + w * wcy
            hx = (1.0 - w) * cx + w * wcx

            best_tid, best_score = None, -1.0
            for tid, tr in ghost_candidates.items():
                if tid in used_ids:
                    continue
                pprev = np.array(tr.get('hcentroid', tr.get('wcentroid', tr['centroid'])), dtype=float)[[0,1]]
                dist = float(np.linalg.norm(np.array([hy, hx], dtype=float) - pprev))
                if dist > max_dist:
                    continue

                prev_mask = tr['mask']
                curr_mask = current_label_masks[region.label]
                iou = _iou_boolean_masks(prev_mask, curr_mask, use_gpu=gpu_ok)

                # require decent IoU OR very close hybrid distance
                if iou < float(iou_threshold) and dist > (0.5 * max_dist):
                    continue

                norm_dist = 1.0 - min(dist, max_dist) / max_dist
                score = 0.65 * iou + 0.35 * norm_dist
                if score > best_score:
                    best_tid, best_score = tid, score

            if best_tid is not None:
                label_assignment[region.label] = best_tid
                m = current_label_masks[region.label]
                active_tracks[best_tid] = {
                    'centroid':   (cy, cx),
                    'wcentroid':  (wcy, wcx),
                    'hcentroid':  (hy, hx),
                    'mask': m,
                    'last_frame': f
                }
                matched.add(region.label)
                used_ids.add(best_tid)

        # assign new/fallback ids
        for region in current_props:
            m = (current_mask == region.label)
            if region.label in matched:
                assigned_id = label_assignment[region.label]
                segmentation_labels[f][m] = assigned_id
            else:
                fallback_label = None
                if f > 0:
                    prev_labels = segmentation_labels[f - 1]
                    y, x = map(int, region.centroid)
                    r = int(max(1, search_radius))
                    y_min, y_max = max(0, y - r), min(H, y + r + 1)
                    x_min, x_max = max(0, x - r), min(W, x + r + 1)
                    local_prev = prev_labels[y_min:y_max, x_min:x_max]
                    candidate_labels, counts = np.unique(local_prev[local_prev > 0], return_counts=True)
                    if len(candidate_labels) > 0:
                        order = np.argsort(-counts)
                        for idx in order:
                            cand = int(candidate_labels[idx])
                            if cand not in used_ids:
                                fallback_label = cand
                                break
                if fallback_label is None:
                    fallback_label = next_label_id
                    next_label_id += 1

                cy, cx = region.centroid
                if region.weighted_centroid is not None:
                    wcy, wcx = region.weighted_centroid
                else:
                    wcy, wcx = cy, cx
                hy = (1.0 - w) * cy + w * wcy
                hx = (1.0 - w) * cx + w * wcx

                segmentation_labels[f][m] = fallback_label
                active_tracks[fallback_label] = {
                    'centroid':   (cy, cx),
                    'wcentroid':  (wcy, wcx),
                    'hcentroid':  (hy, hx),
                    'mask': m,
                    'last_frame': f
                }
                used_ids.add(fallback_label)

    return segmentation_labels

def _angle_between(v1, v2):
    v1 = np.asarray(v1); v2 = np.asarray(v2)
    n1 = np.linalg.norm(v1) + 1e-6
    n2 = np.linalg.norm(v2) + 1e-6
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))

# --- CPU tracker ---
def link_tracks_cpu(frame_objects, width, height,
                    max_frame_gap=5, max_tracking_distance=90,
                    min_jump_distance=10, max_merge_angle_deg=180,
                    max_pred_error=30, border_margin=5, min_track_length=1):
    active_tracks = {}
    last_seen = {}
    next_track_id = 0

    for (x, y) in frame_objects.get(0, []):
        if border_margin < x < width - border_margin and border_margin < y < height - border_margin:
            active_tracks[next_track_id] = [(0, x, y)]
            last_seen[next_track_id] = 0
            next_track_id += 1

    for f in tqdm(range(1, len(frame_objects)), desc="Track (CPU)", colour="cyan"):
        detections = frame_objects.get(f, np.array([]))
        if len(detections) == 0:
            continue

        active_positions, track_ids, track_last_frames, track_velocities = [], [], [], []
        for tid, tr in active_tracks.items():
            if f - last_seen[tid] <= int(max_frame_gap):
                if len(tr) >= 2:
                    dx = tr[-1][1] - tr[-2][1]
                    dy = tr[-1][2] - tr[-2][2]
                    track_velocities.append((dx, dy))
                else:
                    track_velocities.append((0.0, 0.0))
                active_positions.append(tr[-1][1:])
                track_ids.append(tid)
                track_last_frames.append(last_seen[tid])

        if len(active_positions) == 0:
            for det in detections:
                if border_margin < det[0] < width - border_margin and border_margin < det[1] < height - border_margin:
                    active_tracks[next_track_id] = [(f, det[0], det[1])]
                    last_seen[next_track_id] = f
                    next_track_id += 1
            continue

        active_positions = np.array(active_positions)
        track_last_frames = np.array(track_last_frames)
        tree = cKDTree(active_positions)
        unmatched = set(range(len(detections)))

        for det_idx, det in enumerate(detections):
            dist, nn_idx = tree.query(det)
            if dist == np.inf:
                continue
            track_id = track_ids[nn_idx]
            frame_gap = f - track_last_frames[nn_idx]
            adaptive_distance = float(max_tracking_distance) + (frame_gap - 1) * 10.0

            predicted_x = active_positions[nn_idx][0] + track_velocities[nn_idx][0] * frame_gap
            predicted_y = active_positions[nn_idx][1] + track_velocities[nn_idx][1] * frame_gap
            pred_error  = np.linalg.norm(np.array([det[0] - predicted_x, det[1] - predicted_y]))

            if dist <= adaptive_distance and pred_error <= float(max_pred_error):
                if len(active_tracks[track_id]) >= 2:
                    prev_x, prev_y = active_tracks[track_id][-2][1:]
                    last_x, last_y = active_tracks[track_id][-1][1:]
                    dir_prev = np.array([last_x - prev_x, last_y - prev_y])
                    dir_next = np.array([det[0] - last_x, det[1] - last_y])
                    if np.linalg.norm(dir_prev) > float(min_jump_distance):
                        angle = _angle_between(dir_prev, dir_next)
                        if angle > float(max_merge_angle_deg):
                            continue
                if frame_gap > 1:
                    start_x, start_y = active_tracks[track_id][-1][1:]
                    for interp_frame in range(last_seen[track_id] + 1, f):
                        frac = (interp_frame - last_seen[track_id]) / frame_gap
                        ix = start_x + frac * (det[0] - start_x)
                        iy = start_y + frac * (det[1] - start_y)
                        active_tracks[track_id].append((interp_frame, ix, iy))

                active_tracks[track_id].append((f, det[0], det[1]))
                last_seen[track_id] = f
                unmatched.discard(det_idx)

        for det_idx in unmatched:
            det = detections[det_idx]
            if border_margin < det[0] < width - border_margin and border_margin < det[1] < height - border_margin:
                active_tracks[next_track_id] = [(f, det[0], det[1])]
                last_seen[next_track_id] = f
                next_track_id += 1

    tracks = [v for v in active_tracks.values() if len(v) >= int(min_track_length)]
    return tracks

# --- CuPy tracker ---
def link_tracks_gpu(frame_objects, width, height,
                    max_frame_gap=5, max_tracking_distance=90,
                    min_jump_distance=10, max_merge_angle_deg=180,
                    max_pred_error=30, border_margin=5, min_track_length=1):
    if not CUPY_AVAILABLE:
        return link_tracks_cpu(frame_objects, width, height,
                               max_frame_gap, max_tracking_distance,
                               min_jump_distance, max_merge_angle_deg,
                               max_pred_error, border_margin, min_track_length)

    active_tracks = {}
    last_seen = {}
    next_track_id = 0

    for (x, y) in frame_objects.get(0, []):
        if border_margin < x < width - border_margin and border_margin < y < height - border_margin:
            active_tracks[next_track_id] = [(0, float(x), float(y))]
            last_seen[next_track_id] = 0
            next_track_id += 1

    for f in tqdm(range(1, len(frame_objects)), desc="Track (GPU)", colour="cyan"):
        detections_np = frame_objects.get(f, np.array([]))
        if len(detections_np) == 0:
            continue
        detections = cp.asarray(detections_np, dtype=cp.float32)

        act_pos, act_ids, act_last, act_vel = [], [], [], []
        for tid, tr in active_tracks.items():
            if f - last_seen[tid] <= int(max_frame_gap):
                if len(tr) >= 2:
                    dx = tr[-1][1] - tr[-2][1]
                    dy = tr[-1][2] - tr[-2][2]
                    act_vel.append((dx, dy))
                else:
                    act_vel.append((0.0, 0.0))
                act_pos.append((tr[-1][1], tr[-1][2]))
                act_ids.append(tid)
                act_last.append(last_seen[tid])

        if len(act_pos) == 0:
            for det in detections_np:
                if border_margin < det[0] < width - border_margin and border_margin < det[1] < height - border_margin:
                    active_tracks[next_track_id] = [(f, float(det[0]), float(det[1]))]
                    last_seen[next_track_id] = f
                    next_track_id += 1
            continue

        act_pos = cp.asarray(act_pos, dtype=cp.float32)
        act_vel = cp.asarray(act_vel, dtype=cp.float32)
        act_last = cp.asarray(act_last, dtype=cp.int32)
        N = detections.shape[0]

        d2 = cp.sum((detections[:, None, :] - act_pos[None, :, :])**2, axis=2)
        d = cp.sqrt(d2 + 1e-9)
        nn_idx = cp.argmin(d, axis=1)
        nn_dist = d[cp.arange(N), nn_idx]

        unmatched = set(range(N))
        for det_idx in range(N):
            j = int(nn_idx[det_idx].get())
            dist = float(nn_dist[det_idx].get())
            track_id = act_ids[j]
            frame_gap = int(f - int(act_last[j].get()))
            adaptive_distance = float(max_tracking_distance) + (frame_gap - 1) * 10.0

            predicted = act_pos[j] + act_vel[j] * float(frame_gap)
            px, py = float(predicted[0].get()), float(predicted[1].get())
            pred_error = float(cp.linalg.norm(detections[det_idx] - cp.asarray([px, py], dtype=cp.float32)).get())

            if dist <= adaptive_distance and pred_error <= float(max_pred_error):
                if len(active_tracks[track_id]) >= 2:
                    prev_x, prev_y = active_tracks[track_id][-2][1:]
                    last_x, last_y = active_tracks[track_id][-1][1:]
                    dir_prev = np.array([last_x - prev_x, last_y - prev_y], dtype=float)
                    detx, dety = float(detections[det_idx,0].get()), float(detections[det_idx,1].get())
                    dir_next = np.array([detx - last_x, dety - last_y], dtype=float)
                    if np.linalg.norm(dir_prev) > float(min_jump_distance):
                        angle = _angle_between(dir_prev, dir_next)
                        if angle > float(max_merge_angle_deg):
                            continue
                if frame_gap > 1:
                    start_x, start_y = active_tracks[track_id][-1][1:]
                    detx, dety = float(detections[det_idx,0].get()), float(detections[det_idx,1].get())
                    for interp_frame in range(last_seen[track_id] + 1, f):
                        frac = (interp_frame - last_seen[track_id]) / frame_gap
                        ix = start_x + frac * (detx - start_x)
                        iy = start_y + frac * (dety - start_y)
                        active_tracks[track_id].append((interp_frame, ix, iy))

                active_tracks[track_id].append((f, float(detections[det_idx,0].get()),
                                                   float(detections[det_idx,1].get())))
                last_seen[track_id] = f
                unmatched.discard(int(det_idx))

        for det_idx in unmatched:
            detx, dety = float(detections[det_idx,0].get()), float(detections[det_idx,1].get())
            if border_margin < detx < width - border_margin and border_margin < dety < height - border_margin:
                active_tracks[next_track_id] = [(f, detx, dety)]
                last_seen[next_track_id] = f
                next_track_id += 1

    tracks = [v for v in active_tracks.values() if len(v) >= int(min_track_length)]
    return tracks

def merge_track_fragments(tracks, max_gap=2, max_join_distance=60):
    final_tracks = []
    used = set()
    for i, tr1 in enumerate(tracks):
        if i in used:
            continue
        merged = list(tr1)
        for j, tr2 in enumerate(tracks):
            if i == j or j in used:
                continue
            end_frame, end_x, end_y = merged[-1]
            start_frame, start_x, start_y = tr2[0]
            if 0 < (start_frame - end_frame) <= int(max_gap):
                if np.linalg.norm([start_x - end_x, start_y - end_y]) <= float(max_join_distance):
                    merged.extend(tr2)
                    used.add(j)
        final_tracks.append(merged)
    return final_tracks

def tracks_to_napari_array(tracks):
    data = []
    for tidx, tr in enumerate(tracks):
        for f, x, y in tr:
            data.append([tidx, f, y, x])
    if len(data) == 0:
        return np.zeros((0, 4), dtype=float)
    return np.array(data, dtype=float)

# -----------------------------
# Initial segmentation
# -----------------------------
print("Running initial segmentation (seeded_ws defaults)…")
initial_source = green_minmax
per_frame_labeled_masks, frame_objects, raw_label_stack, frame_objects_com = segment_frames(
    initial_source,
    method=default_method,
    min_object_size=min_object_size,
    max_object_size=max_object_size,
    strict_min_area=strict_min_area,
    use_clear_border=use_clear_border,
    seg_smooth_sigma=seg_smooth_sigma,
    unsharp_radius=unsharp_radius,
    unsharp_amount=unsharp_amount,
    low_pct=hyst_low_pct,
    high_pct=hyst_high_pct,
    sobel_weight=sobel_weight,
    edge_sigma=edge_sigma,
    log_sigma=log_sigma,
    log_thresh_rel=log_thresh_rel,
    ws_min_distance=ws_min_distance,
    ws_footprint=ws_footprint,
    ws_compactness=ws_compactness,
    ws_erode_before=ws_erode_before,
    ws_opening=ws_opening,
    ws_h_minima=ws_h_minima,
    ws_dilate_after=ws_dilate_after,
    use_gpu_seg=CLE_AVAILABLE,
)

segmentation_labels = ghost_label_tracking(
    per_frame_labeled_masks,
    intensity_stack=initial_source,
    iou_threshold=iou_threshold,
    max_centroid_distance=max_centroid_distance,
    max_ghost_gap=max_ghost_gap,
    search_radius=search_radius,
    use_gpu=CUPY_AVAILABLE,
    com_weight=ghost_com_weight
)

# First pass tracks
def _link_dispatch(frame_objs, use_gpu):
    if use_gpu and CUPY_AVAILABLE:
        return link_tracks_gpu(frame_objs, width, height,
                               max_frame_gap=max_frame_gap,
                               max_tracking_distance=max_tracking_distance,
                               min_jump_distance=min_jump_distance,
                               max_merge_angle_deg=max_merge_angle_deg,
                               max_pred_error=max_pred_error)
    return link_tracks_cpu(frame_objs, width, height,
                           max_frame_gap=max_frame_gap,
                           max_tracking_distance=max_tracking_distance,
                           min_jump_distance=min_jump_distance,
                           max_merge_angle_deg=max_merge_angle_deg,
                           max_pred_error=max_pred_error)

tracks = _link_dispatch(frame_objects, use_gpu=False)
tracks = filter_tracks_displacement(tracks, frame_interval_sec=frame_interval_sec,
                                    max_allowed_velocity=8, min_total_displacement=5,
                                    max_stall_duration=6, min_total_duration=5)
tracks = merge_track_fragments(tracks, max_gap=2, max_join_distance=60)

# Build ghost tracks from labels (HYBRID point)
ghost_tracks_raw = {}
w_hybrid_default = float(np.clip(ghost_com_weight, 0.0, 1.0))
for f in range(segmentation_labels.shape[0]):
    props = regionprops(segmentation_labels[f], intensity_image=initial_source[f])
    for r in props:
        if r.label <= 0:
            continue
        cy, cx = r.centroid
        if r.weighted_centroid is not None:
            wcy, wcx = r.weighted_centroid
        else:
            wcy, wcx = cy, cx
        hy = (1.0 - w_hybrid_default) * cy + w_hybrid_default * wcy
        hx = (1.0 - w_hybrid_default) * cx + w_hybrid_default * wcx
        ghost_tracks_raw.setdefault(int(r.label), []).append((f, float(hx), float(hy)))
ghost_tracks = [v for v in ghost_tracks_raw.values() if len(v) >= 3]

# -----------------------------
# Napari viewer + widgets
# -----------------------------
try:
    from magicgui import magicgui
except Exception:
    raise RuntimeError("magicgui is required. Install with: pip install magicgui")

STATE = {
    'per_frame_labeled_masks': per_frame_labeled_masks,
    'raw_label_stack': raw_label_stack,
    'segmentation_labels': segmentation_labels,
    'frame_objects': frame_objects,
    'tracks': tracks,
    'ghost_tracks': ghost_tracks,
    'intensity_stack': initial_source,
    'ghost_com_weight': ghost_com_weight,
}

print("\n🔧 Use 'Update Segmentation' first until objects look good, then 'Compute Tracks'.")
print("   When satisfied, CLOSE this Napari window to export figures/CSVs and capture the movie.\n")

viewer = napari.Viewer()
viewer.add_image(green_normalized, name='Green Channel', contrast_limits=[0, 1], colormap='green', gamma=0.5)
viewer.add_image(red_normalized,   name='Red Channel',   contrast_limits=[0, 1], colormap='magenta', opacity=0.8)

raw_labels_layer   = viewer.add_labels(STATE['raw_label_stack'],        name='Per-frame Labels (raw)', opacity=0.6)
labels_layer       = viewer.add_labels(STATE['segmentation_labels'],    name='Ghost-Inferred Labels', opacity=0.5)
tracks_layer       = viewer.add_tracks(tracks_to_napari_array(STATE['tracks']),        name='Tracks',       tail_length=150, tail_width=5, colormap='turbo')
ghost_tracks_layer = viewer.add_tracks(tracks_to_napari_array(STATE['ghost_tracks']),  name='Ghost Tracks', tail_length=150, tail_width=5, colormap='viridis')

@magicgui(
    auto_call=False, layout='vertical', call_button='Update Segmentation',
    method={'label':'Segmentation method','choices':['seeded_ws','edgefusion','hysteresis'],'value':default_method},
    use_gpu_seg={'label':'Use GPU for segmentation','widget_type':'CheckBox','value':CLE_AVAILABLE},
    # Shared
    min_object_size={'label':'Min area','min':1,'max':2000,'step':1,'value':min_object_size},
    max_object_size={'label':'Max area','min':10,'max':10000,'step':10,'value':max_object_size},
    strict_min_area={'label':'Strict min area','min':0,'max':5000,'step':10,'value':strict_min_area},
    use_clear_border={'label':'Clear border','widget_type':'CheckBox','value':use_clear_border},
    seg_smooth_sigma={'label':'Smooth σ','min':0.0,'max':5.0,'step':0.1,'value':seg_smooth_sigma},
    unsharp_radius={'label':'Unsharp radius','min':0,'max':40,'step':1,'value':unsharp_radius},
    unsharp_amount={'label':'Unsharp amount','min':0.0,'max':30.0,'step':0.1,'value':unsharp_amount},
    # Hysteresis
    low_pct={'label':'Hyst. low %','min':0.0,'max':95.0,'step':1.0,'value':hyst_low_pct},
    high_pct={'label':'Hyst. high %','min':5.0,'max':100.0,'step':1.0,'value':hyst_high_pct},
    # Edge-fusion
    sobel_weight={'label':'Sobel weight','min':0.0,'max':1.0,'step':0.05,'value':sobel_weight},
    edge_sigma={'label':'Edge smooth σ','min':0.0,'max':5.0,'step':0.1,'value':edge_sigma},
    # Seeded watershed
    log_sigma={'label':'LoG σ','min':0.5,'max':6.0,'step':0.1,'value':log_sigma},
    log_thresh_rel={'label':'LoG rel. thr','min':0.0,'max':1.0,'step':0.01,'value':log_thresh_rel},
    ws_min_distance={'label':'WS min distance','min':1,'max':50,'step':1,'value':ws_min_distance},
    ws_footprint={'label':'WS footprint (odd)','min':3,'max':51,'step':2,'value':ws_footprint},
    ws_compactness={'label':'WS compactness','min':0.0,'max':2.0,'step':0.05,'value':ws_compactness},
    ws_erode_before={'label':'WS erode before','min':0,'max':5,'step':1,'value':ws_erode_before},
    ws_opening={'label':'WS opening','min':0,'max':5,'step':1,'value':ws_opening},
    ws_h_minima={'label':'WS h-minima','min':0.0,'max':5.0,'step':0.1,'value':ws_h_minima},
    ws_dilate_after={'label':'WS dilate after','min':0,'max':5,'step':1,'value':ws_dilate_after},
    ws_enable_extra_split={'label':'Extra WS split','widget_type':'CheckBox','value':ws_enable_extra_split},
)
def update_segmentation(
    method: str,
    use_gpu_seg: bool,
    min_object_size: int,
    max_object_size: int,
    strict_min_area: int,
    use_clear_border: bool,
    seg_smooth_sigma: float,
    unsharp_radius: int,
    unsharp_amount: float,
    low_pct: float,
    high_pct: float,
    sobel_weight: float,
    edge_sigma: float,
    log_sigma: float,
    log_thresh_rel: float,
    ws_min_distance: int,
    ws_footprint: int,
    ws_compactness: float,
    ws_erode_before: int,
    ws_opening: int,
    ws_h_minima: float,
    ws_dilate_after: int,
    ws_enable_extra_split: bool,
):
    print(f"\n🧩 Updating segmentation → {method} | GPU: {'ON' if (use_gpu_seg and CLE_AVAILABLE) else 'OFF'}")
    source_stack = green_minmax

    per_lbl, frame_objs, raw_stack, frame_objs_com = segment_frames(
        source_stack,
        method=method,
        min_object_size=min_object_size,
        max_object_size=max_object_size,
        strict_min_area=strict_min_area,
        use_clear_border=use_clear_border,
        seg_smooth_sigma=seg_smooth_sigma,
        unsharp_radius=unsharp_radius,
        unsharp_amount=unsharp_amount,
        low_pct=low_pct,
        high_pct=high_pct,
        sobel_weight=sobel_weight,
        edge_sigma=edge_sigma,
        log_sigma=log_sigma,
        log_thresh_rel=log_thresh_rel,
        ws_min_distance=ws_min_distance,
        ws_footprint=ws_footprint,
        ws_compactness=ws_compactness,
        ws_erode_before=ws_erode_before,
        ws_opening=ws_opening,
        ws_h_minima=ws_h_minima,
        ws_dilate_after=ws_dilate_after,
        use_gpu_seg=(use_gpu_seg and CLE_AVAILABLE),
    )

    STATE['per_frame_labeled_masks'] = per_lbl
    STATE['raw_label_stack'] = raw_stack
    STATE['frame_objects'] = frame_objs
    STATE['intensity_stack'] = source_stack
    raw_labels_layer.data = raw_stack
    print("Segmentation updated. Now click 'Compute Tracks' to re-run ghost relabeling + linking.")

try:
    from magicgui import magicgui  # already imported above; keeps mypy happy
    DEFAULT_GPU_TRACK = CUPY_AVAILABLE
except Exception:
    DEFAULT_GPU_TRACK = False

@magicgui(
    auto_call=False, layout='vertical', call_button='Compute Tracks',
    use_gpu_track={'label':'Use GPU for tracking (CuPy)','widget_type':'CheckBox','value':DEFAULT_GPU_TRACK},
    iou_threshold={'label':'IoU threshold','min':0.01,'max':0.9,'step':0.01,'value':iou_threshold},
    max_centroid_distance={'label':'Max centroid dist','min':3,'max':100,'step':1,'value':max_centroid_distance},
    max_ghost_gap={'label':'Max ghost gap (frames)','min':1,'max':20,'step':1,'value':max_ghost_gap},
    search_radius={'label':'Relabel search radius','min':3,'max':80,'step':1,'value':search_radius},
    com_weight={'label':'Ghost COM weight','min':0.0,'max':1.0,'step':0.05,'value':ghost_com_weight},
    max_tracking_distance={'label':'Max link distance','min':10,'max':600,'step':5,'value':max_tracking_distance},
    max_frame_gap={'label':'Max frame gap','min':1,'max':10,'step':1,'value':max_frame_gap},
    min_jump_distance={'label':'Min jump (dir test)','min':0,'max':50,'step':1,'value':min_jump_distance},
    max_merge_angle_deg={'label':'Max turn angle','min':5,'max':180,'step':1,'value':max_merge_angle_deg},
    max_pred_error={'label':'Max pred. error','min':1,'max':200,'step':1,'value':max_pred_error},
    min_final_track_len={'label':'Min final track length','min':1,'max':500,'step':1,'value':10},
    # Displacement QC
    max_allowed_velocity={'label':'Max velocity (px/s)','min':0.1,'max':100.0,'step':0.1,'value':8.0},
    min_total_displacement={'label':'Min net disp (px)','min':0.0,'max':200.0,'step':1.0,'value':5.0},
    max_stall_duration={'label':'Max stall (frames)','min':1,'max':50,'step':1,'value':6},
    min_total_duration={'label':'Min duration (frames)','min':1,'max':500,'step':1,'value':5},
)
def compute_tracks(
    use_gpu_track: bool,
    iou_threshold: float,
    max_centroid_distance: int,
    max_ghost_gap: int,
    search_radius: int,
    com_weight: float,
    max_tracking_distance: int,
    max_frame_gap: int,
    min_jump_distance: int,
    max_merge_angle_deg: int,
    max_pred_error: int,
    min_final_track_len: int,
    max_allowed_velocity: float,
    min_total_displacement: float,
    max_stall_duration: int,
    min_total_duration: int,
):
    if STATE.get('per_frame_labeled_masks') is None:
        print("⚠️  No segmentation present. Click 'Update Segmentation' first.")
        return

    gpu_on = bool(use_gpu_track and CUPY_AVAILABLE)
    print(f"\n🔁 Computing ghost relabeling + tracks … (GPU tracking: {'ON' if gpu_on else 'OFF'}, COM weight={com_weight:.2f})")

    seg_lbls = ghost_label_tracking(
        STATE['per_frame_labeled_masks'],
        intensity_stack=STATE['intensity_stack'],
        iou_threshold=iou_threshold,
        max_centroid_distance=max_centroid_distance,
        max_ghost_gap=max_ghost_gap,
        search_radius=search_radius,
        use_gpu=gpu_on,
        com_weight=com_weight
    )

    linker = link_tracks_gpu if gpu_on else link_tracks_cpu

    tr = linker(
        STATE['frame_objects'], width, height,
        max_frame_gap=max_frame_gap,
        max_tracking_distance=max_tracking_distance,
        min_jump_distance=min_jump_distance,
        max_merge_angle_deg=max_merge_angle_deg,
        max_pred_error=max_pred_error,
    )
    tr = merge_track_fragments(tr, max_gap=2, max_join_distance=60)

    # Build ghost tracks from ghost-inferred labels (HYBRID points)
    ghost_raw = {}
    w = float(np.clip(com_weight, 0.0, 1.0))
    for f in range(seg_lbls.shape[0]):
        props = regionprops(seg_lbls[f], intensity_image=STATE['intensity_stack'][f])
        for r in props:
            if r.label <= 0:
                continue
            cy, cx = r.centroid
            if r.weighted_centroid is not None:
                wcy, wcx = r.weighted_centroid
            else:
                wcy, wcx = cy, cx
            hy = (1.0 - w) * cy + w * wcy
            hx = (1.0 - w) * cx + w * wcx
            ghost_raw.setdefault(int(r.label), []).append((f, float(hx), float(hy)))

    # Displacement-aware QC applied uniformly
    tr = filter_tracks_displacement(
        tr,
        frame_interval_sec=frame_interval_sec,
        max_allowed_velocity=max_allowed_velocity,
        min_total_displacement=min_total_displacement,
        max_stall_duration=max_stall_duration,
        min_total_duration=min_total_duration
    )
    tr = length_filter(tr, min_final_track_len)

    ghost = list(ghost_raw.values())
    ghost = filter_tracks_displacement(
        ghost,
        frame_interval_sec=frame_interval_sec,
        max_allowed_velocity=max_allowed_velocity,
        min_total_displacement=min_total_displacement,
        max_stall_duration=max_stall_duration,
        min_total_duration=max(min_total_duration, 3)
    )
    ghost = length_filter(ghost, max(min_final_track_len, 3))

    STATE['segmentation_labels'] = seg_lbls
    STATE['tracks'] = tr
    STATE['ghost_tracks'] = ghost
    STATE['ghost_com_weight'] = float(com_weight)

    labels_layer.data = seg_lbls
    tracks_layer.data = tracks_to_napari_array(tr)
    ghost_tracks_layer.data = tracks_to_napari_array(ghost)

    print(f"Done. Tracks: {len(tr)} | Ghost: {len(ghost)}")

viewer.window.add_dock_widget(update_segmentation, area='right')
viewer.window.add_dock_widget(compute_tracks,    area='right')
viewer.camera.zoom = 0.9

print("👉 Default is seeded_ws. 1) Update Segmentation  2) Compute Tracks  3) Close window to export/movie")
napari.run()

# =============================
# Exports using FINAL results (RIGHT-FORMAT OUTPUTS)
# =============================
print("\n✅ Viewer closed. Proceeding with exports and figures using FINAL parameters/results.")
segmentation_labels     = STATE['segmentation_labels']
frame_objects           = STATE['frame_objects']
tracks                  = STATE['tracks']
ghost_tracks            = STATE['ghost_tracks']
intensity_stack         = STATE['intensity_stack']
w_hybrid                = float(np.clip(STATE.get('ghost_com_weight', ghost_com_weight), 0.0, 1.0))

# --- TRACK CSVs (motion tracks) --- schema: track_id,frame,x,y
print("Saving tracks to CSV…")
track_rows = []
for tidx, tr in enumerate(tracks):
    for f, x, y in tr:
        track_rows.append([tidx, int(f), float(x), float(y)])
df_tracks = pd.DataFrame(track_rows, columns=["track_id","frame","x","y"])
df_tracks = df_tracks[["track_id","frame","x","y"]]  # enforce order
df_tracks.to_csv(os.path.join(output_dir, f"{base_filename}_tracks.csv"), index=False)

# --- GHOST TRACKS CSV --- schema EXACTLY: track_id,frame,x,y
print("Saving ghost tracks CSV…")
ghost_rows_simple = []
for tidx, tr in enumerate(ghost_tracks):
    for f, x, y in tr:
        ghost_rows_simple.append([tidx, int(f), float(x), float(y)])
df_ghosttracks = pd.DataFrame(ghost_rows_simple, columns=["track_id","frame","x","y"])
df_ghosttracks = df_ghosttracks[["track_id","frame","x","y"]]  # enforce order
df_ghosttracks.to_csv(os.path.join(output_dir, f"{base_filename}_ghosttracks.csv"), index=False)

# --- METRICS & PLOTS (Tracks) ---
print("Generating motility metrics (tracks)…")
velocity_per_frame = {}
for tr in tracks:
    coords = np.array([[x, y] for _, x, y in tr], dtype=float)
    frames_track = np.array([f for f, _, _ in tr], dtype=int)
    if len(coords) >= 2:
        distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        frames_mid = (frames_track[:-1] + frames_track[1:]) // 2
        for idx, d in enumerate(distances):
            fidx = int(frames_mid[idx])
            velocity_per_frame.setdefault(fidx, []).append(d / frame_interval_sec)

all_frames = sorted(velocity_per_frame.keys())
velocities_mean = [np.mean(velocity_per_frame[f]) for f in all_frames] if all_frames else []
velocities_sem  = [np.std(velocity_per_frame[f]) / max(1, np.sqrt(len(velocity_per_frame[f]))) for f in all_frames] if all_frames else []

if all_frames:
    plt.figure(); plt.plot(np.array(all_frames) * frame_interval_sec, velocities_mean)
    plt.fill_between(np.array(all_frames) * frame_interval_sec,
                     np.array(velocities_mean) - 1.96 * np.array(velocities_sem),
                     np.array(velocities_mean) + 1.96 * np.array(velocities_sem), alpha=0.3)
    plt.xlabel("Time (sec)"); plt.ylabel("Velocity (pixels/sec)"); plt.title("Velocity Over Time")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"{base_filename}_velocity.png")); plt.close()

window_size_frames = 5
persistence_per_frame = {}
for tr in tracks:
    frames_track = np.array([f for f, _, _ in tr], dtype=int)
    coords = np.array([[x, y] for _, x, y in tr], dtype=float)
    if len(coords) < window_size_frames + 1:
        continue
    for i in range(len(coords) - window_size_frames):
        start_frame = frames_track[i]
        end_frame   = frames_track[i + window_size_frames]
        path        = coords[i:i + window_size_frames + 1]
        net_disp    = np.linalg.norm(coords[i + window_size_frames] - coords[i])
        path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        if path_length > 0:
            persistence = net_disp / path_length
            midf = int((start_frame + end_frame) // 2)
            persistence_per_frame.setdefault(midf, []).append(persistence)

all_frames_persistence = sorted(persistence_per_frame.keys())
if all_frames_persistence:
    persistence_mean = [np.mean(persistence_per_frame[f]) for f in all_frames_persistence]
    persistence_sem  = [np.std(persistence_per_frame[f]) / max(1, np.sqrt(len(persistence_per_frame[f]))) for f in all_frames_persistence]
    plt.figure(); plt.plot(np.array(all_frames_persistence) * frame_interval_sec, persistence_mean)
    plt.fill_between(np.array(all_frames_persistence) * frame_interval_sec,
                     np.array(persistence_mean) - 1.96 * np.array(persistence_sem),
                     np.array(persistence_mean) + 1.96 * np.array(persistence_sem), alpha=0.3)
    plt.xlabel("Time (sec)"); plt.ylabel("Directional Persistence"); plt.title("Directional Persistence Over Time")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"{base_filename}_persistence.png")); plt.close()

if all_frames and len(all_frames) >= 2:
    frame_times = np.array(all_frames) * frame_interval_sec
    vel_arr = np.array(velocities_mean)
    accelerations = np.diff(vel_arr) / np.diff(frame_times)
    frame_times_acc = (frame_times[:-1] + frame_times[1:]) / 2
    plt.figure(); plt.plot(frame_times_acc, accelerations)
    plt.xlabel("Time (sec)"); plt.ylabel("Acceleration (pixels/sec²)"); plt.title("Acceleration Over Time")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"{base_filename}_acceleration.png")); plt.close()

displacement_per_frame = {}
for tr in tracks:
    frames_track = np.array([f for f, _, _ in tr], dtype=int)
    coords = np.array([[x, y] for _, x, y in tr], dtype=float)
    if len(coords) < 2:
        continue
    start_coord = coords[0]
    for idx in range(len(coords)):
        frame_idx = int(frames_track[idx])
        net_disp  = np.linalg.norm(coords[idx] - start_coord)
        displacement_per_frame.setdefault(frame_idx, []).append(net_disp)

all_frames_disp = sorted(displacement_per_frame.keys())
if all_frames_disp:
    displacement_mean = [np.mean(displacement_per_frame[f]) for f in all_frames_disp]
    displacement_sem  = [np.std(displacement_per_frame[f]) / max(1, np.sqrt(len(displacement_per_frame[f]))) for f in all_frames_disp]
    plt.figure(); plt.plot(np.array(all_frames_disp) * frame_interval_sec, displacement_mean)
    plt.fill_between(np.array(all_frames_disp) * frame_interval_sec,
                     np.array(displacement_mean) - 1.96 * np.array(displacement_sem),
                     np.array(displacement_mean) + 1.96 * np.array(displacement_sem), alpha=0.3)
    plt.xlabel("Time (sec)"); plt.ylabel("Net Displacement (pixels)"); plt.title("Net Displacement Over Time")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"{base_filename}_netdisplacement.png")); plt.close()

angles = []
for tr in tracks:
    coords = np.array([[x, y] for _, x, y in tr], dtype=float)
    if len(coords) >= 2:
        deltas = np.diff(coords, axis=0)
        angles.extend(np.arctan2(deltas[:, 1], deltas[:, 0]))
if angles:
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.hist(angles, bins=32, density=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    plt.title("Movement Direction Rose Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_filename}_roseplot.png"))
    plt.close()

# --- GHOST-BASED METRICS & PLOTS ---
print("Generating ghost-based metrics + right-format CSVs…")

ghost_velocity_per_frame = {}
for tr in ghost_tracks:
    coords = np.array([[x, y] for _, x, y in tr], dtype=float)
    frames_track = np.array([f for f, _, _ in tr], dtype=int)
    if len(coords) >= 2:
        distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        frames_mid = (frames_track[:-1] + frames_track[1:]) // 2
        for idx, d in enumerate(distances):
            fidx = int(frames_mid[idx])
            ghost_velocity_per_frame.setdefault(fidx, []).append(d / frame_interval_sec)

all_frames_ghost = sorted(ghost_velocity_per_frame.keys())
ghost_vel_mean = [np.mean(ghost_velocity_per_frame[f]) for f in all_frames_ghost] if all_frames_ghost else []
ghost_vel_sem  = [np.std(ghost_velocity_per_frame[f]) / max(1, np.sqrt(len(ghost_velocity_per_frame[f]))) for f in all_frames_ghost] if all_frames_ghost else []

if all_frames_ghost:
    plt.figure(); plt.plot(np.array(all_frames_ghost) * frame_interval_sec, ghost_vel_mean)
    plt.fill_between(np.array(all_frames_ghost) * frame_interval_sec,
                     np.array(ghost_vel_mean) - 1.96 * np.array(ghost_vel_sem),
                     np.array(ghost_vel_mean) + 1.96 * np.array(ghost_vel_sem), alpha=0.3)
    plt.xlabel("Time (sec)"); plt.ylabel("Velocity (pixels/sec)"); plt.title("Ghost-Based Velocity Over Time")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"{base_filename}_ghost_velocity.png")); plt.close()

ghost_persistence_per_frame = {}
window_size_frames = 5
for tr in ghost_tracks:
    frames_track = np.array([f for f, _, _ in tr], dtype=int)
    coords = np.array([[x, y] for _, x, y in tr], dtype=float)
    if len(coords) < window_size_frames + 1:
        continue
    for i in range(len(coords) - window_size_frames):
        start_frame = frames_track[i]
        end_frame   = frames_track[i + window_size_frames]
        path        = coords[i:i + window_size_frames + 1]
        net_disp    = np.linalg.norm(coords[i + window_size_frames] - coords[i])
        path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        if path_length > 0:
            persistence = net_disp / path_length
            midf = int((start_frame + end_frame) // 2)
            ghost_persistence_per_frame.setdefault(midf, []).append(persistence)

frames_ghost_pers = sorted(ghost_persistence_per_frame.keys())
if frames_ghost_pers:
    ghost_pers_mean = [np.mean(ghost_persistence_per_frame[f]) for f in frames_ghost_pers]
    ghost_pers_sem  = [np.std(ghost_persistence_per_frame[f]) / max(1, np.sqrt(len(ghost_persistence_per_frame[f]))) for f in frames_ghost_pers]
    plt.figure(); plt.plot(np.array(frames_ghost_pers) * frame_interval_sec, ghost_pers_mean)
    plt.fill_between(np.array(frames_ghost_pers) * frame_interval_sec,
                     np.array(ghost_pers_mean) - 1.96 * np.array(ghost_pers_sem),
                     np.array(ghost_pers_mean) + 1.96 * np.array(ghost_pers_sem), alpha=0.3)
    plt.xlabel("Time (sec)"); plt.ylabel("Directional Persistence"); plt.title("Ghost-Based Directional Persistence")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"{base_filename}_ghost_persistence.png")); plt.close()

if all_frames_ghost and len(all_frames_ghost) >= 2:
    frame_times_ghost = np.array(all_frames_ghost) * frame_interval_sec
    vel_arr_ghost     = np.array(ghost_vel_mean)
    ghost_acc         = np.diff(vel_arr_ghost) / np.diff(frame_times_ghost)
    acc_frame_mid     = (frame_times_ghost[:-1] + frame_times_ghost[1:]) / 2
    plt.figure(); plt.plot(acc_frame_mid, ghost_acc)
    plt.xlabel("Time (sec)"); plt.ylabel("Acceleration (pixels/sec²)"); plt.title("Ghost-Based Acceleration")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"{base_filename}_ghost_acceleration.png")); plt.close()

# ghost displacement (per frame mean/sem) for plots + CSV
ghost_disp_per_frame = {}
for tr in ghost_tracks:
    coords = np.array([[x, y] for _, x, y in tr], dtype=float)
    frames = np.array([f for f, _, _ in tr], dtype=int)
    if len(coords) < 2:
        continue
    start = coords[0]
    for i in range(len(coords)):
        disp = float(np.linalg.norm(coords[i] - start))
        ghost_disp_per_frame.setdefault(int(frames[i]), []).append(disp)

frames_disp = sorted(ghost_disp_per_frame.keys())
disp_mean = [float(np.mean(ghost_disp_per_frame[f])) for f in frames_disp] if frames_disp else []
disp_sem  = [float(np.std(ghost_disp_per_frame[f]) / max(1, np.sqrt(len(ghost_disp_per_frame[f])))) for f in frames_disp] if frames_disp else []

if frames_disp:
    plt.figure(); plt.plot(np.array(frames_disp) * frame_interval_sec, disp_mean)
    plt.fill_between(np.array(frames_disp) * frame_interval_sec,
                     np.array(disp_mean) - 1.96 * np.array(disp_sem),
                     np.array(disp_mean) + 1.96 * np.array(disp_sem), alpha=0.3)
    plt.xlabel("Time (sec)"); plt.ylabel("Net Displacement (pixels)"); plt.title("Ghost-Based Net Displacement")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"{base_filename}_ghost_displacement.png")); plt.close()

# --- RIGHT-FORMAT ghost_metrics.csv ---
# Columns and order are enforced explicitly: frame,time_sec,velocity,velocity_sem,displacement,displacement_sem
print("Saving ghost metrics CSV (STRICT column names/order)…")

# Build per-frame maps from earlier results (already computed above)
vel_map      = {int(f): float(v)  for f, v in zip(all_frames_ghost or [], ghost_vel_mean or [])}
vel_sem_map  = {int(f): float(vs) for f, vs in zip(all_frames_ghost or [], ghost_vel_sem or [])}
disp_map     = {int(f): float(d)  for f, d in zip(frames_disp or [], disp_mean or [])}
disp_sem_map = {int(f): float(ds) for f, ds in zip(frames_disp or [], disp_sem or [])}

# Union of frames seen in either velocity or displacement
frames_union = sorted(set(vel_map.keys()) | set(disp_map.keys()))

# If absolutely no frames, still write an empty CSV with headers
if len(frames_union) == 0:
    ghost_metrics_df = pd.DataFrame(columns=[
        "frame","time_sec","velocity","velocity_sem","displacement","displacement_sem"
    ])
else:
    rows = []
    for f in frames_union:
        rows.append([
            int(f),
            float(f) * float(frame_interval_sec),
            vel_map.get(f, np.nan),
            vel_sem_map.get(f, np.nan),
            disp_map.get(f, np.nan),
            disp_sem_map.get(f, np.nan),
        ])
    ghost_metrics_df = pd.DataFrame(
        rows,
        columns=["frame","time_sec","velocity","velocity_sem","displacement","displacement_sem"]
    )

# Enforce exact column order and write
ghost_metrics_df = ghost_metrics_df[["frame","time_sec","velocity","velocity_sem","displacement","displacement_sem"]]
ghost_metrics_df.to_csv(os.path.join(output_dir, f"{base_filename}_ghost_metrics.csv"), index=False)

# --- RIGHT-FORMAT ghost_angles.csv ---
# Columns and order are enforced explicitly: angle_rad
print("Saving ghost angles CSV (STRICT column name/order)…")
ghost_angles = []
for tr in ghost_tracks:
    coords = np.array([[x, y] for _, x, y in tr], dtype=float)
    if len(coords) >= 2:
        deltas = np.diff(coords, axis=0)
        ghost_angles.extend(np.arctan2(deltas[:, 1], deltas[:, 0]))
df_angles = pd.DataFrame({"angle_rad": ghost_angles})
df_angles = df_angles[["angle_rad"]]  # enforce order
df_angles.to_csv(os.path.join(output_dir, f"{base_filename}_ghost_angles.csv"), index=False)

# -----------------------------
# Overlays
# -----------------------------
print("Saving overlay images…")
background_proj = np.max(green_minmax, axis=0)
colors = cm.get_cmap('tab20', max(1, len(tracks)))

plt.figure(figsize=(8, 8)); plt.imshow(background_proj, cmap='gray')
for tidx, tr in enumerate(tracks):
    coords = np.array([[x, y] for _, x, y in tr])
    if len(coords) > 0:
        plt.plot(coords[:, 0], coords[:, 1], linewidth=1, color=colors(tidx % 20))
plt.axis('off'); plt.title('Motion Tracks Overlay'); plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{base_filename}_motion_overlay.png")); plt.close()

plt.figure(figsize=(8, 8)); plt.imshow(background_proj, cmap='gray')
for tidx, tr in enumerate(ghost_tracks):
    coords = np.array([[x, y] for _, x, y in tr])
    if len(coords) > 0:
        plt.plot(coords[:, 0], coords[:, 1], linewidth=1, color=colors(tidx % 20))
plt.axis('off'); plt.title('Ghost Tracks Overlay'); plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{base_filename}_ghost_overlay.png")); plt.close()

# -----------------------------
# Movie capture (optional via a second Napari session)
# -----------------------------
if make_movie:
    print("\n📣 Re-opening Napari to capture the movie.")
    import tempfile, imageio
    from qtpy.QtCore import QTimer

    def generate_napari_movie():
        viewer2 = napari.Viewer()
        viewer2.add_image(green_normalized, name='Green Channel', contrast_limits=[0, 1],
                          colormap='green', gamma=0.5, opacity=1.0)
        viewer2.add_image(red_normalized, name='Red Channel', contrast_limits=[0, 1],
                          colormap='magenta', opacity=1.0)
        viewer2.add_labels(segmentation_labels, name='Ghost-Inferred Labels', opacity=0.40)
        ghost_arr = tracks_to_napari_array(ghost_tracks)
        if ghost_arr.shape[0] > 0:
            viewer2.add_tracks(ghost_arr, name='Ghost-Inferred Tracks',
                               tail_length=150, tail_width=5, colormap='turbo')
        viewer2.camera.zoom = 0.9

        n_frames = green_normalized.shape[0]
        temp_dir = tempfile.mkdtemp()
        frame_paths = []

        def capture_all():
            print("🎥 Capturing frames from Napari (second session)…")
            for t in tqdm(range(n_frames), desc="Movie frames", colour="white"):
                viewer2.dims.set_current_step(0, t)
                try:
                    viewer2.window._qt_viewer.canvas.native.repaint()
                except Exception:
                    pass
                shot = viewer2.screenshot(canvas_only=True)
                fp = os.path.join(temp_dir, f"frame_{t:04d}.png")
                imageio.imwrite(fp, shot)
                frame_paths.append(fp)

            out_base = f"{base_filename}_napari_direct"
            out_path = os.path.join(output_dir, f"{out_base}.{movie_format_tracks}")
            try:
                with imageio.get_writer(out_path, fps=fps_movie) as w:
                    for pth in frame_paths:
                        img = imageio.imread(pth)
                        w.append_data(img)
                print(f"✅ Napari movie saved to: {out_path}")
            except Exception as e:
                print(f"⚠️ Movie generation failed: {e}")

        QTimer.singleShot(1000, capture_all)
        napari.run()

    generate_napari_movie()

print(f"\n✅ All done! Full results saved to: {output_dir}")
