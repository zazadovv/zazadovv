# ---  UPDATED SCRIPT ---

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from skimage import io, filters, measure, morphology, exposure
from skimage.segmentation import clear_border
from scipy.spatial import cKDTree
from tqdm import tqdm
import napari
import tkinter as tk
from tkinter import filedialog
import warnings
from matplotlib import cm
import shutil
import matplotlib as mpl
from skimage.draw import disk
warnings.filterwarnings('ignore')

# --- PARAMETERS ---
min_object_size = 15
max_object_size = 600
max_tracking_distance = 250
max_tracking_distance_extended = 300
max_frame_gap = 3
min_track_length = 1
frame_interval_sec = 30
show_arrows = True
make_movie = True
fps_movie = 10
movie_format_tracks = 'avi'  # Choice: 'avi', 'mp4', 'mkv', 'gif'

# --- Safe ffmpeg Setup ---
ffmpeg_path = shutil.which('ffmpeg')
if ffmpeg_path is not None:
    mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path
else:
    print("\u26a0\ufe0f Warning: ffmpeg not found, will save movie as GIF.")

# --- GUI File Selection ---
root = tk.Tk()
root.withdraw()
input_path = filedialog.askopenfilename(title="Select your TIFF image")
output_dir = os.path.dirname(input_path) 

# --- Get base filename ---
base_filename = os.path.splitext(os.path.basename(input_path))[0]
print(f"Selected file base name: {base_filename}")

# --- Load Image ---
print(f"Loading image: {input_path}")
image = io.imread(input_path)
frames, channels, height, width = image.shape
print(f"Image shape: {image.shape}")
green_channel = image[:, 0, :, :]
red_channel = image[:, 1, :, :]
# --- PATCHED GREEN CHANNEL PREPROCESSING & SEGMENTATION ONLY ---
# Drop this directly into your original script in place of the old preprocessing and segmentation sections
# --- GENTLER ROLLING BALL PREPROCESSING ---
from skimage.restoration import rolling_ball
from skimage import exposure, filters
# --- Optional pyclesperanto GPU setup ---
try:
    import pyclesperanto_prototype as cle
    cle.select_device("auto")
    print(f"⚡ Using GPU: {cle.get_device()}")
    CLE_AVAILABLE = True
except Exception as e:
    print("⚠️ pyclesperanto not found or no compatible GPU, using CPU only.")
    CLE_AVAILABLE = False

green_normalized = np.zeros_like(green_channel, dtype=np.float32)
red_normalized = np.zeros_like(red_channel, dtype=np.float32)

def gpu_preprocess(img):
    # Gentle rolling background (top-hat box), then gaussian, then normalize, then gamma
    img_gpu = cle.push(img.astype(np.float32))
    # Remove background (rolling ball/tophat style)
    bg_sub = cle.top_hat_box(img_gpu, radius_x=15, radius_y=15)
    # Smoothing
    smoothed = cle.gaussian_blur(bg_sub, sigma_x=1.5, sigma_y=1.5)
    # Normalize (avoid max_of_images bug!)
    arr = cle.pull(smoothed)
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    # Gamma boost for faint objects
    arr_norm = arr_norm ** 0.8
    return arr_norm.astype(np.float32)

for i in tqdm(range(frames), desc="Preprocessing frames (GPU/CPU)"):
    img = green_channel[i]
    try:
        if CLE_AVAILABLE:
            green_normalized[i] = gpu_preprocess(img)
        else:
            raise Exception("Force CPU")
    except Exception as e:
        # CPU fallback - your exact CPU pipeline
        background = filters.gaussian(img, sigma=10)
        background_smoothed = morphology.opening(background, morphology.disk(15))
        img_subtracted = np.clip(img - background_smoothed, 0, None)
        img_subtracted_norm = exposure.rescale_intensity(img_subtracted, in_range='image', out_range=(0, 1))
        img_eq = exposure.equalize_adapthist(img_subtracted_norm, clip_limit=0.008)
        img_final = filters.gaussian(img_eq, sigma=0.5)
        img_final = exposure.rescale_intensity(img_final, in_range='image', out_range=(0, 1))
        green_normalized[i] = img_final

    # Always rescale red for overlay
    red_normalized[i] = exposure.rescale_intensity(red_channel[i], in_range='image', out_range=(0, 1))


# --- Segmentation ---
print("Segmenting objects...")
all_objects = []
frame_objects = {}
binary_masks = []
per_frame_labeled_masks = []

for i in tqdm(range(frames), desc="Segmenting frames"):
    frame = green_normalized[i]
    smoothed = filters.gaussian(frame, sigma=0.8)
    sharp = filters.unsharp_mask(smoothed, radius=5, amount=2)
    block_size = 101
    local_thresh = filters.threshold_local(sharp, block_size, offset=-0.1)
    binary = sharp > local_thresh
    binary = morphology.binary_closing(binary, morphology.disk(2))
    binary = morphology.remove_small_objects(binary, min_size=min_object_size)
    binary = morphology.remove_small_holes(binary, area_threshold=30)
    binary = clear_border(binary)
    binary_masks.append(binary)
    labeled = measure.label(binary)
    per_frame_labeled_masks.append(labeled)
    props = measure.regionprops(labeled)
    objects_in_frame = []
    for prop in props:
        if min_object_size <= prop.area <= max_object_size:
            y, x = prop.centroid
            all_objects.append((i, x, y))
            objects_in_frame.append((x, y))
    frame_objects[i] = np.array(objects_in_frame)

all_objects = np.array(all_objects)
print(f"Total segmented objects: {len(all_objects)}")


# --- Strict Filter (Updated Area Threshold) ---
print("Applying strict filter to remove small segmented objects...")
filtered_binary_masks = []
filtered_per_frame_labeled_masks = []

for frame_idx, labeled_mask in enumerate(per_frame_labeled_masks):
    props = measure.regionprops(labeled_mask)
    keep_labels = [prop.label for prop in props if prop.area >= 120]
    filtered_mask = np.isin(labeled_mask, keep_labels).astype(np.uint16) * labeled_mask
    filtered_binary_masks.append(filtered_mask > 0)
    filtered_per_frame_labeled_masks.append(measure.label(filtered_mask > 0))

binary_masks = filtered_binary_masks
per_frame_labeled_masks = filtered_per_frame_labeled_masks
# --- GHOST-AWARE SEGMENTATION LABEL TRACKING ---
# Relabel segmented masks over time using ghost recovery to reduce flickering

from skimage.measure import regionprops, label
from scipy.spatial.distance import cdist
from collections import defaultdict

fallback_search_radius = 20
search_radius=7
max_ghost_gap = 4              # maximum frame gap to allow recovery
max_centroid_distance = 15     # maximum distance to consider for relinking
iou_threshold = 0.15            # minimum IoU for spatial match

# Initialize label map
segmentation_labels = np.zeros((frames, height, width), dtype=np.uint16)
next_label_id = 1

# Keep track of object metadata
active_tracks = {}  # tid -> {'centroid': (y,x), 'mask': mask, 'last_frame': f}

# --- PATCHED GHOST-AWARE LABEL REASSIGNMENT LOGIC ---
# Replace the loop that starts with `for region in current_props:` inside your ghost-aware tracking block

# Frame-by-frame tracking with ghost recovery + fallback ID reassignment
for f in range(frames):
    current_mask = per_frame_labeled_masks[f]
    current_props = regionprops(current_mask)

    matched = set()
    used_ids = set()
    label_assignment = {}

    if f == 0:
        for region in current_props:
            segmentation_labels[f][current_mask == region.label] = next_label_id
            active_tracks[next_label_id] = {
                'centroid': region.centroid,
                'mask': current_mask == region.label,
                'last_frame': f
            }
            next_label_id += 1
        continue

    # Prepare ghost candidates from previous tracks
    ghost_candidates = {
        tid: track for tid, track in active_tracks.items()
        if f - track['last_frame'] <= max_ghost_gap
    }

    # Match current objects to ghosts
    for region in current_props:
        c_yx = region.centroid
        best_tid = None
        best_score = 0

        for tid, track in ghost_candidates.items():
            if tid in used_ids:
                continue

            prev_yx = track['centroid']
            dist = np.linalg.norm(np.array(c_yx) - np.array(prev_yx))
            if dist > max_centroid_distance:
                continue

            # Compute IoU
            overlap = np.logical_and(track['mask'], current_mask == region.label)
            union = np.logical_or(track['mask'], current_mask == region.label)
            iou = np.sum(overlap) / np.sum(union)

            if iou > iou_threshold and iou > best_score:
                best_tid = tid
                best_score = iou

        if best_tid is not None:
            label_assignment[region.label] = best_tid
            active_tracks[best_tid] = {
                'centroid': region.centroid,
                'mask': current_mask == region.label,
                'last_frame': f
            }
            matched.add(region.label)
            used_ids.add(best_tid)

    # Assign new labels for unmatched objects with fallback reuse from previous frame
    for region in current_props:
        if region.label in matched:
            assigned_id = label_assignment[region.label]
            segmentation_labels[f][current_mask == region.label] = assigned_id
        else:
            fallback_label = None
            if f > 0:
                prev_labels = segmentation_labels[f - 1]
                y, x = map(int, region.centroid)
                y_min, y_max = max(0, y - 5), min(height, y + 6)
                x_min, x_max = max(0, x - 5), min(width, x + 6)
                local_prev = prev_labels[y_min:y_max, x_min:x_max]
                candidate_labels, counts = np.unique(local_prev[local_prev > 0], return_counts=True)

                if len(candidate_labels) > 0:
                    sorted_indices = np.argsort(-counts)
                    for idx in sorted_indices:
                        label_candidate = candidate_labels[idx]
                        if label_candidate not in used_ids:
                            fallback_label = label_candidate
                            break

            assigned_id = fallback_label if fallback_label else next_label_id
            segmentation_labels[f][current_mask == region.label] = assigned_id
            active_tracks[assigned_id] = {
                'centroid': region.centroid,
                'mask': current_mask == region.label,
                'last_frame': f
            }
            if fallback_label is None or fallback_label in used_ids:
                assigned_id = next_label_id
                next_label_id += 1
            else:
                assigned_id = fallback_label
                used_ids.add(fallback_label)



print("Tracking objects frame-to-frame (smarter motion-aware gap closing)...")

tracks = []
next_track_id = 0
active_tracks = {}
last_seen = {}

border_margin = 5
min_jump_distance = 10  # minimum meaningful movement to allow interpolation
max_merge_angle_deg = 30  # stricter turn constraint
max_pred_error = 20  # max error allowed for prediction-based matching

def angle_between(v1, v2):
    unit_v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    unit_v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    angle_rad = np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0))
    return np.degrees(angle_rad)

# Initialize tracks at first frame
for idx, (x, y) in enumerate(frame_objects.get(0, [])):
    if border_margin < x < width - border_margin and border_margin < y < height - border_margin:
        active_tracks[next_track_id] = [(0, x, y)]
        last_seen[next_track_id] = 0
        next_track_id += 1

# Smart frame-to-frame linking
for f in range(1, frames):
    detections = frame_objects.get(f, np.array([]))
    if len(detections) == 0:
        continue

    active_positions = []
    track_ids = []
    track_last_frames = []
    track_velocities = []

    for tid, track in active_tracks.items():
        if f - last_seen[tid] <= max_frame_gap:
            if len(track) >= 2:
                dx = track[-1][1] - track[-2][1]
                dy = track[-1][2] - track[-2][2]
                track_velocities.append((dx, dy))
            else:
                track_velocities.append((0.0, 0.0))
            active_positions.append(track[-1][1:])
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

    unmatched_detections = set(range(len(detections)))
    matched_tracks = {}

    for det_idx, det in enumerate(detections):
        dist, nn_idx = tree.query(det)
        if dist == np.inf:
            continue

        track_id = track_ids[nn_idx]
        frame_gap = f - track_last_frames[nn_idx]
        adaptive_distance = max_tracking_distance + (frame_gap - 1) * 10

        # Predict next position (simple velocity prediction)
        predicted_x = active_positions[nn_idx][0] + track_velocities[nn_idx][0] * frame_gap
        predicted_y = active_positions[nn_idx][1] + track_velocities[nn_idx][1] * frame_gap
        pred_error = np.linalg.norm(np.array([det[0] - predicted_x, det[1] - predicted_y]))

        if dist <= adaptive_distance and pred_error <= max_pred_error:
            # Check direction coherence if enough movement
            if len(active_tracks[track_id]) >= 2:
                prev_x, prev_y = active_tracks[track_id][-2][1:]
                last_x, last_y = active_tracks[track_id][-1][1:]
                dir_prev = np.array([last_x - prev_x, last_y - prev_y])
                dir_next = np.array([det[0] - last_x, det[1] - last_y])

                if np.linalg.norm(dir_prev) > min_jump_distance:
                    angle = angle_between(dir_prev, dir_next)
                    if angle > max_merge_angle_deg:
                        continue  # Sharp turn, probably wrong

            # Interpolate if necessary
            if frame_gap > 1:
                start_x, start_y = active_tracks[track_id][-1][1:]
                for interp_frame in range(last_seen[track_id] + 1, f):
                    frac = (interp_frame - last_seen[track_id]) / frame_gap
                    interp_x = start_x + frac * (det[0] - start_x)
                    interp_y = start_y + frac * (det[1] - start_y)
                    active_tracks[track_id].append((interp_frame, interp_x, interp_y))

            active_tracks[track_id].append((f, det[0], det[1]))
            last_seen[track_id] = f
            matched_tracks[det_idx] = track_id
            unmatched_detections.discard(det_idx)

    for det_idx in unmatched_detections:
        det = detections[det_idx]
        if border_margin < det[0] < width - border_margin and border_margin < det[1] < height - border_margin:
            active_tracks[next_track_id] = [(f, det[0], det[1])]
            last_seen[next_track_id] = f
            next_track_id += 1

tracks = [v for v in active_tracks.values() if len(v) >= min_track_length]
print(f"Initial tracks with motion-aware gap closing: {len(tracks)}")





# --- SMART TRACK QUALITY CONTROL ---
print("Applying smart track filtering...")

max_allowed_velocity = 8  # pixels/sec
min_total_displacement = 5  # pixels

filtered_tracks = []
for track in tracks:
    coords = np.array([[x, y] for _, x, y in track])
    frames_track = np.array([frame for frame, _, _ in track])

    if len(coords) < 2:
        continue

    distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    mean_velocity = np.mean(distances) / frame_interval_sec
    net_disp = np.linalg.norm(coords[-1] - coords[0])

    if mean_velocity <= max_allowed_velocity and net_disp >= min_total_displacement:
        filtered_tracks.append(track)

tracks = filtered_tracks
print(f"Tracks after smart filtering: {len(tracks)}")

# --- STRONGER RE-MERGING OF TRACKS ---
print("Re-merging fragmented tracks (smart)...")

final_tracks = []
used = set()

for i, track1 in enumerate(tracks):
    if i in used:
        continue
    for j, track2 in enumerate(tracks):
        if i != j and j not in used:
            end_frame, end_x, end_y = track1[-1]
            start_frame, start_x, start_y = track2[0]
            if 0 < (start_frame - end_frame) <= 2:
                distance = np.linalg.norm([start_x - end_x, start_y - end_y])
                if distance <= 60:
                    track1.extend(track2)
                    used.add(j)
    final_tracks.append(track1)

tracks = final_tracks
print(f"Final tracks after smart re-merging: {len(tracks)}")

# --- GENTLE TRACK FILTERING + STALLED TRACK REMOVAL ---
print("Filtering short and stalled tracks...")
min_total_duration = 5  # frames
max_stall_duration = 6  # frames with zero movement

filtered_tracks = []
for track in tracks:  # was motion_tracks (undefined)
    motion_tracks = filtered_tracks

    frame_numbers = [f for f, _, _ in track]
    coords = np.array([[x, y] for _, x, y in track])

    if len(frame_numbers) < min_total_duration:
        continue

    displacements = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    stall_count = 0
    for d in displacements:
        if d < 1e-2:
            stall_count += 1
        else:
            stall_count = 0
        if stall_count >= max_stall_duration:
            break
    else:
        filtered_tracks.append(track)

motion_tracks = filtered_tracks
print(f"Motion-inferred tracks after gentle filtering: {len(motion_tracks)}")


# --- PATCH: Fix label assignment to reduce flickering ---
# Replace the current relabeling section (disk-based) with overlap-based matching to original segmentation masks

print("Building segmented surfaces from tracked objects (overlap-based relabeling)...")
segmentation_labels_retracked = np.zeros((frames, height, width), dtype=np.uint16)

for tidx, track in enumerate(tracks):
    for frame_idx, x, y in track:
        frame_idx = int(frame_idx)
        xi, yi = int(round(x)), int(round(y))
        labeled_mask = per_frame_labeled_masks[frame_idx]

        # Define a local window
        y_min = max(0, yi - search_radius)
        y_max = min(height, yi + search_radius + 1)
        x_min = max(0, xi - search_radius)
        x_max = min(width, xi + search_radius + 1)

        local_patch = labeled_mask[y_min:y_max, x_min:x_max]
        label_ids, counts = np.unique(local_patch[local_patch > 0], return_counts=True)

        if len(label_ids) > 0:
            best_label = label_ids[np.argmax(counts)]
            segmentation_labels_retracked[frame_idx][labeled_mask == best_label] = tidx + 1
        else:
            # If no overlap match, assign nearest label by centroid proximity
            props = measure.regionprops(labeled_mask)
            min_dist = float('inf')
            best_label = None
            for p in props:
                dist = np.linalg.norm(np.array([yi, xi]) - np.array(p.centroid[::-1]))
                if dist < min_dist:
                    min_dist = dist
                    best_label = p.label
            if best_label:
                segmentation_labels_retracked[frame_idx][labeled_mask == best_label] = tidx + 1



# --- Save Tracks to CSV ---
print("Saving tracks to CSV...")
track_list = []
for tidx, track in enumerate(tracks):
    for frame, x, y in track:
        track_list.append([tidx, frame, x, y])

track_df = pd.DataFrame(track_list, columns=["track_id", "frame", "x", "y"])
track_df.to_csv(os.path.join(output_dir, f"{base_filename}_tracks.csv"), index=False)


# --- VELOCITY PLOT ---
print("Generating velocity plot...")
velocity_per_frame = {}

for track in motion_tracks:
    coords = np.array([[x, y] for _, x, y in track])
    frames_track = np.array([frame for frame, _, _ in track])
    if len(coords) >= 2:
        distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        frames_mid = (frames_track[:-1] + frames_track[1:]) // 2
        for idx, d in enumerate(distances):
            frame_idx = frames_mid[idx]
            velocity_per_frame.setdefault(frame_idx, []).append(d / frame_interval_sec)

all_frames = sorted(velocity_per_frame.keys())
velocities_mean = [np.mean(velocity_per_frame[f]) for f in all_frames]
velocities_sem = [np.std(velocity_per_frame[f]) / np.sqrt(len(velocity_per_frame[f])) for f in all_frames]

plt.figure()
plt.plot(np.array(all_frames) * frame_interval_sec, velocities_mean)
plt.fill_between(np.array(all_frames) * frame_interval_sec,
                 np.array(velocities_mean) - 1.96 * np.array(velocities_sem),
                 np.array(velocities_mean) + 1.96 * np.array(velocities_sem), alpha=0.3)
plt.xlabel("Time (sec)")
plt.ylabel("Velocity (pixels/sec)")
plt.title("Velocity Over Time")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{base_filename}_velocity.png"))
plt.close()
# --- MOTILITY METRICS: Persistence, Acceleration, Net Displacement ---
print("Generating motility metric plots...")

# --- 1. Directional Persistence Over Time ---
print("Calculating persistence...")
window_size_frames = 5  # You can adjust if you want larger/smaller smoothing
persistence_per_frame = {}

for track in motion_tracks:
    frames_track = np.array([frame for frame, _, _ in track])
    coords = np.array([[x, y] for _, x, y in track])

    if len(coords) < window_size_frames + 1:
        continue

    for i in range(len(coords) - window_size_frames):
        start_frame = frames_track[i]
        end_frame = frames_track[i + window_size_frames]
        start_coord = coords[i]
        end_coord = coords[i + window_size_frames]
        path = coords[i:i + window_size_frames + 1]

        net_disp = np.linalg.norm(end_coord - start_coord)
        path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

        if path_length > 0:
            persistence = net_disp / path_length
            mid_frame = (start_frame + end_frame) // 2
            persistence_per_frame.setdefault(mid_frame, []).append(persistence)

# Plot Persistence
all_frames_persistence = sorted(persistence_per_frame.keys())
persistence_mean = [np.mean(persistence_per_frame[f]) for f in all_frames_persistence]
persistence_sem = [np.std(persistence_per_frame[f]) / np.sqrt(len(persistence_per_frame[f])) for f in all_frames_persistence]

plt.figure()
plt.plot(np.array(all_frames_persistence) * frame_interval_sec, persistence_mean)
plt.fill_between(np.array(all_frames_persistence) * frame_interval_sec,
                 np.array(persistence_mean) - 1.96 * np.array(persistence_sem),
                 np.array(persistence_mean) + 1.96 * np.array(persistence_sem), alpha=0.3)
plt.xlabel("Time (sec)")
plt.ylabel("Directional Persistence")
plt.title("Directional Persistence Over Time")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{base_filename}_persistence.png"))
plt.close()

# --- 2. Acceleration Over Time ---
print("Calculating acceleration...")
frame_times = np.array(all_frames) * frame_interval_sec
velocities_mean_arr = np.array(velocities_mean)

# Calculate discrete derivative (velocity change over time)
accelerations = np.diff(velocities_mean_arr) / np.diff(frame_times)
frame_times_acc = (frame_times[:-1] + frame_times[1:]) / 2  # Midpoints

plt.figure()
plt.plot(frame_times_acc, accelerations)
plt.xlabel("Time (sec)")
plt.ylabel("Acceleration (pixels/sec²)")
plt.title("Acceleration Over Time")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{base_filename}_acceleration.png"))
plt.close()

# --- 3. Net Displacement Over Time ---
print("Calculating net displacement...")
displacement_per_frame = {}

for track in motion_tracks:
    frames_track = np.array([frame for frame, _, _ in track])
    coords = np.array([[x, y] for _, x, y in track])

    if len(coords) < 2:
        continue

    start_coord = coords[0]

    for idx in range(len(coords)):
        frame_idx = frames_track[idx]
        current_coord = coords[idx]
        net_disp = np.linalg.norm(current_coord - start_coord)
        displacement_per_frame.setdefault(frame_idx, []).append(net_disp)

all_frames_disp = sorted(displacement_per_frame.keys())
displacement_mean = [np.mean(displacement_per_frame[f]) for f in all_frames_disp]
displacement_sem = [np.std(displacement_per_frame[f]) / np.sqrt(len(displacement_per_frame[f])) for f in all_frames_disp]

plt.figure()
plt.plot(np.array(all_frames_disp) * frame_interval_sec, displacement_mean)
plt.fill_between(np.array(all_frames_disp) * frame_interval_sec,
                 np.array(displacement_mean) - 1.96 * np.array(displacement_sem),
                 np.array(displacement_mean) + 1.96 * np.array(displacement_sem), alpha=0.3)
plt.xlabel("Time (sec)")
plt.ylabel("Net Displacement (pixels)")
plt.title("Net Displacement Over Time")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{base_filename}_netdisplacement.png"))
plt.close()

print("✅ Motility metrics generated successfully.")
import numpy as np
import pandas as pd
import os
from scipy.spatial import cKDTree
from tqdm import tqdm

# --- PARAMETERS ---
ghost_segmentation = segmentation_labels.copy()  # From original ghost-aware relabeling
min_track_length = 3
frame_interval_sec = 30
border_margin = 5
max_tracking_gap = 3
min_total_displacement = 5
max_stall_duration = 6
max_velocity = 8  # pixels/sec

# --- GHOST-LABEL BASED OBJECT CENTROIDS ---
print("Extracting ghost-label centroids...")
ghost_tracks_raw = {}  # tid -> list of (frame, x, y)

for f in range(ghost_segmentation.shape[0]):
    labels_in_frame = np.unique(ghost_segmentation[f])
    labels_in_frame = labels_in_frame[labels_in_frame > 0]
    for label_id in labels_in_frame:
        mask = ghost_segmentation[f] == label_id
        if np.count_nonzero(mask) == 0:
            continue
        y, x = np.argwhere(mask).mean(axis=0)
        ghost_tracks_raw.setdefault(label_id, []).append((f, x, y))

# Convert dict to list of tracks
ghost_tracks = list(ghost_tracks_raw.values())
print(f"Initial ghost tracks extracted: {len(ghost_tracks)}")

# --- QUALITY CONTROL + MERGING ---
print("Filtering ghost tracks by motion and continuity...")

# Remove short tracks
ghost_tracks = [trk for trk in ghost_tracks if len(trk) >= min_track_length]

# Filter tracks by stall + displacement
filtered_ghost_tracks = []
for track in ghost_tracks:
    frames = [f for f, _, _ in track]
    coords = np.array([[x, y] for _, x, y in track])

    if len(coords) < 2:
        continue

    # Net displacement
    net_disp = np.linalg.norm(coords[-1] - coords[0])
    if net_disp < min_total_displacement:
        continue

    # Stall detection
    displacements = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    stall_count = 0
    for d in displacements:
        if d < 1e-2:
            stall_count += 1
        else:
            stall_count = 0
        if stall_count >= max_stall_duration:
            break
    else:
        filtered_ghost_tracks.append(track)

ghost_tracks = filtered_ghost_tracks
print(f"Tracks after filtering: {len(ghost_tracks)}")

# --- OPTIONAL: Merge nearby track fragments ---
print("Merging ghost track fragments...")
final_ghost_tracks = []
used = set()

for i, t1 in enumerate(ghost_tracks):
    if i in used:
        continue
    merged = list(t1)
    for j, t2 in enumerate(ghost_tracks):
        if j <= i or j in used:
            continue
        f1, x1, y1 = merged[-1]
        f2, x2, y2 = t2[0]
        if 0 < (f2 - f1) <= max_tracking_gap:
            if np.linalg.norm([x2 - x1, y2 - y1]) <= 60:
                merged.extend(t2)
                used.add(j)
    final_ghost_tracks.append(merged)

ghost_tracks = final_ghost_tracks
print(f"Final merged ghost tracks: {len(ghost_tracks)}")

# --- EXPORT ---
print("Saving ghost-label-based tracks...")
ghost_track_data = []
for tidx, track in enumerate(ghost_tracks):
    for f, x, y in track:
        ghost_track_data.append([tidx, f, x, y])

ghost_track_df = pd.DataFrame(ghost_track_data, columns=["track_id", "frame", "x", "y"])
ghost_csv_path = os.path.join(output_dir, f"{base_filename}_ghosttracks.csv")
ghost_track_df.to_csv(ghost_csv_path, index=False)
# --- GHOST TRACK METRICS + PLOTS ---
print("Generating ghost-based motility metrics and overlay...")

# Reuse same logic as before, applied to ghost_tracks

# 1. Velocity
ghost_velocity_per_frame = {}
for track in ghost_tracks:
    coords = np.array([[x, y] for _, x, y in track])
    frames_track = np.array([frame for frame, _, _ in track])
    if len(coords) >= 2:
        distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        frames_mid = (frames_track[:-1] + frames_track[1:]) // 2
        for idx, d in enumerate(distances):
            frame_idx = frames_mid[idx]
            ghost_velocity_per_frame.setdefault(frame_idx, []).append(d / frame_interval_sec)

all_frames_ghost = sorted(ghost_velocity_per_frame.keys())
ghost_velocities_mean = [np.mean(ghost_velocity_per_frame[f]) for f in all_frames_ghost]
ghost_velocities_sem = [np.std(ghost_velocity_per_frame[f]) / np.sqrt(len(ghost_velocity_per_frame[f])) for f in all_frames_ghost]

plt.figure()
plt.plot(np.array(all_frames_ghost) * frame_interval_sec, ghost_velocities_mean)
plt.fill_between(np.array(all_frames_ghost) * frame_interval_sec,
                 np.array(ghost_velocities_mean) - 1.96 * np.array(ghost_velocities_sem),
                 np.array(ghost_velocities_mean) + 1.96 * np.array(ghost_velocities_sem), alpha=0.3)
plt.xlabel("Time (sec)")
plt.ylabel("Velocity (pixels/sec)")
plt.title("Ghost-Based Velocity Over Time")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{base_filename}_ghost_velocity.png"))
plt.close()

# 2. Persistence
print("Calculating ghost-based persistence...")
ghost_persistence_per_frame = {}
for track in ghost_tracks:
    frames_track = np.array([frame for frame, _, _ in track])
    coords = np.array([[x, y] for _, x, y in track])

    if len(coords) < window_size_frames + 1:
        continue

    for i in range(len(coords) - window_size_frames):
        start_frame = frames_track[i]
        end_frame = frames_track[i + window_size_frames]
        path = coords[i:i + window_size_frames + 1]
        net_disp = np.linalg.norm(coords[i + window_size_frames] - coords[i])
        path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

        if path_length > 0:
            persistence = net_disp / path_length
            mid_frame = (start_frame + end_frame) // 2
            ghost_persistence_per_frame.setdefault(mid_frame, []).append(persistence)

frames_ghost_persistence = sorted(ghost_persistence_per_frame.keys())
ghost_persistence_mean = [np.mean(ghost_persistence_per_frame[f]) for f in frames_ghost_persistence]
ghost_persistence_sem = [np.std(ghost_persistence_per_frame[f]) / np.sqrt(len(ghost_persistence_per_frame[f])) for f in frames_ghost_persistence]

plt.figure()
plt.plot(np.array(frames_ghost_persistence) * frame_interval_sec, ghost_persistence_mean)
plt.fill_between(np.array(frames_ghost_persistence) * frame_interval_sec,
                 np.array(ghost_persistence_mean) - 1.96 * np.array(ghost_persistence_sem),
                 np.array(ghost_persistence_mean) + 1.96 * np.array(ghost_persistence_sem), alpha=0.3)
plt.xlabel("Time (sec)")
plt.ylabel("Directional Persistence")
plt.title("Ghost-Based Directional Persistence")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{base_filename}_ghost_persistence.png"))
plt.close()

# 3. Acceleration
print("Calculating ghost-based acceleration...")
frame_times_ghost = np.array(all_frames_ghost) * frame_interval_sec
vel_arr_ghost = np.array(ghost_velocities_mean)
ghost_acc = np.diff(vel_arr_ghost) / np.diff(frame_times_ghost)
acc_frame_mid = (frame_times_ghost[:-1] + frame_times_ghost[1:]) / 2

plt.figure()
plt.plot(acc_frame_mid, ghost_acc)
plt.xlabel("Time (sec)")
plt.ylabel("Acceleration (pixels/sec²)")
plt.title("Ghost-Based Acceleration")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{base_filename}_ghost_acceleration.png"))
plt.close()

# 4. Net Displacement
print("Calculating ghost-based displacement...")
ghost_disp_per_frame = {}
for track in ghost_tracks:
    coords = np.array([[x, y] for _, x, y in track])
    frames = np.array([f for f, _, _ in track])
    if len(coords) < 2:
        continue
    for i in range(len(coords)):
        disp = np.linalg.norm(coords[i] - coords[0])
        ghost_disp_per_frame.setdefault(frames[i], []).append(disp)

frames_disp = sorted(ghost_disp_per_frame.keys())
disp_mean = [np.mean(ghost_disp_per_frame[f]) for f in frames_disp]
disp_sem = [np.std(ghost_disp_per_frame[f]) / np.sqrt(len(ghost_disp_per_frame[f])) for f in frames_disp]

plt.figure()
plt.plot(np.array(frames_disp) * frame_interval_sec, disp_mean)
plt.fill_between(np.array(frames_disp) * frame_interval_sec,
                 np.array(disp_mean) - 1.96 * np.array(disp_sem),
                 np.array(disp_mean) + 1.96 * np.array(disp_sem), alpha=0.3)
plt.xlabel("Time (sec)")
plt.ylabel("Net Displacement (pixels)")
plt.title("Ghost-Based Net Displacement")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{base_filename}_ghost_displacement.png"))
plt.close()

print(f"✅ Ghost tracks saved to: {ghost_csv_path}")

# --- ROSE PLOT ---
print("Generating rose plot...")
angles = []
for track in motion_tracks:
    coords = np.array([[x, y] for _, x, y in track])
    if len(coords) >= 2:
        deltas = np.diff(coords, axis=0)
        angles.extend(np.arctan2(deltas[:, 1], deltas[:, 0]))

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.hist(angles, bins=32, density=True)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
plt.title("Movement Direction Rose Plot")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{base_filename}_roseplot.png"))
plt.close()

# --- ROSE PLOT BASED ON GHOST TRACKS ---
print("Generating ghost-based rose plot...")
ghost_angles = []

for track in ghost_tracks:
    coords = np.array([[x, y] for _, x, y in track])
    if len(coords) >= 2:
        deltas = np.diff(coords, axis=0)
        ghost_angles.extend(np.arctan2(deltas[:, 1], deltas[:, 0]))  # dy, dx

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.hist(ghost_angles, bins=32, density=True, alpha=0.9)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
plt.title("Ghost-Inferred Movement Direction Rose Plot")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{base_filename}_ghost_roseplot.png"))
plt.close()

# --- EXPORT GHOST METRICS TO CSV ---  <<<< INSERT START
print("Saving ghost metrics to CSV for downstream combination...")

ghost_metrics = pd.DataFrame({
    "frame": all_frames_ghost,
    "time_sec": np.array(all_frames_ghost) * frame_interval_sec,
    "velocity": ghost_velocities_mean,
    "velocity_sem": ghost_velocities_sem
})
ghost_metrics["displacement"] = pd.Series(disp_mean, index=frames_disp)
ghost_metrics["displacement_sem"] = pd.Series(disp_sem, index=frames_disp)

ghost_metrics_csv = os.path.join(output_dir, f"{base_filename}_ghost_metrics.csv")
ghost_metrics.to_csv(ghost_metrics_csv, index=False)

ghost_angles_csv = os.path.join(output_dir, f"{base_filename}_ghost_angles.csv")
pd.DataFrame({"angle_rad": ghost_angles}).to_csv(ghost_angles_csv, index=False)

print(f"📊 Exported ghost metrics CSV: {ghost_metrics_csv}")
print(f"📐 Exported ghost angles CSV: {ghost_angles_csv}")

# --- EXPORT GHOST METRICS TO CSV ---  <<<< INSERT END
# --- TRACKS OVERLAY PLOT ---
print("Generating tracks overlay image...")
background_proj = np.max(green_normalized, axis=0)
colors = cm.get_cmap('tab20', len(tracks))

# Motion tracks overlay
plt.figure(figsize=(8, 8))
plt.imshow(background_proj, cmap='gray')
for tidx, track in enumerate(motion_tracks + ghost_tracks):
    coords = np.array([[x, y] for _, x, y in track])
    color = colors(tidx % 20)
    plt.plot(coords[:, 0], coords[:, 1], linewidth=1, color=color)
plt.axis('off')
plt.title('Motion Tracks Overlay')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{base_filename}_motion_overlay.png"))
plt.close()

# Ghost tracks overlay
plt.figure(figsize=(8, 8))
plt.imshow(background_proj, cmap='gray')
for tidx, track in enumerate(ghost_tracks):
    coords = np.array([[x, y] for _, x, y in track])
    color = colors(tidx % 20)
    plt.plot(coords[:, 0], coords[:, 1], linewidth=1, color=color)
plt.axis('off')
plt.title('Ghost Tracks Overlay')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{base_filename}_ghost_overlay.png"))
plt.close()


if make_movie:
    print("Generating ghost-like movie with red background and evolving tracks (Napari-matched)...")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')

    # Precompute ghost track data like Napari
    ghost_track_data_arr = np.array([
        [tidx, frame, y, x]
        for tidx, track in enumerate(ghost_tracks)
        for frame, x, y in track
    ])

    colors = cm.get_cmap('tab20', len(ghost_tracks))

    def update_overlay(frame_idx):
        ax.clear()
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis('off')

        # Red channel background as magenta
        red_img = red_normalized[frame_idx]
        magenta_rgb = np.stack([red_img, np.zeros_like(red_img), red_img], axis=-1)
        ax.imshow(magenta_rgb, interpolation='none')

        # Ghost segmentation overlay
        ax.imshow(ghost_segmentation[frame_idx], cmap='tab20', alpha=0.4, interpolation='none')


        # Plot ghost tracks up to current frame (Napari-style)
        frame_data = ghost_track_data_arr[ghost_track_data_arr[:, 1] <= frame_idx]
        for tid in np.unique(frame_data[:, 0]):
            track_pts = frame_data[frame_data[:, 0] == tid][:, [3, 2]]  # x, y
            ax.plot(track_pts[:, 0], track_pts[:, 1], linewidth=1, color=colors(int(tid) % 20))

        ax.set_title(f'Frame {frame_idx} ({frame_idx * frame_interval_sec} sec)')

    ani = animation.FuncAnimation(
        fig, update_overlay,
        frames=green_normalized.shape[0],
        interval=1000 / fps_movie,
        blit=False,
        repeat=False
    )

    out_filename = f"{base_filename}_ghost_overlay"
    try:
        if ffmpeg_path and movie_format_tracks.lower() != 'gif':
            writer = animation.FFMpegWriter(fps=fps_movie, metadata={'artist': 'Ghost-Inferred'}, bitrate=2000)
            ani.save(os.path.join(output_dir, out_filename + ".mp4"), writer=writer)
            print(f"🎥 MP4 movie saved: {out_filename}.mp4")
        else:
            gif_path = os.path.join(output_dir, out_filename + ".gif")
            ani.save(gif_path, writer='pillow', fps=fps_movie)
            print(f"🎥 GIF movie saved: {out_filename}.gif")
    except Exception as e:
        print(f"⚠️ Movie generation failed: {e}")

    plt.close(fig)




# --- MINIMAL NAPARI VISUALIZATION ---
print("Launching simplified napari viewer (ghost-only)...")
viewer = napari.Viewer()

# Original image channels
viewer.add_image(
    green_normalized,
    name='Green Channel',
    contrast_limits=[0, 1],
    colormap='green',
    gamma=0.5
)

viewer.add_image(
    red_normalized,
    name='Red Channel',
    contrast_limits=[0, 1],
    colormap='magenta',
    opacity=0.8
)

# Ghost-inferred segmentation
viewer.add_labels(
    segmentation_labels,
    name='Ghost-Inferred Labels'
)

# Ghost-inferred tracks
ghost_track_data_napari = []
for tidx, track in enumerate(ghost_tracks):
    for frame, x, y in track:
        ghost_track_data_napari.append([tidx, frame, y, x])
ghost_track_data_napari = np.array(ghost_track_data_napari)
viewer.add_tracks(
    ghost_track_data_napari,
    name='Ghost-Inferred Tracks',
    tail_length=150,
    tail_width=5,
    colormap='turbo'
)
# Set zoom level
viewer.camera.zoom = .9

from qtpy.QtCore import QTimer
import tempfile
import imageio

n_frames = green_normalized.shape[0]
napari_movie_fps = fps_movie  # match with your earlier setting

def capture_napari_movie():
    print("🎥 Capturing Napari viewer screenshots...")

    temp_dir = tempfile.mkdtemp()
    frame_paths = []

    for t in tqdm(range(n_frames), desc="Capturing Napari frames"):
        viewer.dims.set_current_step(0, t)
        viewer.window._qt_viewer.canvas.native.repaint()
        screenshot = viewer.screenshot(canvas_only=True)
        frame_path = os.path.join(temp_dir, f"frame_{t:04d}.png")
        imageio.imwrite(frame_path, screenshot)
        frame_paths.append(frame_path)

    output_path = os.path.join(output_dir, f"{base_filename}_napari_direct.{movie_format_tracks}")
    with imageio.get_writer(output_path, fps=napari_movie_fps) as writer:
        for path in frame_paths:
            img = imageio.imread(path)
            writer.append_data(img)

    print(f"✅ Napari movie saved to: {output_path}")


QTimer.singleShot(1000, capture_napari_movie)
napari.run()



 
print(f"✅ All done! Full results saved to: {output_dir}")