"""
MM-Fit IMU loading, resampling and splitting.

Design decisions worth stating up front:

1. SINGLE WRIST BY DEFAULT. MM-Fit participants wore a smartwatch on *both*
   wrists. Training on both inflates accuracy but describes a device
   configuration almost nobody has. The default here is the right wrist only;
   `wrists="both"` is available and reported separately as an upper bound.

2. UNIFORM RESAMPLING. Raw sample spacing is irregular (nominal 100 Hz, actual
   ~104 Hz with jitter). Frequency-domain features assume a fixed rate, so every
   stream is linearly resampled onto a common 100 Hz grid before anything else.

3. PARTICIPANT-LEVEL SPLIT. MM-Fit's default test set (w09/w10/w11) reuses
   participants that also appear in training. The `unseen` split holds out
   participants entirely and is the one we headline.
"""
import csv
import os

import numpy as np

TARGET_HZ = 100.0
VIDEO_FPS = 30.0

ACTIONS = [
    "squats", "lunges", "bicep_curls", "situps", "pushups", "tricep_extensions",
    "dumbbell_rows", "jumping_jacks", "dumbbell_shoulder_press",
    "lateral_shoulder_raises",
]
NON_ACTIVITY = "non_activity"
ALL_CLASSES = ACTIONS + [NON_ACTIVITY]
CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}

TRAIN_W = ["01", "02", "03", "04", "06", "07", "08", "16", "17", "18"]
VAL_W = ["14", "15", "19"]
TEST_SEEN_W = ["09", "10", "11"]      # participants also present in training
TEST_UNSEEN_W = ["00", "05", "12", "13", "20"]  # participants held out entirely


def load_labels(path):
    """-> list of (start_frame, end_frame, reps, action)."""
    out = []
    with open(path) as f:
        for row in csv.reader(f):
            if row:
                out.append((int(row[0]), int(row[1]), int(row[2]), row[3]))
    return out


def _resample(arr, grid_t):
    """arr is (N,5) = [frame, unix_ms, x, y, z]. Interpolate xyz + frame onto grid_t."""
    t = arr[:, 1] / 1000.0
    # Timestamps must be strictly increasing for np.interp to behave.
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, arr = t[keep], arr[keep]
    xyz = np.stack([np.interp(grid_t, t, arr[:, c]) for c in (2, 3, 4)], axis=1)
    frame = np.interp(grid_t, t, arr[:, 0])
    return xyz, frame


def load_workout(w_id, root="data/raw/mm-fit", wrists="right"):
    """
    Load one workout, resampled to a common 100 Hz grid.

    Returns dict with:
      t      (T,)   seconds since stream start
      frame  (T,)   video frame index, for mapping labels onto samples
      sig    (T,C)  channels, acc xyz then gyr xyz, per wrist
      names  list of channel names
      labels list of (start_frame, end_frame, reps, action)
    """
    d = os.path.join(root, f"w{w_id}")
    devs = {"right": ["sw_r"], "left": ["sw_l"], "both": ["sw_r", "sw_l"]}[wrists]

    raw = {}
    for dev in devs:
        for sig in ("acc", "gyr"):
            p = os.path.join(d, f"w{w_id}_{dev}_{sig}.npy")
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            raw[f"{dev}_{sig}"] = np.load(p)

    # Common time grid = overlap of every stream, at TARGET_HZ.
    t0 = max(a[0, 1] for a in raw.values()) / 1000.0
    t1 = min(a[-1, 1] for a in raw.values()) / 1000.0
    grid_t = np.arange(t0, t1, 1.0 / TARGET_HZ)

    cols, names, frame = [], [], None
    for key in sorted(raw):  # deterministic channel order
        xyz, fr = _resample(raw[key], grid_t)
        cols.append(xyz)
        names += [f"{key}_{ax}" for ax in "xyz"]
        if frame is None:
            frame = fr

    return {
        "w_id": w_id,
        "t": grid_t - grid_t[0],
        "frame": frame,
        "sig": np.concatenate(cols, axis=1),
        "names": names,
        "labels": load_labels(os.path.join(d, f"w{w_id}_labels.csv")),
    }


def frame_to_index(frame_arr, f):
    """Map a video frame number to the nearest resampled sample index."""
    return int(np.clip(np.searchsorted(frame_arr, f), 0, len(frame_arr) - 1))


def label_per_sample(wk):
    """Dense per-sample class index; everything outside a labelled set is non_activity."""
    y = np.full(len(wk["t"]), CLASS_TO_IDX[NON_ACTIVITY], dtype=np.int16)
    for s, e, _reps, act in wk["labels"]:
        if act not in CLASS_TO_IDX:
            continue
        i0, i1 = frame_to_index(wk["frame"], s), frame_to_index(wk["frame"], e)
        y[i0:i1] = CLASS_TO_IDX[act]
    return y


def available_workouts(root="data/raw/mm-fit"):
    out = []
    for name in sorted(os.listdir(root)):
        if name.startswith("w") and os.path.isdir(os.path.join(root, name)):
            out.append(name[1:])
    return out
