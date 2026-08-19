"""
Turn a stream of per-window class probabilities into discrete, confident sets.

The classifier sees 4 s windows every 1 s and has no notion of a "set". Three
things stand between its output and something a user would accept in a log:

  1. Flicker. A single misclassified window mid-set must not split one set into
     three. We smooth the probability stream before taking an argmax.
  2. Fragmentation. Real sets contain brief pauses. Adjacent same-class segments
     separated by a short gap are merged.
  3. Implausible sets. A "set" lasting 2 s is noise, not a set.

Everything here is a decision the user will eventually see, so each threshold is
named and lives in one place rather than being scattered through the code.
"""
import numpy as np
from scipy import ndimage

HOP_S = 1.0
SMOOTH_WINDOWS = 5      # 5 s of context; shorter left set edges ragged
MERGE_GAP_S = 3.0       # pauses shorter than this are within-set, not between
MIN_SET_S = 8.0         # below this a detection is not a plausible working set
MIN_REPS = 5            # a 'set' of three reps is noise; applied once reps are counted


def smooth_proba(proba, k=SMOOTH_WINDOWS):
    """Moving average over the probability stream (not the argmax)."""
    if len(proba) < k:
        return proba
    kern = np.ones(k) / k
    return np.stack([np.convolve(proba[:, c], kern, mode="same")
                     for c in range(proba.shape[1])], axis=1)


def segments_from_proba(proba, centres, classes, non_activity_idx,
                        merge_gap_s=MERGE_GAP_S, min_set_s=MIN_SET_S):
    """
    -> list of dicts: start/end sample index, class index, mean confidence.
    """
    sm = smooth_proba(proba)
    pred = sm.argmax(1)
    conf = sm.max(1)

    out = []
    for cls in range(len(classes)):
        if cls == non_activity_idx:
            continue
        mask = pred == cls
        if not mask.any():
            continue
        lab, n = ndimage.label(mask)
        spans = []
        for i in range(1, n + 1):
            idx = np.where(lab == i)[0]
            spans.append([idx[0], idx[-1]])

        # merge same-class spans separated by a short gap
        spans.sort()
        merged = [spans[0]]
        for a, b in spans[1:]:
            if (a - merged[-1][1]) * HOP_S <= merge_gap_s:
                merged[-1][1] = b
            else:
                merged.append([a, b])

        for a, b in merged:
            dur = (b - a + 1) * HOP_S
            if dur < min_set_s:
                continue
            out.append({
                "cls": int(cls),
                "exercise": classes[cls],
                "win_start": int(a), "win_end": int(b),
                "start_idx": int(centres[a]), "end_idx": int(centres[b]),
                "duration_s": float(dur),
                "cls_confidence": float(conf[a:b + 1].mean()),
            })

    out.sort(key=lambda s: s["start_idx"])

    # Resolve overlaps between different classes by keeping the more confident.
    resolved = []
    for s in out:
        if resolved and s["start_idx"] < resolved[-1]["end_idx"]:
            if s["cls_confidence"] > resolved[-1]["cls_confidence"]:
                resolved[-1] = s
        else:
            resolved.append(s)
    return resolved


def match_to_truth(pred_segs, true_segs, iou_thresh=0.5):
    """
    Greedy temporal-IoU matching between detected and ground-truth sets.
    -> (matches [(pred_i, true_j, iou)], unmatched_pred, unmatched_true)
    """
    pairs = []
    for i, p in enumerate(pred_segs):
        for j, t in enumerate(true_segs):
            inter = max(0, min(p["end_idx"], t["end_idx"]) - max(p["start_idx"], t["start_idx"]))
            if inter <= 0:
                continue
            union = (max(p["end_idx"], t["end_idx"]) - min(p["start_idx"], t["start_idx"]))
            iou = inter / union if union else 0.0
            if iou >= iou_thresh:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)

    used_p, used_t, matches = set(), set(), []
    for iou, i, j in pairs:
        if i in used_p or j in used_t:
            continue
        used_p.add(i); used_t.add(j)
        matches.append((i, j, iou))
    return (matches,
            [i for i in range(len(pred_segs)) if i not in used_p],
            [j for j in range(len(true_segs)) if j not in used_t])
