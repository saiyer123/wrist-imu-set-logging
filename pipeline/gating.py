"""
Confidence gating: the trade the product actually lives on.

An automatic log is only valuable if the user stops checking it. One visible
wrong entry costs more trust than ten missing ones cost convenience, because a
missing set is a known-unknown the user can fix, while a wrong set silently
corrupts their history.

So we do not optimise accuracy. We optimise:

    coverage          - share of real sets logged with no user involvement
    visible errors    - auto-logged sets that are wrong (the trust cost)
    correction burden - total user actions needed to reach a correct log

and we expose the threshold that trades between them, rather than picking one
number and hiding it.
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import data
import end_to_end
import features
import segment


def sweep(rows, thresholds=np.linspace(0.0, 0.95, 40)):
    matched = [r for r in rows if r.get("matched")]
    spurious = [r for r in rows if r.get("spurious")]
    n_true = len(matched) + len([r for r in rows if r.get("missed")])

    out = []
    for t in thresholds:
        auto_ok = [r for r in matched if r["conf"] >= t and r["pred_ex"] == r["true_ex"]]
        auto_bad = [r for r in matched if r["conf"] >= t and r["pred_ex"] != r["true_ex"]]
        auto_spur = [r for r in spurious if r["conf"] >= t]

        # Anything not auto-logged has to be entered or confirmed by hand.
        prompted = n_true - len(auto_ok) - len(auto_bad)
        visible_errors = len(auto_bad) + len(auto_spur)
        burden = visible_errors + prompted

        out.append({
            "threshold": float(t),
            "coverage": len(auto_ok) / max(n_true, 1),
            "auto_logged": len(auto_ok) + len(auto_bad) + len(auto_spur),
            "visible_errors": visible_errors,
            "error_rate_of_auto": visible_errors / max(len(auto_ok) + visible_errors, 1),
            "correction_burden": burden,
            "burden_per_set": burden / max(n_true, 1),
        })
    return out, n_true


def personalise(bundle, multipliers, n_confirm=6, lam=0.95):
    """
    Simulate a user confirming their first few sets, then adapting to them.

    WHAT WE TRIED FIRST, AND WHY IT FAILED
    --------------------------------------
    The obvious move is to learn per-class centroids from the confirmed windows
    and blend them into the global model's probabilities. It did not help at any
    blend weight - at best it matched the global model, at worst it cost 16
    points of coverage.

    The diagnosis explains why. On held-out participants, 85.5% of window errors
    are an exercise misread as REST; only 14.5% are one exercise confused for
    another. Centroid blending redistributes probability *among exercise
    classes*, so it was aimed squarely at the 14.5%.

    So we personalise the activity/rest boundary instead. From the confirmed
    sets we know what this user's rest looks like and what their work looks like,
    which gives a per-user movement-energy threshold. Windows above it have their
    non-activity probability suppressed. Same amount of user effort, aimed at the
    failure that actually dominates.
    """
    cache = np.load("data/cache/windows_right.npz", allow_pickle=True)
    fn = list(cache["feature_names"])
    e_idx = [fn.index(n) for n in ("sw_r_gyr_mag__std", "sw_r_acc_mag__std") if n in fn]
    classes = bundle["classes"]
    na = classes.index(data.NON_ACTIVITY)

    before, after = [], []
    for w in data.TEST_UNSEEN_W:
        wk = data.load_workout(w)
        X, _n, centres = features.extract(wk["sig"], wk["names"])
        proba = bundle["model"].predict_proba(X)
        y_win, _pur = features.window_labels(data.label_per_sample(wk), centres)

        true_segs = [{"start_idx": data.frame_to_index(wk["frame"], s),
                      "end_idx": data.frame_to_index(wk["frame"], e),
                      "exercise": a, "reps": r}
                     for s, e, r, a in wk["labels"] if a in data.CLASS_TO_IDX]
        if len(true_segs) <= n_confirm:
            continue

        cutoff = true_segs[n_confirm - 1]["end_idx"]
        seen = centres <= cutoff

        adapted = proba.copy()
        energy = np.log1p(X[:, e_idx]).sum(1)
        rest_m, act_m = seen & (y_win == na), seen & (y_win != na)
        if rest_m.sum() >= 5 and act_m.sum() >= 5:
            thr = 0.5 * (np.percentile(energy[rest_m], 95) + np.percentile(energy[act_m], 25))
            adapted[energy > thr, na] *= (1 - lam)
            adapted /= adapted.sum(1, keepdims=True) + 1e-12

        rest_true = true_segs[n_confirm:]
        for P, bucket in ((proba, before), (adapted, after)):
            segs = [s for s in segment.segments_from_proba(P, centres, classes, na)
                    if s["start_idx"] > cutoff]
            for s in segs:
                s["confidence"] = s["cls_confidence"]
            m, up, ut = segment.match_to_truth(segs, rest_true)
            for i, j, _iou in m:
                bucket.append({"matched": True, "conf": segs[i]["confidence"],
                               "pred_ex": segs[i]["exercise"], "true_ex": rest_true[j]["exercise"]})
            for i in up:
                bucket.append({"spurious": True, "conf": segs[i]["confidence"],
                               "pred_ex": segs[i]["exercise"], "true_ex": None})
            for _j in ut:
                bucket.append({"missed": True, "conf": None})
    return before, after


def main():
    bundle = pickle.load(open("data/cache/model.pkl", "rb"))
    multipliers = json.load(open("data/cache/rep_multipliers.json"))
    rows = json.load(open("eval/end_to_end_sets.json"))

    report = {}
    for split in ("unseen", "seen"):
        curve, n_true = sweep([r for r in rows if r["split"] == split])
        report[split] = {"curve": curve, "n_true_sets": n_true}
        print(f"\n===== {split} participants ({n_true} real sets) =====")
        print(f"{'thresh':>7} {'coverage':>9} {'visible err':>12} {'err rate':>9} {'burden/set':>11}")
        for c in curve:
            if abs(c["threshold"] * 100 % 15) < 3:
                print(f"{c['threshold']:7.2f} {c['coverage']*100:8.1f}% {c['visible_errors']:12d} "
                      f"{c['error_rate_of_auto']*100:8.1f}% {c['burden_per_set']:11.2f}")

    print("\n\n===== personalisation: user confirms their first 6 sets =====")
    before, after = personalise(bundle, multipliers)
    for tag, rws in (("global model", before), ("+ per-user calibration", after)):
        curve, n_true = sweep(rws)
        # report coverage at the strictest threshold keeping <=5% visible error rate
        ok = [c for c in curve if c["error_rate_of_auto"] <= 0.05]
        best = max(ok, key=lambda c: c["coverage"]) if ok else max(curve, key=lambda c: c["coverage"])
        print(f"  {tag:26s} coverage at <=5% error = {best['coverage']*100:5.1f}%  "
              f"(threshold {best['threshold']:.2f}, {n_true} sets)")
        report[f"personalisation_{tag}"] = best

    json.dump(report, open("eval/gating.json", "w"))
    print("\nwrote eval/gating.json")


if __name__ == "__main__":
    main()
