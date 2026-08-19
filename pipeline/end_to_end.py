"""End-to-end: raw IMU -> detected sets -> reps -> confidence. No oracle segmentation."""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import data, features, reps, segment

REP_PAD_S = 1.0  # see analyse_workout(); chosen on the validation split


def analyse_workout(w_id, bundle, multipliers, wrists="right"):
    wk = data.load_workout(w_id, wrists=wrists)
    X, _names, centres = features.extract(wk["sig"], wk["names"])
    proba = bundle["model"].predict_proba(X)
    classes = bundle["classes"]
    segs = segment.segments_from_proba(proba, centres, classes,
                                       classes.index(data.NON_ACTIVITY))

    # Grow each detected span by REP_PAD_S before counting. Classification
    # boundaries are systematically tighter than real sets - probability
    # smoothing erodes the edges, so the first and last repetition of a set
    # often fall outside the detected span and go uncounted. Padding is chosen
    # on the VALIDATION split: 1 s takes rep MAE from 1.09 to 0.70 there, while
    # 2 s and beyond starts pulling in rest and makes it worse.
    pad = int(REP_PAD_S * data.TARGET_HZ)
    n_samples = len(wk["t"])
    for s in segs:
        a = max(0, s["start_idx"] - pad)
        b = min(n_samples, s["end_idx"] + pad)
        r = reps.count_reps(wk["sig"][a:b], wk["names"])
        r = reps.apply_multiplier(r, s["exercise"], multipliers)
        s.update(rep_count=r["rep_count"], period_s=r["period_s"],
                 consistency=r["consistency"], rep_confidence=r["confidence"],
                 channel=r["channel"], multiplier=r.get("multiplier", 1.0))
        # A set is only as trustworthy as its weakest stage.
        s["confidence"] = float(min(s["cls_confidence"], 0.35 + 0.65 * r["confidence"]))

    # Plausibility gate, thresholds chosen on the VALIDATION split only: real sets
    # there had a 5th percentile of 10.4 s and 7.4 reps, so 8 s / 5 reps sits well
    # clear of them while removing most spurious detections.
    segs = [s for s in segs if s["rep_count"] >= segment.MIN_REPS]

    true_segs = []
    for st, en, rr, act in wk["labels"]:
        if act not in data.CLASS_TO_IDX:
            continue
        true_segs.append({"start_idx": data.frame_to_index(wk["frame"], st),
                          "end_idx": data.frame_to_index(wk["frame"], en),
                          "exercise": act, "reps": rr})
    return wk, X, proba, centres, segs, true_segs


def main():
    bundle = pickle.load(open("data/cache/model.pkl", "rb"))
    multipliers = json.load(open("data/cache/rep_multipliers.json"))
    rows, all_segs = [], []

    for split, ws in [("unseen", data.TEST_UNSEEN_W), ("seen", data.TEST_SEEN_W)]:
        for w in ws:
            _wk, _X, _p, _c, segs, true_segs = analyse_workout(w, bundle, multipliers)
            m, up, ut = segment.match_to_truth(segs, true_segs)
            for i, j, iou in m:
                rows.append({"split": split, "w": w, "matched": True, "iou": iou,
                             "pred_ex": segs[i]["exercise"], "true_ex": true_segs[j]["exercise"],
                             "pred_reps": segs[i]["rep_count"], "true_reps": true_segs[j]["reps"],
                             "conf": segs[i]["confidence"]})
            for i in up:
                rows.append({"split": split, "w": w, "matched": False, "spurious": True,
                             "pred_ex": segs[i]["exercise"], "true_ex": None,
                             "pred_reps": segs[i]["rep_count"], "true_reps": None,
                             "conf": segs[i]["confidence"]})
            for j in ut:
                rows.append({"split": split, "w": w, "matched": False, "missed": True,
                             "pred_ex": None, "true_ex": true_segs[j]["exercise"],
                             "conf": None})
            all_segs.append({"w": w, "split": split, "n_pred": len(segs), "n_true": len(true_segs)})
            print(f"  w{w} ({split}): {len(segs)} detected / {len(true_segs)} true, "
                  f"{len(m)} matched, {len(up)} spurious, {len(ut)} missed", flush=True)

    json.dump(rows, open("eval/end_to_end_sets.json", "w"))
    print(f"\nwrote eval/end_to_end_sets.json ({len(rows)} rows)")

    for split in ("unseen", "seen"):
        r = [x for x in rows if x["split"] == split]
        matched = [x for x in r if x.get("matched")]
        spurious = [x for x in r if x.get("spurious")]
        missed = [x for x in r if x.get("missed")]
        ntrue = len(matched) + len(missed)
        correct = [x for x in matched if x["pred_ex"] == x["true_ex"]]
        print(f"\n== {split} participants ==")
        print(f"  set detection recall     {len(matched)/max(ntrue,1)*100:5.1f}%  ({len(matched)}/{ntrue})")
        print(f"  spurious sets            {len(spurious)}")
        print(f"  exercise correct | match {len(correct)/max(len(matched),1)*100:5.1f}%")
        if correct:
            err = np.array([abs(x["pred_reps"] - x["true_reps"]) for x in correct])
            print(f"  rep MAE on matched sets  {err.mean():.2f}")


if __name__ == "__main__":
    main()
