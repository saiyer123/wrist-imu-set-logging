"""Export one workout's analysis as JSON for the interactive demo."""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import data, end_to_end, reps, segment

PLOT_HZ = 20  # enough to see rep structure, small enough to ship in a web page


def export(w_id="00", out="demo/public/workout.json"):
    bundle = pickle.load(open("data/cache/model.pkl", "rb"))
    mult = json.load(open("data/cache/rep_multipliers.json"))
    wk, X, proba, centres, segs, true_segs = end_to_end.analyse_workout(w_id, bundle, mult)

    step = int(data.TARGET_HZ / PLOT_HZ)
    acc = np.linalg.norm(wk["sig"][:, 0:3], axis=1)[::step]
    gyr = np.linalg.norm(wk["sig"][:, 3:6], axis=1)[::step]

    m, up, ut = segment.match_to_truth(segs, true_segs)
    truth_of = {i: j for i, j, _ in m}

    out_segs = []
    for i, s in enumerate(segs):
        j = truth_of.get(i)
        r = reps.count_reps(wk["sig"][s["start_idx"]:s["end_idx"]], wk["names"])
        tr = r["trace"]
        ds = max(1, len(tr) // 600)
        out_segs.append({
            "id": i,
            "start_s": round(s["start_idx"] / data.TARGET_HZ, 2),
            "end_s": round(s["end_idx"] / data.TARGET_HZ, 2),
            "exercise": s["exercise"],
            "reps": s["rep_count"],
            "tempo_s": round(s["period_s"], 2) if s["period_s"] else None,
            "consistency": round(s["consistency"], 3),
            "confidence": round(s["confidence"], 3),
            "cls_confidence": round(s["cls_confidence"], 3),
            "channel": s["channel"],
            "multiplier": s["multiplier"],
            "trace": [round(float(v), 3) for v in tr[::ds]],
            "trace_hz": round(data.TARGET_HZ / ds, 2),
            "peaks": [int(p // ds) for p in r["peaks"]],
            "truth": None if j is None else {
                "exercise": true_segs[j]["exercise"], "reps": true_segs[j]["reps"]},
        })

    payload = {
        "workout_id": f"w{w_id}",
        "split": "held-out participant",
        "duration_s": round(len(wk["t"]) / data.TARGET_HZ, 1),
        "plot_hz": PLOT_HZ,
        "acc_mag": [round(float(v), 2) for v in acc],
        "gyr_mag": [round(float(v), 2) for v in gyr],
        "segments": out_segs,
        "missed_truth": [{"start_s": round(true_segs[j]["start_idx"] / data.TARGET_HZ, 1),
                          "end_s": round(true_segs[j]["end_idx"] / data.TARGET_HZ, 1),
                          "exercise": true_segs[j]["exercise"],
                          "reps": true_segs[j]["reps"]} for j in ut],
        "n_true_sets": len(true_segs),
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(payload, open(out, "w"))
    print(f"wrote {out}  {os.path.getsize(out)/1e6:.2f} MB  "
          f"({len(out_segs)} detected, {len(ut)} missed, {len(true_segs)} true)")


if __name__ == "__main__":
    export(sys.argv[1] if len(sys.argv) > 1 else "00")
