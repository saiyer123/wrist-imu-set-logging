"""Extract features for every workout once and cache them to data/cache/."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import data
import features


def build(wrists="right", out_dir="data/cache"):
    os.makedirs(out_dir, exist_ok=True)
    tag = wrists
    Xs, ys, purs, wids, cents = [], [], [], [], []
    names = None

    for w in data.available_workouts():
        t0 = time.time()
        wk = data.load_workout(w, wrists=wrists)
        X, names, centres = features.extract(wk["sig"], wk["names"])
        y, pur = features.window_labels(data.label_per_sample(wk), centres)
        Xs.append(X.astype(np.float32))
        ys.append(y)
        purs.append(pur.astype(np.float32))
        cents.append(centres)
        wids.append(np.full(len(y), int(w), dtype=np.int16))
        print(f"  w{w}: {X.shape[0]:5d} windows  {time.time()-t0:.1f}s", flush=True)

    out = os.path.join(out_dir, f"windows_{tag}.npz")
    np.savez_compressed(
        out,
        X=np.concatenate(Xs), y=np.concatenate(ys), purity=np.concatenate(purs),
        w_id=np.concatenate(wids), centre=np.concatenate(cents),
        feature_names=np.array(names),
    )
    print(f"wrote {out}  ({os.path.getsize(out)/1e6:.0f} MB)")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "right")
