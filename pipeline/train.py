"""
Train the exercise classifier and evaluate it under a participant-level split.

Two test sets are reported deliberately:
  seen   (w09,w10,w11)          - participants that also appear in training
  unseen (w00,w05,w12,w13,w20)  - participants held out entirely
The gap between them is the honest estimate of how much of the model's apparent
skill is really a memorised per-person movement signature.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, f1_score)

sys.path.insert(0, os.path.dirname(__file__))
import data

CACHE = "data/cache/windows_right.npz"


def load(cache=CACHE, min_purity=0.8):
    d = np.load(cache, allow_pickle=True)
    X, y, w, pur = d["X"], d["y"], d["w_id"], d["purity"]
    # Windows straddling a set boundary carry an ambiguous label. Drop them from
    # TRAINING only - they are kept at inference, where they are exactly the
    # windows confidence gating has to handle.
    return X, y, w, pur, list(d["feature_names"])


def split_mask(w_id, ids):
    return np.isin(w_id, [int(i) for i in ids])


def evaluate(model, X, y, name, classes):
    p = model.predict(X)
    proba = model.predict_proba(X)
    f1 = f1_score(y, p, average="macro", zero_division=0)
    acc = (p == y).mean()
    print(f"\n--- {name} ---")
    print(f"macro F1 {f1:.3f}   accuracy {acc:.3f}   n={len(y)}")
    print(classification_report(y, p, labels=range(len(classes)),
                                target_names=classes, zero_division=0, digits=3))
    return {"name": name, "macro_f1": float(f1), "accuracy": float(acc), "n": int(len(y)),
            "confusion": confusion_matrix(y, p, labels=range(len(classes))).tolist(),
            "y_true": y.tolist(), "y_pred": p.tolist(),
            "confidence": proba.max(1).tolist()}


def main():
    X, y, w, pur, feat_names = load()
    classes = data.ALL_CLASSES

    tr = split_mask(w, data.TRAIN_W) & (pur >= 0.8)
    va = split_mask(w, data.VAL_W)
    te_seen = split_mask(w, data.TEST_SEEN_W)
    te_unseen = split_mask(w, data.TEST_UNSEEN_W)

    print(f"train {tr.sum()}  val {va.sum()}  test-seen {te_seen.sum()}  test-unseen {te_unseen.sum()}")

    results = {}
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0,
            class_weight="balanced_subsample"),
        "hist_gbt": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, random_state=0),
    }

    trained = {}
    for name, m in models.items():
        t0 = time.time()
        m.fit(X[tr], y[tr])
        fit_s = time.time() - t0
        t0 = time.time()
        m.predict(X[va][:1000])
        infer_ms = (time.time() - t0) / 1000 * 1000
        print(f"\n===== {name} (fit {fit_s:.1f}s, {infer_ms:.3f} ms/window inference) =====")
        r = {
            "val": evaluate(m, X[va], y[va], "val", classes),
            "test_seen": evaluate(m, X[te_seen], y[te_seen], "test (participants seen in training)", classes),
            "test_unseen": evaluate(m, X[te_unseen], y[te_unseen], "test (participants held out)", classes),
            "fit_seconds": fit_s, "inference_ms_per_window": infer_ms,
        }
        results[name] = r
        trained[name] = m

    best = max(results, key=lambda k: results[k]["val"]["macro_f1"])
    print(f"\nselected on val: {best}")

    if hasattr(trained[best], "feature_importances_"):
        imp = trained[best].feature_importances_
        top = np.argsort(imp)[::-1][:20]
        print("\ntop features:")
        for i in top:
            print(f"  {feat_names[i]:42s} {imp[i]:.4f}")
        results[best]["top_features"] = [[feat_names[i], float(imp[i])] for i in top]

    os.makedirs("eval", exist_ok=True)
    with open("eval/classifier_results.json", "w") as f:
        json.dump({"selected": best, "classes": classes,
                   "results": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}},
                  f)

    import pickle
    os.makedirs("data/cache", exist_ok=True)
    with open("data/cache/model.pkl", "wb") as f:
        pickle.dump({"model": trained[best], "name": best, "classes": classes,
                     "feature_names": feat_names}, f)
    print("\nwrote eval/classifier_results.json and data/cache/model.pkl")


if __name__ == "__main__":
    main()
