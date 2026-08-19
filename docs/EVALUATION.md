# Evaluation

All numbers produced by `pipeline/train.py`, `pipeline/end_to_end.py` and
`pipeline/gating.py`. Raw artefacts in `eval/`.

## Splits

MM-Fit ships two test sets and the distinction matters more than it looks.

| split | workouts | note |
|---|---|---|
| train | 01,02,03,04,06,07,08,16,17,18 | |
| val | 14,15,19 | |
| test (seen) | 09,10,11 | **participants also appear in training** |
| test (unseen) | 00,05,12,13,20 | participants held out entirely |

Windows straddling a set boundary (label purity < 0.8) are dropped from training
only. They are kept at inference, where they are exactly the ambiguous windows
confidence gating exists to handle.

## Exercise classification

Gradient-boosted trees (`HistGradientBoostingClassifier`, 300 iterations),
selected over a random forest on validation macro F1. 207 features from 9
channels (acc xyz, gyr xyz, acc magnitude, gyr magnitude, jerk) × 23 features.

| | macro F1 | accuracy |
|---|---|---|
| validation | 0.871 | 0.919 |
| test, participants seen in training | 0.957 | 0.982 |
| **test, participants held out** | **0.825** | **0.938** |

On held-out participants macro precision is 0.937 but macro recall is 0.760. The
model is conservative with strangers: when it fires it is usually right, but it
stays quiet often. That asymmetry is what makes confidence gating viable.

### Where the errors go

Of windows whose true label is an exercise, on held-out participants:

- 27.6% are misclassified
- **85.5% of those are predicted `non_activity`** — the model does not recognise
  the person is exercising at all
- only 14.5% are confused with a different exercise

The binding constraint is the activity/rest boundary, not inter-class confusion.
This single measurement redirected the personalisation work (below).

### Per-exercise, held-out participants

Squats (recall 0.44) and bicep curls (0.43) are hardest, and both make physical
sense from a wrist: during squats the wrist is comparatively still, and bicep
curls alternate arms so the sensor sees half the movement. Jumping jacks (0.93)
and lunges (0.86) are easiest — whole-body movements with large wrist excursion.

## Repetition detection

Ground truth for **tempo** is exact: `set_duration / rep_count`. It has a
coefficient of variation of 0.297 against 0.055 for rep count, so it is the
metric that can actually be failed.

| oracle-segmented sets | held-out | seen |
|---|---|---|
| tempo, median absolute error | **46 ms** | 52 ms |
| tempo, mean absolute error | 137 ms | 82 ms |
| tempo within 10% of truth | 80% | 98% |
| rep count MAE | 0.55 | 0.22 |
| *baseline: predict the global median tempo* | 545 ms | 489 ms |

### Why rep-count MAE is reported but not headlined

Rep-count distribution across all 616 sets:

| reps | 1 | 6 | 9 | 10 | 11 | 12 | 14 |
|---|---|---|---|---|---|---|---|
| sets | 1 | 1 | 8 | **467** | 18 | 3 | 1 |

93.6% are exactly 10. A constant predictor scores MAE 0.098 and 93.6%
exact-match — better than the detector. That is a statement about the labels,
not the method.

Evaluating on randomly truncated sub-segments (30–100% of each set, ground truth
scaled proportionally — approximate, but it *varies*) separates them:

| held-out participants, 556 sub-segments | rep MAE | correlation with truth |
|---|---|---|
| detector | **0.78** | **0.80** |
| always predict 10 | 3.43 | 0.00 |

## End-to-end, no oracle segmentation

Set boundaries discovered by the system itself; matched to ground truth by
temporal IoU ≥ 0.5.

| | held-out | seen |
|---|---|---|
| set detection recall | 75.5% (105/139) | 95.6% (86/90) |
| spurious sets | 1 | 0 |
| exercise correct, given a match | 96.2% | 100% |
| rep MAE on matched sets | 1.77 | 0.86 |

Rep error is higher here than on oracle segments because detected boundaries are
tighter than true ones — the detector finds the middle of a set and misses
ramp-in and ramp-out reps. This is the honest number for a deployed system.

## Confidence gating

Held-out participants, 139 real sets. "Correction burden" counts every action a
user must take to reach a correct log: deleting a wrong auto-entry, confirming an
uncertain one, or adding a missed set by hand.

| threshold | coverage | visible errors | error rate of auto entries | burden/set |
|---|---|---|---|---|
| 0.00 | 72.7% | 5 | 4.7% | 0.28 |
| 0.61 | 71.9% | 4 | 3.8% | 0.29 |
| **0.76** | **69.8%** | **2** | **2.0%** | **0.30** |
| 0.78 | 68.3% | 2 | 2.1% | 0.32 |
| 0.90 | 20.1% | 1 | 3.4% | 0.80 |

Past 0.78 the system stops being useful — it is correct because it says nothing.

The floor is much lower than it was before the plausibility gate below: even at
threshold 0, the error rate is now 4.7% rather than 15.6%, because most of what
the gate removed was spurious rather than merely uncertain.

### Plausibility gate on detected sets

Thresholds chosen on the **validation split only**, where real detections had a
5th percentile of 10.4 s duration and 7.4 reps:

| rule | keeps real | keeps spurious |
|---|---|---|
| duration ≥ 8 s **and** reps ≥ 5 | 100% | 20% |

On held-out participants this cut spurious sets from **15 to 1** and cost 1.5
points of recall (77.0% → 75.5%).

## Personalisation

Simulated: the user confirms their first 6 sets; the system adapts; evaluation
runs only on the remainder of that workout, which the user has not touched.

**Attempt 1 — per-user class centroids (rejected).** Class centroids in
standardised feature space from confirmed windows, blended with global
probabilities.

| blend weight α | 0.0 (global) | 0.15 | 0.30 | 0.50 |
|---|---|---|---|---|
| coverage at ≤5% error | 75.2% | 74.3% | 73.4% | 58.7% |

Never better than not doing it. The diagnosis above explains why: it targets
inter-class confusion, which is 14.5% of the errors.

**Attempt 2 — per-user activity/rest boundary (kept).** From the confirmed sets,
take the user's own rest and work movement-energy distributions, place a
threshold between them, and suppress `non_activity` probability above it.

| λ (suppression) | 0.0 | 0.50 | 0.80 | 0.95 |
|---|---|---|---|---|
| coverage at ≤5% error, 6 confirmations | 75.2% | 78.0% | 78.0% | **83.5%** |

+8.3 points of coverage at the same error ceiling, for six taps, with no gradient
step and no retraining.

## Rejected changes worth recording

**Unbiased autocorrelation normalisation.** Dividing by per-lag overlap count is
the textbook correction for autocorrelation's decay with lag. It doubled tempo
error (145 ms → 289 ms mean, held-out). The decay is not purely an artefact — it
acts as a prior against period-doubling, and removing it let long lags win too
often.

**Octave promotion.** Since magnitude rectification can let the first harmonic
outrank the fundamental, promote to a longer lag when it is "almost as strong".
Worse at every threshold tested (0.85 → 1082 ms; 1.15 → no change). A periodic
signal has autocorrelation peaks at *every* multiple of its period, so the test
passes almost always and doubles periods that were already correct. Measured rate
of genuine halving: ~2% of short segments.

The real fix was upstream — don't rectify in the first place. See finding 4 in
the README.

**Per-exercise tempo priors for octave selection.** Once the classifier names the
exercise, per-class observed-period priors from training are tight (log-sd
0.075–0.22), so a factor-of-two error should sit 3–9 σ out. Choosing among
{P/2, P, 2P} by likelihood tripled tempo error on held-out participants
(165 ms → 390 ms), because those participants genuinely perform lateral raises at
1.40 s against a training median of 2.27 s. The prior cannot separate "this
person is faster" from "the detector doubled", so it converted real between-person
variation into fabricated corrections. Worst affected: lateral raises
(119 ms → 827 ms), tricep extensions (90 ms → 763 ms).

**Alternation-based confidence penalty.** Downgrading confidence on segments with
borderline peak-height alternation cost 10 points of coverage (69.8% → 59.7% at
threshold 0.76) to remove a single visible error. Segment confidence already
takes the minimum of the classification and rep stages, so the penalty mostly
suppressed sets whose rep signal was fine, and the borderline case it targeted
was already being flagged by classification confidence.

**What was kept.** A conservative peak-height alternation corrector at a
threshold of 4.0. It fires on 3 of 229 test sets, all bicep curls, and corrects
all three to within 25 ms. The honest caveat: the training split contains exactly
**one** clearly frequency-doubled segment, so this threshold is calibrated on
almost no data. It is set to fire rarely and be right when it does, and the
residual doubling cases are left to the confidence gate.

## Known limitations

- Rep-count ground truth for truncated segments assumes uniform rep spacing
  within a set. MM-Fit provides no per-rep timestamps, so this is approximate.
- 10 exercises, one equipment context, 21 workouts. Nothing here has met a
  machine, a cable, or an unlabelled movement.
- The alternating-arm multiplier is learned per exercise class, so it transfers
  to new participants but not to a new exercise never seen in training.
- Personalisation is simulated from ground-truth labels standing in for user
  confirmations. Real users confirm inconsistently and sometimes wrongly.
- Single session per participant: no test of drift across weeks, watch
  re-positioning, or wrist swapping.
