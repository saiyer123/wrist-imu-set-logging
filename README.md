# Automatic set logging from a single wrist IMU

Detects exercises, set boundaries, repetitions, tempo and rep-to-rep consistency
from wrist accelerometer and gyroscope data alone — and, more importantly,
decides when it is confident enough to write to your log without asking.

Built on [MM-Fit](https://mmfit.github.io/) (21 workouts, 13.7 hours, 616
labelled sets, 10 exercises).

**[Interactive demo](demo/)** — replay a held-out participant's workout and watch
the log populate. Click any set to see the sensor trace the decision came from.

---

## The problem this is actually solving

Nobody wants to log sets by hand. But an automatic log that is *sometimes wrong*
is worse than no automatic log at all, because a wrong entry silently corrupts
the history you are trying to build a relationship with. A missing set is a
known-unknown the user can fix; a wrong set is a lie they may never notice.

So the objective is not accuracy. It is:

> **maximise the share of sets logged with zero user effort, subject to a hard
> ceiling on how often the log is visibly wrong.**

That reframing drives every design decision below, and it is why the demo's
central control is a confidence threshold rather than a model selector.

## Pipeline

```
wrist accelerometer + gyroscope  (~100 Hz, irregular)
        ↓  resample to a uniform 100 Hz grid, align to label frames
        ↓  4 s sliding windows, 1 s hop
        ↓  207 time / frequency / autocorrelation features (9 channels × 23)
        ↓  gradient-boosted classifier → per-window class probabilities
        ↓  smooth, merge, drop implausible spans → candidate SETS
        ↓  signed principal-axis projection → period, peaks → reps, tempo, consistency
        ↓  confidence gate → auto-log  |  ask the user
```

Everything is deterministic and inspectable except the classifier. No LLM is
involved in sensor interpretation.

## Headline results

Evaluated on **participants never seen in training** (`w00, w05, w12, w13, w20`).

| | held-out participants | participants seen in training |
|---|---|---|
| exercise classification, macro F1 | **0.825** | 0.957 |
| set detection recall (self-segmented) | **77.0%** | 95.6% |
| exercise correct, given a detected set | **96.3%** | 100% |
| rep tempo, median absolute error | **48 ms** | 52 ms |
| rep count MAE (oracle segments) | **0.72** | 0.22 |

At a confidence threshold of 0.76: **70.5% of all real sets are logged with zero
taps, at a 4.9% visible error rate**, leaving 0.32 user actions per set.

Cost: 0.78 ms per window for features, 0.11 ms for inference — roughly 1 ms of
compute per second of workout, comfortably real-time on a phone.

## Five things the data taught me

**1. The standard MM-Fit split leaks participants.**
The published test set (`w09,w10,w11`) contains people who also appear in
training. Macro F1 is 0.957 there and **0.825** on genuinely held-out
participants. Reporting the first number would have overstated real-world
performance by 13 points. The demo deliberately replays a held-out participant.

**2. Rep-count MAE is a nearly meaningless metric on this dataset.**
93.6% of MM-Fit sets contain exactly 10 reps. A model that ignores its input and
always answers "10" scores MAE 0.098 and 93.6% exact-match — better than my
detector. That is not a real result, it is a property of the labels. Two fixes:
report **tempo** instead (exactly recoverable as `duration / reps`, with 5.4× the
relative spread), and evaluate counting on **randomly truncated sub-segments**
where the answer actually varies. There the illusion collapses:

| | rep MAE | correlation with truth |
|---|---|---|
| detector | **0.79** | **0.80** |
| always-predict-10 | 3.43 | 0.00 |

**3. One wrist physically cannot see a two-armed alternating exercise.**
Bicep curls came back at exactly 2.00× the true period. Not a tuning bug — MM-Fit
curls alternate arms, so each wrist witnesses half the reps. Both wrists
independently report 2.00×, which is how I confirmed it. No amount of model
capacity recovers a repetition the sensor never moved for; it needs prior
knowledge, applied after the exercise is identified. A per-exercise multiplier
learned from training participants alone recovers it, and learns `×2` for bicep
curls and `×1` for all nine other exercises without being told which is which.

**4. Vector magnitude is a rectifier, and it silently doubles your cadence.**
Every badly-wrong segment had one thing in common: the gyroscope channel. Taking
`‖gyro‖` discards direction, so a reciprocating movement peaks on the way out
*and* on the way back — exactly twice per rep. The accelerometer mostly escapes
this because gravity breaks the symmetry. Replacing magnitude with a **signed
projection onto the first principal component** fixed it: jumping-jack rep MAE
went from 4.14 to **0.08**, and overall count MAE from 0.96 to 0.72.

**5. The obvious personalisation was aimed at the wrong failure.**
Learning per-user class centroids from a few confirmed sets *hurt* at every blend
weight. The diagnosis: on held-out participants, **85.5% of window errors are an
exercise misread as rest**, not one exercise confused for another. Centroid
blending redistributes probability among exercise classes, so it addressed the
other 14.5%. Personalising the **activity/rest boundary** instead — one
movement-energy threshold from the user's own confirmed sets — lifts coverage
from 75.2% to **79.8%** at the same error ceiling, from six confirmations.

## Deliberate tradeoffs

- **Right wrist only.** MM-Fit participants wore watches on both wrists. Using
  both raises accuracy but describes a setup nobody has. Single-wrist is the
  default; `wrists="both"` exists for measuring the gap.
- **Interpretable baseline before a neural net.** A 1D CNN was scoped and not
  built: the feature model already reaches 0.96 precision on held-out
  participants, and the binding constraint is the activity/rest boundary and
  single-wrist observability, neither of which more capacity fixes. Spending the
  time on evaluation design was worth more than spending it on architecture.
- **Confidence is the weakest link, not an average.** A set's confidence is the
  minimum of its classification and rep-detection confidence, so a clean
  classification cannot paper over an unreadable rep signal.
- **No LLM in the sensing path.** Deterministic metrics first; any natural
  language summary would sit strictly downstream of verified measurements.

## Layout

```
pipeline/
  fetch_mmfit.py     selective range-request download of the IMU subset
  data.py            loading, 100 Hz resampling, participant-level splits
  features.py        207 windowed time/frequency/autocorrelation features
  train.py           classifier training + per-split evaluation
  reps.py            period, count, tempo, consistency, alternating correction
  segment.py         probability stream → discrete sets
  end_to_end.py      full path with no oracle segmentation
  gating.py          coverage / error / burden curves + personalisation
  export_demo.py     JSON for the web demo
demo/                React + TypeScript replay interface
eval/                metrics written by the pipeline
docs/EVALUATION.md   full results, including what failed
```

## Running it

```bash
python3 pipeline/fetch_mmfit.py data/raw    # ~338 MB, IMU + labels only
```
```bash
python3 pipeline/build_dataset.py right && python3 pipeline/train.py
```
```bash
python3 pipeline/end_to_end.py && python3 pipeline/gating.py && python3 pipeline/export_demo.py 00
```
```bash
cd demo && npm install && npm run dev
```

`fetch_mmfit.py` reads the zip's central directory over HTTP range requests and
pulls only the accelerometer, gyroscope and label members — 338 MB instead of
downloading and unpacking the full 1.74 GB archive.
