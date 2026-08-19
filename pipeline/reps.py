"""
Repetition detection: cadence, count, tempo and rep-to-rep consistency.

Method is deliberately deterministic and inspectable - no learned model. A rep is
a periodic event, so we estimate the period by autocorrelation, then use that
period to constrain peak picking. The two have to agree; when they disagree we
say so via a lower confidence rather than silently picking one.

WHY WE DO NOT HEADLINE REP-COUNT MAE
------------------------------------
93.6% of MM-Fit sets contain exactly 10 reps. A model that ignores its input and
always answers "10" scores MAE 0.098 and 93.6% exact-match. Any rep-count MAE
quoted on oracle-segmented sets is therefore close to meaningless.

Mean rep period, by contrast, is exactly recoverable from the labels
(set_duration / rep_count) and has 5.4x the relative spread of rep count
(CV 0.297 vs 0.055). It is the metric that can actually be failed, so it is the
one we report first.
"""
import numpy as np
from scipy import signal as sps

FS = 100.0
FMIN, FMAX = 0.2, 3.0   # 0.33-5 s per rep


def _bandpass(x, lo=FMIN, hi=FMAX, fs=FS, order=4):
    b, a = sps.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return sps.filtfilt(b, a, x)


def _autocorr_period(x, fs=FS):
    """
    Dominant period in seconds via normalised autocorrelation, plus its strength.

    Two "improvements" were tried here and both were reverted, which is worth
    recording because each sounds correct in isolation:

    1. UNBIASED NORMALISATION. Dividing by the per-lag overlap count is the
       textbook fix for autocorrelation's decay with lag. It made tempo error
       twice as bad (145 ms -> 289 ms on held-out participants). The decay is
       not merely an artefact here: it acts as a prior against period-doubling,
       and removing it let long lags win more often than they should.

    2. OCTAVE PROMOTION. Because we track vector magnitude, which discards
       direction, an up-down movement can put two magnitude peaks in one
       repetition, so the first harmonic can outrank the fundamental. Promoting
       to a longer lag when it is "almost as strong" made things worse at every
       threshold tested - a periodic signal has autocorrelation peaks at EVERY
       multiple of its period, so that test passes almost always and doubles
       periods that were already right. Measured rate of genuine halving: ~2%.

    Both experiments are written up in docs/EVALUATION.md.
    """
    x = x - x.mean()
    n = 1 << int(np.ceil(np.log2(2 * len(x))))
    f = np.fft.rfft(x, n=n)
    ac = np.fft.irfft(f * np.conj(f), n=n)[: len(x)]
    if ac[0] <= 0:
        return np.nan, 0.0
    ac = ac / ac[0]

    lo, hi = int(fs / FMAX), min(len(x) - 1, int(fs / FMIN))
    if hi <= lo:
        return np.nan, 0.0

    seg = ac[lo:hi]
    k = int(np.argmax(seg))
    i = k + lo
    if 0 < k < len(seg) - 1:                 # parabolic sub-sample refinement
        y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
        denom = y0 - 2 * y1 + y2
        i = i + (0.5 * (y0 - y2) / denom if denom != 0 else 0.0)
    return i / fs, float(seg[k])


def _project(tri):
    """
    Signed 1-D projection of a tri-axial signal onto its dominant axis of motion.

    We deliberately do NOT use vector magnitude here. Magnitude is a rectifier:
    it throws away sign, so a reciprocating movement puts a peak in the trace on
    the way out AND on the way back, at exactly twice the true cadence. The
    gyroscope suffers worst, because angular velocity is symmetric about the
    turnaround and has no gravity offset to break the tie - measured on MM-Fit,
    every gyroscope-selected segment came back at exactly half the true period.

    Projecting onto the first principal component keeps the sign, so one
    repetition produces one cycle.
    """
    x = tri - tri.mean(axis=0, keepdims=True)
    # first principal component via SVD of the (T,3) block
    try:
        _u, _s, vt = np.linalg.svd(x, full_matrices=False)
        axis = vt[0]
    except np.linalg.LinAlgError:
        axis = np.array([1.0, 0.0, 0.0])
    proj = x @ axis
    # sign convention: make the largest-magnitude excursion positive so traces
    # look consistent between sets in the UI
    if np.abs(proj.min()) > np.abs(proj.max()):
        proj = -proj
    return proj


def _pick_channel(sig, names):
    """
    Choose the signal that best exposes this movement's periodicity.

    A wrist gyroscope sees curls and presses clearly but barely moves during
    squats, where the accelerometer's gravity component carries the signal.
    Rather than hard-code a channel per exercise, score both by autocorrelation
    strength and take the winner.
    """
    cands = {}
    for kind in ("acc", "gyr"):
        idx = [i for i, n in enumerate(names) if f"_{kind}_" in n][:3]
        if len(idx) == 3:
            cands[kind] = _project(sig[:, idx])

    best, best_s, best_p = None, -1.0, np.nan
    for kind, x in cands.items():
        if len(x) < int(FS):
            continue
        xf = _bandpass(x)
        p, s = _autocorr_period(xf)
        if s > best_s:
            best, best_s, best_p = (kind, xf), s, p
    if best is None:
        return "acc", np.zeros(len(sig)), np.nan, 0.0
    return best[0], best[1], best_p, best_s


ALTERNATION_THRESHOLD = 4.0


def _alternation_score(trace, peaks):
    """
    Evidence that the trace is frequency-doubled: do peak heights alternate?

    If a reciprocating movement resolves into two similar half-cycles, the
    detector locks onto half the true period and every OTHER peak belongs to the
    return stroke. Those return strokes are rarely identical to the drive
    strokes, so peak heights alternate high-low-high-low. Comparing even-indexed
    against odd-indexed peaks, in units of within-group spread, separates the two
    cases by more than an order of magnitude on training data (median 20.5 for
    doubled segments against 0.57 for correctly-tracked ones).

    This replaced an earlier attempt that used per-exercise tempo priors to pick
    the octave. That failed for a reason worth remembering: held-out participants
    genuinely perform lateral raises at 1.40 s against a training median of
    2.27 s, and the prior "corrected" that real between-person difference into a
    fabricated doubling, tripling tempo error. A prior cannot tell "this person
    is faster" from "the detector doubled". The signal can.
    """
    if len(peaks) < 6:
        return 0.0
    h = np.asarray(trace)[peaks]
    a, b = h[0::2], h[1::2]
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    pooled = np.sqrt((a.var() + b.var()) / 2.0) + 1e-9
    return float(abs(a.mean() - b.mean()) / pooled)


def count_reps(sig, names):
    """
    Analyse one candidate set segment.

    Returns dict with rep_count, period_s (tempo), consistency, confidence and the
    filtered trace + peak indices so the UI can show its working.
    """
    kind, xf, period, ac_strength = _pick_channel(sig, names)
    dur = len(sig) / FS
    out = {"channel": kind, "duration_s": float(dur), "period_s": float(period)
           if np.isfinite(period) else None, "ac_strength": float(ac_strength),
           "trace": xf, "peaks": np.array([], dtype=int)}

    if not np.isfinite(period) or period <= 0 or dur < 2 * period:
        out.update(rep_count=0, consistency=0.0, confidence=0.0,
                   count_from_period=0, count_from_peaks=0)
        return out

    # Peak picking constrained by the period we just estimated.
    prom = 0.4 * np.std(xf)
    peaks, _ = sps.find_peaks(xf, distance=max(1, int(0.6 * period * FS)),
                              prominence=prom)
    n_peaks = len(peaks)
    n_period = int(round(dur / period))

    # Two independent estimates; their agreement is the core of our confidence.
    agree = 1.0 - min(1.0, abs(n_peaks - n_period) / max(n_period, 1))
    count = n_peaks if n_peaks > 0 else n_period

    # Frequency-doubling check before anything is derived from the peaks.
    alt = _alternation_score(xf, peaks)
    if alt > ALTERNATION_THRESHOLD and n_peaks >= 6:
        h = xf[peaks]
        keep_even = h[0::2].mean() >= h[1::2].mean()
        peaks = peaks[0::2] if keep_even else peaks[1::2]
        n_peaks = len(peaks)
        period = period * 2.0
        n_period = int(round(dur / period))
        agree = 1.0 - min(1.0, abs(n_peaks - n_period) / max(n_period, 1))
        count = n_peaks
    out["alternation"] = alt
    out["frequency_doubled"] = bool(alt > ALTERNATION_THRESHOLD and n_peaks >= 3)

    if n_peaks >= 3:
        iv = np.diff(peaks) / FS
        consistency = float(np.clip(1.0 - iv.std() / max(iv.mean(), 1e-6), 0.0, 1.0))
        period = float(iv.mean())
    else:
        consistency = 0.0

    # An ambiguity penalty on borderline alternation was tried here and reverted:
    # it cost 10 points of coverage (69.8% -> 59.7% at threshold 0.76) to remove a
    # single visible error, because segment confidence already takes the MINIMUM
    # of the classification and rep stages, so the penalty mostly suppressed sets
    # whose rep signal was fine. The borderline doubling case it was meant to
    # catch was already being flagged by classification confidence.
    confidence = float(np.clip(0.5 * ac_strength + 0.3 * agree + 0.2 * consistency, 0, 1))
    out.update(rep_count=int(count), consistency=consistency, confidence=confidence,
               count_from_period=int(n_period), count_from_peaks=int(n_peaks),
               period_s=float(period), peaks=peaks)
    return out


# ---------------------------------------------------------------------------
# Alternating-movement correction
# ---------------------------------------------------------------------------
# Bicep curls in MM-Fit are performed alternating arms. A single wrist therefore
# witnesses exactly half the repetitions, and BOTH wrists independently report a
# period ratio of 2.00. This is an observability limit of single-wrist sensing,
# not a defect in the peak detector - no additional model capacity can recover a
# rep the sensor never moved for.
#
# The fix is prior knowledge, applied only after the exercise has been
# identified: learn one multiplier per exercise from TRAINING participants and
# apply it at inference. Ratios are snapped to simple fractions so we encode
# "this movement alternates" rather than fitting noise.

def fit_multipliers(workouts, load_workout, frame_to_index, valid_classes):
    """Median ground-truth/detected count ratio per exercise, snapped to {1, 2}."""
    import collections
    ratios = collections.defaultdict(list)
    for w in workouts:
        wk = load_workout(w)
        for s, e, r, act in wk["labels"]:
            if act not in valid_classes or r <= 0:
                continue
            i0, i1 = frame_to_index(wk["frame"], s), frame_to_index(wk["frame"], e)
            if i1 - i0 < 200:
                continue
            o = count_reps(wk["sig"][i0:i1], wk["names"])
            if o["rep_count"] > 0:
                ratios[act].append(r / o["rep_count"])

    mult = {}
    for act, v in ratios.items():
        m = float(np.median(v))
        mult[act] = min((1.0, 2.0), key=lambda c: abs(c - m))
    return mult


def apply_multiplier(result, exercise, multipliers):
    """Scale a count_reps() result by the learned per-exercise multiplier."""
    m = multipliers.get(exercise, 1.0)
    out = dict(result)
    out["multiplier"] = m
    out["rep_count"] = int(round(result["rep_count"] * m))
    if result.get("period_s"):
        out["period_s"] = result["period_s"] / m
    return out


