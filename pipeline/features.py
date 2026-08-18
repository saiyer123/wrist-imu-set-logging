"""
Windowed time- and frequency-domain features for wrist IMU.

Deliberately an interpretable baseline: every feature here is something you can
name and reason about (dominant frequency, autocorrelation peak, jerk, spectral
entropy). If a learned model later beats this, we will know what it had to beat
and roughly why.

Window geometry: 4 s at 100 Hz with a 1 s hop. 4 s spans ~2 repetitions of a
typical lift, which is the minimum needed for a frequency-domain estimate of
cadence to mean anything; the 1 s hop sets the timeline's temporal resolution.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal as sps
from scipy import stats as sst

FS = 100.0
WIN_S = 4.0
HOP_S = 1.0
WIN = int(WIN_S * FS)
HOP = int(HOP_S * FS)

# Human exercise cadence lives roughly between 0.2 Hz (a 5 s rep) and 3 Hz.
REP_FMIN, REP_FMAX = 0.2, 3.0
BANDS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 10.0)]


def _augment_channels(sig, names):
    """Append acc/gyr vector magnitudes and jerk magnitude - orientation-invariant
    signals matter because wrist rotation varies between users and sessions."""
    out, out_names = [sig], list(names)
    for dev_sig in ("acc", "gyr"):
        idx = [i for i, n in enumerate(names) if f"_{dev_sig}_" in n]
        for start in range(0, len(idx), 3):
            tri = idx[start:start + 3]
            if len(tri) < 3:
                continue
            mag = np.linalg.norm(sig[:, tri], axis=1, keepdims=True)
            out.append(mag)
            out_names.append(names[tri[0]].rsplit("_", 1)[0] + "_mag")
    stacked = np.concatenate(out, axis=1)

    # jerk magnitude from accelerometer magnitude channels
    jidx = [i for i, n in enumerate(out_names) if n.endswith("acc_mag")]
    jerk = np.abs(np.diff(stacked[:, jidx], axis=0, prepend=stacked[:1, jidx])) * FS
    stacked = np.concatenate([stacked, jerk], axis=1)
    out_names += [out_names[i].replace("acc_mag", "jerk") for i in jidx]
    return stacked, out_names


def _windows(x, win=WIN, hop=HOP):
    """(T,C) -> (nwin, win, C)"""
    if len(x) < win:
        return np.empty((0, win, x.shape[1]))
    return sliding_window_view(x, win, axis=0)[::hop].transpose(0, 2, 1)


def _feature_names(chan_names):
    time_f = ["mean", "std", "min", "max", "ptp", "median", "iqr", "rms", "mad",
              "skew", "kurt", "zcr"]
    freq_f = ["dom_freq", "dom_ratio", "spec_centroid", "spec_entropy"]
    freq_f += [f"band{lo}_{hi}" for lo, hi in BANDS]
    ac_f = ["ac_peak_lag_hz", "ac_peak_val"]
    names = []
    for c in chan_names:
        names += [f"{c}__{f}" for f in time_f + freq_f + ac_f]
    return names


def _per_channel(w):
    """w is (nwin, win) for one channel -> (nwin, n_feat)."""
    n = w.shape[0]
    mean = w.mean(1)
    std = w.std(1)
    centred = w - mean[:, None]

    q75, q25 = np.percentile(w, [75, 25], axis=1)
    feats = [
        mean, std, w.min(1), w.max(1), np.ptp(w, axis=1), np.median(w, axis=1),
        q75 - q25,
        np.sqrt((w ** 2).mean(1)),
        np.abs(centred).mean(1),
        sst.skew(w, axis=1),
        sst.kurtosis(w, axis=1),
        (np.diff(np.signbit(centred), axis=1) != 0).sum(1) / w.shape[1],
    ]

    # --- frequency domain (Hann-windowed, mean-removed) ---
    taper = np.hanning(w.shape[1])
    spec = np.abs(np.fft.rfft(centred * taper, axis=1))
    freqs = np.fft.rfftfreq(w.shape[1], 1.0 / FS)
    power = spec ** 2
    total = power.sum(1) + 1e-12

    band = (freqs >= REP_FMIN) & (freqs <= REP_FMAX)
    sub = power[:, band]
    dom_i = sub.argmax(1)
    feats += [
        freqs[band][dom_i],                      # dominant cadence
        sub.max(1) / total,                      # how periodic the window is
        (power * freqs).sum(1) / total,          # spectral centroid
        -(lambda p: (p * np.log(p + 1e-12)).sum(1))(power / total[:, None]),  # entropy
    ]
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        feats.append(power[:, m].sum(1) / total)

    # --- autocorrelation: strongest periodicity and its lag ---
    nfft = 1 << int(np.ceil(np.log2(2 * w.shape[1])))
    f = np.fft.rfft(centred, n=nfft, axis=1)
    ac = np.fft.irfft(f * np.conj(f), n=nfft, axis=1)[:, : w.shape[1]]
    ac /= ac[:, :1] + 1e-12
    lo_lag = max(1, int(FS / REP_FMAX))
    hi_lag = min(w.shape[1] - 1, int(FS / REP_FMIN))
    seg = ac[:, lo_lag:hi_lag]
    if seg.shape[1] == 0:
        feats += [np.zeros(n), np.zeros(n)]
    else:
        pk = seg.argmax(1) + lo_lag
        feats += [FS / np.maximum(pk, 1), seg.max(1)]

    return np.stack(feats, axis=1)


def extract(sig, names):
    """
    (T,C) signal -> (nwin, F) features, plus feature names and the sample index at
    the centre of each window.
    """
    aug, aug_names = _augment_channels(sig, names)
    w = _windows(aug)
    if w.shape[0] == 0:
        return np.empty((0, len(_feature_names(aug_names)))), _feature_names(aug_names), np.empty(0, int)

    feats = np.concatenate([_per_channel(w[:, :, c]) for c in range(w.shape[2])], axis=1)
    centres = np.arange(w.shape[0]) * HOP + WIN // 2
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0), _feature_names(aug_names), centres


def window_labels(y_dense, centres, win=WIN):
    """Majority class within each window, plus its purity."""
    lab = np.zeros(len(centres), dtype=np.int16)
    pur = np.zeros(len(centres))
    for i, c in enumerate(centres):
        seg = y_dense[max(0, c - win // 2): c + win // 2]
        vals, counts = np.unique(seg, return_counts=True)
        j = counts.argmax()
        lab[i], pur[i] = vals[j], counts[j] / len(seg)
    return lab, pur
