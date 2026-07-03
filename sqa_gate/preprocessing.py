"""
Preprocessing + envelope extraction for Hong Tang's heart-sound Signal Quality
Assessment (SQA), ported from MATLAB to Python (numpy/scipy).

Original MATLAB by Hong Tang, Dalian University of Technology (2019).
This is a faithful port; see README.md for the MATLAB->Python file mapping and
the fidelity caveats (a few MATLAB functions like `pwelch`/`spectrogram` have
convention-dependent scaling that is documented at each call site).

MATLAB's std() normalizes by N-1, so we use ddof=1 everywhere to match.
"""

import numpy as np
from scipy.signal import butter, filtfilt

EPS = np.finfo(float).eps


def _mstd(x):
    """std matching MATLAB (ddof=1)."""
    return np.std(np.asarray(x, float), ddof=1)


def remove_spike(input_signal):
    """
    Port of remove_spike.m.

    Clip samples whose |amplitude| exceeds 3x a robust threshold TH, where TH is
    the mean of the largest 10% of |samples|. In the MATLAB source the "limit to
    1% of samples" branch is immediately overridden by the final line that clips
    *all* detected spikes to +/-3*TH, so the net effect is simply that clip.
    """
    x = np.asarray(input_signal, float).ravel()
    out = x.copy()
    R = 3

    abs_signal = np.abs(x)
    sort_abs = np.sort(abs_signal)[::-1]  # descending
    n_top = int(np.floor(len(x) * 0.1))
    n_top = max(n_top, 1)
    TH = np.mean(sort_abs[:n_top])

    ind_spike = np.where(abs_signal > R * TH)[0]
    if ind_spike.size:
        out[ind_spike] = np.sign(x[ind_spike]) * R * TH
    return out


def pre_processing(input_signal, fs):
    """
    Port of pre_processing.m: normalize -> remove spikes -> remove baseline
    wander (3rd-order Butterworth high-pass, fc=2 Hz) -> normalize.
    """
    x = np.asarray(input_signal, float).ravel()
    x = x / _mstd(x)

    x = remove_spike(x)

    fc = 2.0
    b, a = butter(3, 2 * fc / fs, btype="high")
    x = filtfilt(b, a, x)

    x = x / _mstd(x)
    return x


def get_envelope_from_stft(phs, fs):
    """
    Port of getEnvelopeFromSTFT.m.

    Sliding-window STFT magnitude envelope: a rectangular window of 30 ms slides
    one sample at a time (noverlap = len-1), each frame is DFT'd with nfft=fs,
    and the per-frame magnitude spectrum is summed then normalized by nfft. The
    result is low-pass filtered (fc=20 Hz) and de-spiked.

    MATLAB's spectrogram returns the one-sided STFT for real input, which rfft
    reproduces exactly (no extra scaling), so the magnitudes match.
    """
    phs = np.asarray(phs, float).ravel()
    L = int(round(0.03 * fs))          # window length in samples
    nfft = int(round(fs))              # DFT length

    if len(phs) < L:
        # Degenerate: too short to form a single frame.
        return np.zeros(1)

    # Frames of length L with hop 1 (noverlap = L-1). Window is all ones.
    frames = np.lib.stride_tricks.sliding_window_view(phs, L)  # (nframes, L)

    spec = np.fft.rfft(frames, n=nfft, axis=1)                 # one-sided STFT
    ins_fre_raw = np.sum(np.abs(spec), axis=1) / nfft

    fc = 20.0
    b, a = butter(3, 2 * fc / fs)      # low-pass
    ins_fre = filtfilt(b, a, ins_fre_raw)

    return remove_spike(ins_fre)
