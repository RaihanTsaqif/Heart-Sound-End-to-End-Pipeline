"""
Feature extraction for Hong Tang's heart-sound SQA, ported MATLAB -> Python.

Two entry points, matching the two MATLAB feature sets:
  * get_features(phs, enve, fs)       -> full feature vector  (get_features.m)
  * get_features_tang(phs, enve, fs)  -> 10-feature subset    (get_features_Tang.m)

The 10-feature subset (get_features_tang) is the one the shipped classifiers
(entry_*_classification.m -> features_Tang.mat) actually use.

Dependencies: numpy, scipy, and PyWavelets (pywt) for the single db8 DWT call
inside the Mubarak feature. MATLAB's std() uses N-1, so ddof=1 throughout.
"""

import numpy as np
from scipy.signal import hilbert, czt, welch, resample_poly
from scipy.spatial.distance import pdist
from scipy.stats import zscore

EPS = np.finfo(float).eps


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _mstd(x):
    return np.std(np.asarray(x, float), ddof=1)


def xcorr_coeff_auto(x):
    """
    MATLAB xcorr(x, 'coeff') for a single signal (autocorrelation), returned
    full (length 2N-1), normalized so the zero-lag value is 1. Index N-1 is
    lag 0; slice [N-1:] for the single-sided autocorrelation.
    """
    x = np.asarray(x, float).ravel()
    r = np.correlate(x, x, mode="full")
    zero_lag = r[len(x) - 1]
    if zero_lag == 0:
        return r
    return r / zero_lag


def xcorr_coeff_cross(a, b):
    """
    MATLAB xcorr(a, b, 'coeff'): cross-correlation normalized by
    sqrt(sum(a^2) * sum(b^2)).
    """
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    r = np.correlate(a, b, mode="full")
    if denom == 0:
        return r
    return r / denom


# ---------------------------------------------------------------------------
# individual features
# ---------------------------------------------------------------------------
def get_kurtosis(x):
    """Port of getkurtosis.m (note: this is the *non-normalized* 4th/2nd moment
    ratio the author defines, not scipy.stats.kurtosis)."""
    x = np.asarray(x, float).ravel()
    x = x - np.mean(x)
    k1 = np.sum(x ** 4) / len(x)
    k2 = np.sum(x ** 2) / len(x)
    return k1 / (k2 ** 2 + EPS)


def _welch_psd(x, fs):
    """
    Emulates MATLAB pwelch(x, [], [], nfft, fs): 8 segments, 50% overlap,
    Hamming window, one-sided PSD, nfft = round(fs).

    NOTE (fidelity): MATLAB's default segmenting (8 sections @ 50% overlap ->
    segment length ~ len/4.5) is reproduced here. The *ratio* features below are
    fairly insensitive to the exact windowing, but if you later validate against
    MATLAB, this is the most convention-sensitive spot.
    """
    x = np.asarray(x, float).ravel()
    n = len(x)
    nfft = int(round(fs))
    nperseg = int(np.floor(n / 4.5))
    nperseg = max(1, min(nperseg, n))
    noverlap = nperseg // 2
    f, px = welch(
        x, fs=fs, window="hamming", nperseg=nperseg, noverlap=noverlap,
        nfft=max(nfft, nperseg), detrend=False, return_onesided=True,
        scaling="density",
    )
    return f, px


def get_energy_ratio(x, fre, fs):
    """Port of getEnergyRatio.m: fraction of PSD energy within [fre0, fre1]."""
    f, px = _welch_psd(x, fs)
    idx = (f >= fre[0]) & (f <= fre[1])
    return np.sum(px[idx]) / (np.sum(px) + EPS)


def get_freq_band_ratio(x, fre, fs):
    """Port of getFreqBandRatio.m: fraction of |FFT| magnitude within a band.

    Uses the full two-sided FFT with frequency axis 0..fs, exactly as MATLAB.
    """
    x = np.asarray(x, float).ravel()
    nfft = int(round(fs))
    Fx = np.abs(np.fft.fft(x, nfft))
    w = fs * np.arange(nfft) / nfft
    idx = (w >= fre[0]) & (w <= fre[1])
    return np.sum(Fx[idx]) / (np.sum(Fx) + EPS)


def get_feature_mubarak(hs, fs):
    """
    Port of getFeature_Mubarak.m (Mubarak et al., 2018). Two-level db8 DWT,
    keep the level-2 approximation coefficients, then compute:
      RMSSD - root mean square of successive differences
      RZC   - ratio of zero crossings to the ORIGINAL signal length

    MATLAB's dwt default extension mode is 'sym', which maps to pywt's
    'symmetric'.
    """
    import pywt  # local import so the rest of the module works without pywt

    hs = np.asarray(hs, float).ravel()
    ca1, _ = pywt.dwt(hs, "db8", mode="symmetric")   # level 1
    ca2, _ = pywt.dwt(ca1, "db8", mode="symmetric")  # level 2
    whs = np.asarray(ca2, float).ravel()

    dif_hs = np.diff(whs)
    RMSSD = np.sqrt(np.mean(dif_hs ** 2))

    hs1 = whs[:-1]
    hs2 = whs[1:]
    mhs = hs1 * hs2
    num_zcross = np.sum(mhs < 0)
    RZC = num_zcross / len(hs)   # NOTE: divided by original length, per MATLAB
    return RMSSD, RZC


def get_max_axcor_coef(x, fs):
    """Port of getMaxAxcorCoef.m: max of the single-sided autocorrelation over
    lags [0.3 s, 2 s]. x is the single-sided autocorrelation (index 0 = lag 0).
    """
    x = np.asarray(x, float).ravel()
    start = int(round(0.3 * fs))   # MATLAB 1-based start index
    end = int(round(2 * fs))       # MATLAB 1-based end index (inclusive)
    return np.max(x[start - 1:end])


def get_cycle_dur(x, fs):
    """Port of getCycleDur.m: cardiac cycle length (in samples) = location of the
    autocorrelation peak within lags [0.3 s, 2 s].
    """
    x = np.asarray(x, float).ravel()
    start = int(round(0.3 * fs))
    end = int(round(2 * fs))
    seg = x[start - 1:end]
    locs = int(np.argmax(seg)) + 1   # MATLAB max() returns a 1-based location
    return locs + start              # MATLAB: dur = locs + start_point


def get_aver_coef_std_coef(enve, dur):
    """
    Port of getAverCoefStdCoef.m: reshape the envelope into cardiac cycles of
    length `dur`, and compute the mean/std of the max normalized correlation
    between each cycle and its two neighbors. Fewer than 3 cycles -> (0, 2).
    """
    enve = np.asarray(enve, float).ravel()
    dur = int(dur)
    L = len(enve)
    Ncyc = int(np.floor(L / dur))
    if Ncyc < 3:
        return 0.0, 2.0

    Mat = np.reshape(enve[:Ncyc * dur], (dur, Ncyc), order="F")  # column-major
    coef = []
    for k1 in range(1, Ncyc - 1):  # MATLAB 2..Ncyc-1 -> 0-based cols 1..Ncyc-2
        d1 = xcorr_coeff_cross(Mat[:, k1], Mat[:, k1 - 1])
        coef.append(np.max(np.abs(d1)))
    for k1 in range(1, Ncyc - 1):
        d2 = xcorr_coeff_cross(Mat[:, k1], Mat[:, k1 + 1])
        coef.append(np.max(np.abs(d2)))
    coef = np.asarray(coef)
    return float(np.mean(coef)), float(np.std(coef, ddof=1))


def get_std_heart_rate(enve, fs):
    """
    Port of getStdHeartRate.m: sliding 3 s window (1 s hop); in each window,
    estimate heart rate from the autocorrelation peak (0 if the window is nearly
    silent). Returns (mean_hr, std_hr).
    """
    enve = np.asarray(enve, float).ravel()
    win_len = int(round(3 * fs))
    step = int(round(1 * fs))
    th = win_len / len(enve)
    total_energy = np.sum(enve ** 2)

    hr = []
    c = 0
    while True:
        k1 = c * step
        k2 = k1 + win_len
        if k2 <= len(enve):
            cur = enve[k1:k2]
            c += 1
            if np.sum(cur ** 2) / total_energy < 0.05 * th:
                hr.append(0.0)
            else:
                cur = cur - np.mean(cur)
                ac = xcorr_coeff_auto(cur)
                single = ac[len(cur) - 1:]
                hr.append(fs / get_cycle_dur(single, fs))
        else:
            break

    hr = np.asarray(hr) if hr else np.asarray([0.0])
    return float(np.mean(hr)), float(np.std(hr, ddof=1))


def get_samp_en(x, m, r):
    """
    Port of getSampEn_fast.m (Richman & Moorman sample entropy) using Chebyshev
    pdist. MATLAB zscore normalizes by N-1, so ddof=1.
    """
    x = np.asarray(x, float).ravel()
    x = zscore(x, ddof=1)
    N = len(x)
    Nm = N - m

    ym = np.array([x[i:i + m] for i in range(Nm)])
    ya = np.array([x[i:i + m + 1] for i in range(Nm)])
    if m == 1:
        ym = ym.reshape(-1, 1)

    d_m = pdist(ym, metric="chebyshev")
    cm = np.sum(d_m <= r) * 2.0 / (ym.shape[0] * (ym.shape[0] - 1))

    d_a = pdist(ya, metric="chebyshev")
    ca = np.sum(d_a <= r) * 2.0 / (ya.shape[0] * (ya.shape[0] - 1))

    return -np.log(ca / cm)


def get_svd_score(truncated_autocorrelation, fs):
    """
    Port of get_SVD_score.m (Springer/Kumar SVD-SQI): for a range of window
    sizes, stack the truncated autocorrelation into a matrix, take the SVD, and
    record the squared ratio of the 2nd to 1st singular values. Return the
    minimum over all window sizes.
    """
    tac = np.asarray(truncated_autocorrelation, float).ravel()
    rho = []
    start_window = int(round((215 / 500) * fs))
    stop_window = int(round(2 * fs))

    for T in range(start_window, stop_window + 1, 5):
        if T < 1:
            continue
        nwin = len(tac) // T
        if nwin < 1:
            continue
        Y = np.array([tac[w * T:(w + 1) * T] for w in range(nwin)])  # (nwin, T)
        S = np.linalg.svd(Y.T, compute_uv=False)   # svd of (T x nwin)
        if S.size == 1:
            rho.append(10.0)
        else:
            rho.append((S[1] / S[0]) ** 2)

    return float(min(rho)) if rho else 0.0


def get_cc_sqi(untruncated_autocorrelation, fs):
    """
    Port of get_ccSQI.m (Springer cosine-correlation SQI): find the heart-rate
    frequency whose fitted rectified cosine best matches the autocorrelation
    peaks, then return the correlation between the two (first 5 s).
    """
    uac = np.asarray(untruncated_autocorrelation, float).ravel()
    if len(uac) < 5 * fs:
        return 0.0

    n = len(uac)
    t = np.arange(n)
    frequency_range = np.arange(0.6, 2.33 + 1e-9, 0.01)

    sums = []
    for f in frequency_range:
        cos_peaks1 = np.arange(1, 6) * fs / f - round(fs * 0.12)
        cos_peaks2 = np.arange(1, 6) * fs / f + round(fs * 0.12)
        spread = []
        for i in range(5):
            lo = min(cos_peaks1[i], n)
            hi = min(cos_peaks2[i], n)
            spread.extend(range(int(round(lo)), int(round(hi)) + 1))
        spread = np.asarray(spread)
        spread = spread[(spread >= 1) & (spread <= n)] - 1  # 1-based -> 0-based
        sums.append(np.sum(uac[spread] ** 2))

    b = int(np.argmax(sums))
    f = frequency_range[b]

    oscil = np.cos(2 * np.pi * f / fs * t)
    oscil = (oscil > 0) * oscil

    max_peaks = int(round(5 * fs / f) + round(fs * 0.12))
    if max_peaks > n:
        max_peaks = n

    cc = np.corrcoef(uac[:max_peaks], oscil[:max_peaks])
    return float(cc[1, 0])


def get_degree_cycle(rx, min_cf, max_cf, fs):
    """
    Port of getDegree_cycle.m + fast_cfs.m (cyclostationary "degree of
    periodicity"). Computes the cycle-frequency spectrum via a chirp-z transform
    of the (mean-removed) Hilbert envelope, and returns
    max(cfs) / median(cfs) plus the spectrum itself.

    fast_cfs.m is a hand-rolled chirp-z transform; scipy.signal.czt uses the
    same X[k] = sum_n x[n] a^-n w^{nk} convention as MATLAB czt, so we call it
    directly with the identical w and a.
    """
    rx = np.asarray(rx, float).ravel()
    M = 200  # number of bins in the cycle-frequency domain

    x = np.abs(hilbert(rx))
    x = x - np.mean(x)

    w = np.exp(-1j * 2 * np.pi * (max_cf - min_cf) / (M * fs))
    a = np.exp(1j * 2 * np.pi * min_cf / fs)
    z = np.abs(czt(x, M, w, a))

    cfs = z
    degree_peak = np.max(cfs) / (np.median(cfs) + EPS)
    fz = (np.arange(len(z)) * (max_cf - min_cf) / len(z)) + min_cf
    return float(degree_peak), cfs, fz


# ---------------------------------------------------------------------------
# feature-vector assembly
# ---------------------------------------------------------------------------
def get_features_tang(phs, enve, fs):
    """
    Port of get_features_Tang.m -> the 10-feature vector used by the shipped
    binary/triple SVM classifiers.

    Order: [Kur_hs, EnergyRatioLow, EnergyRatioHigh, EnergyRatioMidd, Std_enve,
            Kur_corr, Max_correlation_coef, SampEn, SampEn_axcor, d_cfs]
    """
    phs = np.asarray(phs, float).ravel()
    enve = np.asarray(enve, float).ravel()

    Kur_hs = get_kurtosis(phs)

    ERL = get_energy_ratio(phs, [24, 144], fs)
    ERH = get_energy_ratio(phs, [200, fs / 2], fs)
    ERM = get_energy_ratio(phs, [144, 200], fs)

    Std_enve = _mstd(enve) / 1000.0

    enve0 = enve - np.mean(enve)
    ac_double = xcorr_coeff_auto(enve0)
    ac_single = ac_double[len(enve0) - 1:]

    Kur_corr = get_kurtosis(ac_double)
    Max_corr = get_max_axcor_coef(ac_single, fs)

    fsd = 30
    down_enve = resample_poly(enve0, fsd, int(round(fs)))
    SampEn = get_samp_en(down_enve / _mstd(down_enve), 2, 0.2)

    down_ax = resample_poly(ac_single, fsd, int(round(fs)))
    down_ax = down_ax / _mstd(down_ax)
    SampEn_ax = get_samp_en(down_ax, 2, 0.2)

    d_cfs, _, _ = get_degree_cycle(phs, 0.3, 2.5, fs)

    return np.array([Kur_hs, ERL, ERH, ERM, Std_enve,
                     Kur_corr, Max_corr, SampEn, SampEn_ax, d_cfs])


def get_features(phs, enve, fs):
    """
    Port of get_features.m -> the full feature vector.

    NOTE on count: the literal MATLAB get_features.m assembles 20 features (the
    order below). The shipped features.mat has 21 columns, which suggests an
    earlier version emitted a 2nd periodicity indicator from getDegree_cycle.
    The classifiers use the 10-feature set (get_features_tang) anyway, so this
    does not affect reproducing the paper's SVM results. This function returns
    the literal 20 from get_features.m.

    Order: [Kur_hs, EnergyRatioLow, EnergyRatioHigh, EnergyRatioMidd,
            MagniRatioLow, MagniRatioHigh, MagniRatioMidd, RMSSD, RZC,
            Std_enve, Kur_corr, Max_correlation_coef, Aver_coef, Std_coef,
            Hr_std, SampEn, SVD_SQI, ccSQI, SampEn_axcor, d_cfs]
    """
    phs = np.asarray(phs, float).ravel()
    enve = np.asarray(enve, float).ravel()

    Kur_hs = get_kurtosis(phs)

    fre_low = [24, 144]
    fre_high = [200, fs / 2]
    fre_midd = [144, 200]

    ERL = get_energy_ratio(phs, fre_low, fs)
    ERH = get_energy_ratio(phs, fre_high, fs)
    ERM = get_energy_ratio(phs, fre_midd, fs)

    MRL = get_freq_band_ratio(phs, fre_low, fs)
    MRH = get_freq_band_ratio(phs, fre_high, fs)
    MRM = get_freq_band_ratio(phs, fre_midd, fs)

    RMSSD, RZC = get_feature_mubarak(phs, fs)

    Std_enve = _mstd(enve) / 1000.0

    enve0 = enve - np.mean(enve)
    ac_double = xcorr_coeff_auto(enve0)
    ac_single = ac_double[len(enve0) - 1:]

    Kur_corr = get_kurtosis(ac_double)
    Max_corr = get_max_axcor_coef(ac_single, fs)

    cycle_dur = get_cycle_dur(ac_single, fs)
    Aver_coef, Std_coef = get_aver_coef_std_coef(enve0, cycle_dur)

    _, Hr_std = get_std_heart_rate(enve0, fs)

    fsd = 30
    down_enve = resample_poly(enve0, fsd, int(round(fs)))
    SampEn = get_samp_en(down_enve / _mstd(down_enve), 2, 0.2)

    SVD_SQI = get_svd_score(ac_single, fs)
    ccSQI = get_cc_sqi(ac_single, fs)

    down_ax = resample_poly(ac_single, fsd, int(round(fs)))
    down_ax = down_ax / _mstd(down_ax)
    SampEn_ax = get_samp_en(down_ax, 2, 0.2)

    d_cfs, _, _ = get_degree_cycle(phs, 0.3, 2.5, fs)

    return np.array([Kur_hs, ERL, ERH, ERM, MRL, MRH, MRM, RMSSD, RZC,
                     Std_enve, Kur_corr, Max_corr, Aver_coef, Std_coef,
                     Hr_std, SampEn, SVD_SQI, ccSQI, SampEn_ax, d_cfs])
