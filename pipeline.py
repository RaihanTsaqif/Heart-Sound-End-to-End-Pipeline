"""Four-stage end-to-end heart sound pipeline.

Stages:
  1. heart/no-heart gate using Tang SQA features + sklearn classifier
  2. murmur present/absent classifier using the CirCor CRNN ONNX model
  3. heart sound segmentation using the 30-epoch/1200-recording ONNX segmenter
  4. systolic/diastolic timing by murmur-band energy in segmented phases
"""
from __future__ import annotations

import csv
import glob
import os
import time
import warnings
from dataclasses import dataclass
from math import gcd
from typing import Any

import librosa
import numpy as np
import onnxruntime as ort
import scipy.signal
import soundfile as sf

from sqa_gate.features import get_features_tang
from sqa_gate.preprocessing import get_envelope_from_stft, pre_processing

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

HERE = os.path.dirname(os.path.abspath(__file__))
SQA_TRAINING_CSV = os.path.join(HERE, "sqa_gate", "heart_no_heart_training_features.csv")
MURMUR_ONNX = os.path.join(HERE, "onnx", "murmur_crnn_circor.onnx")
SEG_ONNX = os.path.join(HERE, "onnx", "segmenter_emissions_datascale1200_ep30.onnx")
CRF_NPZ = os.path.join(HERE, "onnx", "segmenter_crf_transitions_datascale1200_ep30.npz")

AUDIO_EXTS = ("*.wav", "*.flac", "*.ogg")

SQA_FS = 1000
SQA_MAX_SEC = 10
SQA_FEATURE_NAMES = [
    "Kur_hs", "EnergyRatioLow", "EnergyRatioHigh", "EnergyRatioMidd",
    "Std_enve", "Kur_corr", "MaxCorrCoef", "SampEn", "SampEn_axcor",
    "DegPeriodicity",
]

MURMUR_SR = 2000
MURMUR_HOP_MS = 15
MURMUR_WINDOW_FRAMES = 333

SEG_FS = 1000
SEG_FRAME = 2000
SEG_MIN_TAIL = 256
SEG_WINDOW = scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False)
SEG_TRUNC = (25, 200)
SEG_NFFT = 128
SEG_NAMES = ["S1", "Systole", "S2", "Diastole"]

MURMUR_BAND = (150, 450)


def collect_audio(paths: list[str]) -> list[str]:
    out: list[str] = []
    for path in paths:
        if os.path.isdir(path):
            for pattern in AUDIO_EXTS:
                out.extend(sorted(glob.glob(os.path.join(path, pattern))))
        else:
            out.append(path)
    return out


def read_audio_mono(path: str) -> tuple[np.ndarray, int]:
    audio, fs = sf.read(path)
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)
    if np.max(np.abs(audio)) > 1.5:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
    return audio, int(fs)


def resample_poly(sig: np.ndarray, source_fs: int, target_fs: int) -> np.ndarray:
    if source_fs == target_fs:
        return sig
    div = gcd(int(source_fs), int(target_fs))
    return scipy.signal.resample_poly(sig, target_fs // div, source_fs // div)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def viterbi(emissions: np.ndarray, start: np.ndarray, end: np.ndarray, trans: np.ndarray) -> np.ndarray:
    n_steps, n_tags = emissions.shape
    score = start + emissions[0]
    back = np.zeros((n_steps, n_tags), dtype=np.int64)
    for t in range(1, n_steps):
        choices = score[:, None] + trans
        back[t] = choices.argmax(axis=0)
        score = choices.max(axis=0) + emissions[t]
    score = score + end
    last = int(score.argmax())
    path = [last]
    for t in range(n_steps - 1, 0, -1):
        last = int(back[t, last])
        path.append(last)
    return np.array(path[::-1], dtype=np.int64)


@dataclass
class Thresholds:
    heart: float = 0.5
    murmur: float = 0.5


class HeartSoundPipeline:
    def __init__(self, thresholds: Thresholds | None = None, providers: list[str] | None = None):
        self.thresholds = thresholds or Thresholds()
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            ort.preload_dlls()
        except Exception:
            pass
        self.heart_models = self._train_heart_gate()
        self.murmur_session = ort.InferenceSession(MURMUR_ONNX, providers=self.providers)
        self.segmenter_session = ort.InferenceSession(SEG_ONNX, providers=self.providers)
        self.crf = np.load(CRF_NPZ)

    @staticmethod
    def _train_heart_gate():
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        X, y = [], []
        with open(SQA_TRAINING_CSV, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("ok") != "True":
                    continue
                try:
                    X.append([float(row[name]) for name in SQA_FEATURE_NAMES])
                except ValueError:
                    continue
                y.append(1 if row["category"] == "heart_reference" else 0)
        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.int64)
        logreg = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        )
        svm = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(kernel="rbf", class_weight="balanced", gamma="scale", probability=True),
        )
        logreg.fit(X_arr, y_arr)
        svm.fit(X_arr, y_arr)
        return {
            "logreg": logreg,
            "svm_rbf": svm,
            "training_rows": int(len(y_arr)),
            "training_heart_rows": int(y_arr.sum()),
            "training_no_heart_rows": int((y_arr == 0).sum()),
        }

    def heart_gate_features(self, path: str) -> tuple[np.ndarray, int, float]:
        sig, native_fs = read_audio_mono(path)
        sig = resample_poly(sig, native_fs, SQA_FS)
        sig = sig[: SQA_MAX_SEC * SQA_FS]
        phs = pre_processing(sig, SQA_FS)
        envelope = get_envelope_from_stft(phs, SQA_FS)
        feats = np.asarray(get_features_tang(phs, envelope, SQA_FS), dtype=np.float64)
        return feats, native_fs, len(sig) / SQA_FS

    def detect_heart_sound(self, path: str) -> dict[str, Any]:
        feats, native_fs, duration_used = self.heart_gate_features(path)
        x = feats.reshape(1, -1)
        p_logreg = float(self.heart_models["logreg"].predict_proba(x)[0, 1])
        p_svm = float(self.heart_models["svm_rbf"].predict_proba(x)[0, 1])
        detected = p_logreg >= self.thresholds.heart
        return {
            "detected": bool(detected),
            "decision_model": "logreg",
            "threshold": self.thresholds.heart,
            "p_heart": p_logreg,
            "p_heart_logreg": p_logreg,
            "p_heart_svm_rbf": p_svm,
            "native_fs": native_fs,
            "duration_used_s": duration_used,
            "features": {name: float(value) for name, value in zip(SQA_FEATURE_NAMES, feats)},
        }

    @staticmethod
    def murmur_feature(path: str) -> np.ndarray:
        audio, fs = read_audio_mono(path)
        if fs != MURMUR_SR:
            audio = librosa.resample(audio.astype(float), orig_sr=fs, target_sr=MURMUR_SR)
        b, a = scipy.signal.butter(5, [25, 400], btype="bandpass", fs=MURMUR_SR)
        audio = scipy.signal.lfilter(b, a, audio)
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=MURMUR_SR,
            n_mels=128,
            hop_length=int(MURMUR_SR * MURMUR_HOP_MS / 1000),
            win_length=int(MURMUR_SR * 25 / 1000),
        )
        feat = np.log(mel + 1e-8)
        feat = (feat - feat.mean()) / (feat.std() + 1e-12)
        d1 = librosa.feature.delta(feat)
        d2 = librosa.feature.delta(d1)
        return np.concatenate([feat, d1, d2], axis=0).astype(np.float32)

    @staticmethod
    def murmur_windows(feat: np.ndarray) -> list[np.ndarray]:
        frames = feat.shape[1]
        if frames < MURMUR_WINDOW_FRAMES:
            return [np.pad(feat, ((0, 0), (0, MURMUR_WINDOW_FRAMES - frames)), mode="wrap")]
        hop = MURMUR_WINDOW_FRAMES // 2
        starts = list(range(0, frames - MURMUR_WINDOW_FRAMES + 1, hop))
        if starts[-1] != frames - MURMUR_WINDOW_FRAMES:
            starts.append(frames - MURMUR_WINDOW_FRAMES)
        return [feat[:, start:start + MURMUR_WINDOW_FRAMES] for start in starts]

    def detect_murmur(self, path: str) -> dict[str, Any]:
        feat = self.murmur_feature(path)
        windows = np.stack(self.murmur_windows(feat)).astype(np.float32)
        logits = self.murmur_session.run(None, {"log_mel": windows})[0]
        probs = softmax(logits, axis=1).mean(axis=0)
        p_present = float(probs[0])
        p_absent = float(probs[1])
        detected = p_present >= self.thresholds.murmur
        return {
            "detected": bool(detected),
            "threshold": self.thresholds.murmur,
            "p_present": p_present,
            "p_absent": p_absent,
            "n_windows": int(windows.shape[0]),
        }

    @staticmethod
    def fsst_features(sig: np.ndarray) -> np.ndarray:
        import ssqueezepy as sq

        out = sq.ssq_stft(sig.astype(np.float64), window=SEG_WINDOW, n_fft=SEG_NFFT,
                          hop_len=1, fs=SEG_FS, modulated=False)
        tx = out[0]
        if hasattr(tx, "cpu"):
            tx = tx.cpu().numpy()
        freqs = np.asarray(out[2]).squeeze()
        mask = (freqs >= SEG_TRUNC[0]) & (freqs <= SEG_TRUNC[1])
        selected = np.asarray(tx)[mask, :]
        real = selected.real
        imag = selected.imag
        real = (real - real.mean()) / (real.std() + 1e-12)
        imag = (imag - imag.mean()) / (imag.std() + 1e-12)
        return np.concatenate([real, imag], axis=0).T.astype(np.float32)

    def segment(self, sig_1k: np.ndarray) -> np.ndarray:
        labels = []
        for start in range(0, len(sig_1k), SEG_FRAME):
            chunk = sig_1k[start:start + SEG_FRAME]
            if len(chunk) < SEG_MIN_TAIL:
                break
            feats = self.fsst_features(chunk)[None, :, :]
            emissions = self.segmenter_session.run(None, {"fsst": feats})[0][0]
            labels.append(viterbi(
                emissions,
                self.crf["start_transitions"],
                self.crf["end_transitions"],
                self.crf["transitions"],
            ))
        return np.concatenate(labels) if labels else np.zeros(0, dtype=np.int64)

    @staticmethod
    def timing_from_labels(sig_1k: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        sos = scipy.signal.butter(4, MURMUR_BAND, btype="band", fs=SEG_FS, output="sos")
        energy = scipy.signal.sosfiltfilt(sos, sig_1k) ** 2
        n = min(len(labels), len(energy))
        labels = labels[:n]
        energy = energy[:n]
        phase_energy = {
            name: (float(np.median(energy[labels == idx])) if (labels == idx).any() else 0.0)
            for idx, name in enumerate(SEG_NAMES)
        }
        phase_fraction = {
            name: (float((labels == idx).mean()) if len(labels) else 0.0)
            for idx, name in enumerate(SEG_NAMES)
        }
        sys_energy = phase_energy["Systole"]
        dia_energy = phase_energy["Diastole"]
        total = sys_energy + dia_energy + 1e-12
        systolic_pct = 100 * sys_energy / total
        diastolic_pct = 100 * dia_energy / total
        timing = "SYSTOLIC" if sys_energy >= dia_energy else "DIASTOLIC"
        n_cycles = int(np.sum((labels[1:] == 0) & (labels[:-1] != 0))) if len(labels) > 1 else 0
        return {
            "timing": timing,
            "systolic_pct": float(systolic_pct),
            "diastolic_pct": float(diastolic_pct),
            "phase_energy": phase_energy,
            "phase_fraction": phase_fraction,
            "n_cycles": n_cycles,
        }

    @staticmethod
    def labels_to_segments(labels: np.ndarray, fs: int = SEG_FS) -> list[dict[str, Any]]:
        if len(labels) == 0:
            return []
        segments = []
        start = 0
        current = int(labels[0])
        for idx in range(1, len(labels)):
            value = int(labels[idx])
            if value != current:
                segments.append({
                    "start": start / fs,
                    "end": idx / fs,
                    "label": current,
                    "name": SEG_NAMES[current],
                })
                start = idx
                current = value
        segments.append({
            "start": start / fs,
            "end": len(labels) / fs,
            "label": current,
            "name": SEG_NAMES[current],
        })
        return segments

    def run(self, path: str) -> dict[str, Any]:
        t_total = time.perf_counter()
        info = sf.info(path)
        result: dict[str, Any] = {
            "file": os.path.basename(path),
            "path": path,
            "duration_s": float(info.duration),
            "sample_rate": int(info.samplerate),
            "stages": {},
            "segments": [],
        }

        t0 = time.perf_counter()
        heart = self.detect_heart_sound(path)
        heart["runtime_s"] = time.perf_counter() - t0
        result["stages"]["heart_sound_gate"] = heart
        if not heart["detected"]:
            result["final_decision"] = "NO_HEART_SOUND"
            result["runtime_s"] = time.perf_counter() - t_total
            return result

        t0 = time.perf_counter()
        murmur = self.detect_murmur(path)
        murmur["runtime_s"] = time.perf_counter() - t0
        result["stages"]["murmur_classifier"] = murmur
        if not murmur["detected"]:
            result["final_decision"] = "HEART_SOUND_NO_MURMUR"
            result["runtime_s"] = time.perf_counter() - t_total
            return result

        audio, fs = read_audio_mono(path)
        sig_1k = resample_poly(audio, fs, SEG_FS)
        sig_1k = (sig_1k / (np.std(sig_1k) + 1e-9)).astype(np.float64)

        t0 = time.perf_counter()
        labels = self.segment(sig_1k)
        timing = self.timing_from_labels(sig_1k, labels)
        timing["runtime_s"] = time.perf_counter() - t0
        result["segments"] = self.labels_to_segments(labels)
        result["stages"]["segmentation"] = {
            "model": "segmenter_finetuned_circor_datascale1200_ep30",
            "n_segments": len(result["segments"]),
            "n_cycles": timing["n_cycles"],
            "phase_fraction": timing["phase_fraction"],
        }
        result["stages"]["timing"] = timing
        result["final_decision"] = f"MURMUR_{timing['timing']}"
        result["runtime_s"] = time.perf_counter() - t_total
        return result
