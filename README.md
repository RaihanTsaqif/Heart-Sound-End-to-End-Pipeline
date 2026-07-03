# Heart Sound End-to-End Pipeline

Local end-to-end phonocardiogram pipeline:

1. **Heart sound gate**: Tang SQA features decide heart sound vs no-heart input.
2. **Murmur classifier**: CirCor CRNN decides murmur Present vs Absent.
3. **Segmentation**: 30-epoch/1,200-window CirCor-fine-tuned model labels S1,
   Systole, S2, and Diastole.
4. **Timing**: median 150-450 Hz murmur-band energy is compared inside segmented
   systole vs diastole to call Systolic or Diastolic.

The two neural-network stages — the murmur classifier and the segmenter's
emission network — are **PyTorch models exported to ONNX** and served with ONNX
Runtime, so inference has no PyTorch dependency. The heart-sound gate is a
classical scikit-learn model and the timing stage is a rule-based DSP
calculation. The original `.pth` checkpoints are kept in `models/` for provenance
and re-export only (see [Re-export Segmenter ONNX](#re-export-segmenter-onnx)).

The project includes a local browser interface and a command-line runner. It is
structured as a standalone repo so it can be pushed to GitHub.

## Quick Start

```bash
pip install -r requirements.txt
python app/server.py
```

Then open:

```text
http://127.0.0.1:8010
```

Upload a `.wav`, `.flac`, or `.ogg` recording. The app shows each stage's result,
probabilities, runtime, and segmentation bands when a murmur is detected.

No recording handy? The [`heart sound recording examples/`](heart%20sound%20recording%20examples/)
directory has ready-to-use sample files (CirCor and StetoQ heart sounds) you can
download and run through the program. See its
[README](heart%20sound%20recording%20examples/README.md) for the file list.

Command-line use:

```bash
python run_pipeline.py path/to/audio.wav
python run_pipeline.py "heart sound recording examples/85336_MV.wav"
python run_pipeline.py path/to/folder --json results.json --csv results.csv
```

## Pipeline Decisions

### 1. Heart Sound Gate

The gate extracts the 10 Tang SQA features at 1 kHz using the code in
`sqa_gate/`, then trains two small scikit-learn models at startup from
`sqa_gate/heart_no_heart_training_features.csv`.

Upstream reference: [tanghongdlut/signal-quality-assessment-of-heart-sound-signal](https://github.com/tanghongdlut/signal-quality-assessment-of-heart-sound-signal).

The pipeline decision uses logistic regression probability:

```text
p_heart >= 0.5 -> heart sound
p_heart <  0.5 -> no-heart input
```

The RBF-SVM probability is also shown as a secondary check.

### 2. Murmur Classifier

The murmur classifier is the same CirCor CRNN from the previous pipeline. It runs
from `onnx/murmur_crnn_circor.onnx`.

Upstream reference: [SiyuLou/AutomaticHeartSoundClassification](https://github.com/SiyuLou/AutomaticHeartSoundClassification).

```text
P[Present] >= 0.5 -> murmur present
P[Present] <  0.5 -> murmur absent
```

### 3. Segmentation

The segmenter uses:

```text
models/segmenter_finetuned_circor_datascale1200_ep30.pth
onnx/segmenter_emissions_datascale1200_ep30.onnx
onnx/segmenter_crf_transitions_datascale1200_ep30.npz
```

The ONNX graph exports the emission network. Viterbi decoding is done in Python
with the saved CRF transition tables.

Upstream reference: [alvgaona/heart-sounds-segmentation](https://github.com/alvgaona/heart-sounds-segmentation).

### 4. Systolic / Diastolic Timing

This is rule-based, not a trained classifier:

1. Resample to 1 kHz.
2. Bandpass the signal to 150-450 Hz.
3. Square the filtered signal to estimate murmur-band energy.
4. Use the segmentation labels to split energy by phase.
5. Compare median energy in Systole vs Diastole.

```text
median systolic energy >= median diastolic energy -> SYSTOLIC
otherwise -> DIASTOLIC
```

## Performance Notes

These are the current project test numbers. The timing rule is intentionally not
reported as a strong accuracy metric because the available labelled timing data
is too imbalanced for diastolic murmurs.

### Heart Sound Gate

Training source:

```text
849 files total: 300 heart / 549 no-heart
```

Tests run on independent heart-sound datasets:

| Dataset | Files | Logistic regression HEART | RBF-SVM HEART |
|---|---:|---:|---:|
| CirCor (5-fold CV, held-out) | 300 | 96.7% | 98.7% |
| HSCT11 random sample | 50 | 49 / 50 (98.0%) | 48 / 50 (96.0%) |
| StetoQ local recordings | 4 | 4 / 4 (100.0%) | 4 / 4 (100.0%) |

Note that the CirCor row measures something different from the other two. CirCor 
is the gate's training source, so its number is a cross-validated recall. HSCT11 and StetoQ are independent 
datasets, so they measure generalization to unseen recordings and devices.

The tables above only cover the positive (heart) side. The gate is also scored on
non-heart audio — how often it correctly *rejects* recordings that are not heart
sounds. These come from the same 5-fold CV as the CirCor row (549 non-heart files:
lung sounds, speech, and random noise):

| Non-heart category | Files | Logistic regression rejected | RBF-SVM rejected |
|---|---:|---:|---:|
| Lung sounds | 336 | 99.4% | 100.0% |
| Clean speech | 105 | 96.2% | 100.0% |
| Noisy speech | 105 | 91.4% | 93.3% |
| Random noise | 3 | 66.7% | 66.7% |
| **Overall** | **549** | **97.1%** | **98.5%** |

Random noise is the weakest category, but with only 3 files that number is not
reliable and needs more data before it means anything. Lung sounds and speech —
the realistic sources of junk input — are rejected at 96-100%.

### Murmur Classifier



Held-out CirCor patients (`n=133`, threshold `0.5`):

| True \ Pred | Present | Absent |
|---|---:|---:|
| Present | 25 | 3 |
| Absent | 1 | 104 |

| Accuracy | Sensitivity | Specificity | Balanced accuracy | F1 |
|---:|---:|---:|---:|---:|
| 96.99% | 89.29% | 99.05% | 94.17% | 92.59% |

### Segmentation Model

30-epoch/1,200-window CirCor fine-tune, held-out test set (`n=120`
recordings, frame-level):

| Metric | Value |
|---|---:|
| Overall frame accuracy | 88.18% |
| Balanced frame accuracy | 87.21% |

| Class | Recall | Precision | F1 |
|---|---:|---:|---:|
| S1 | 86.40% | 85.67% | 86.04% |
| Systole | 87.32% | 87.09% | 87.20% |
| S2 | 83.71% | 83.02% | 83.36% |
| Diastole | 91.43% | 92.27% | 91.84% |

The ONNX export was checked against PyTorch emissions:

```text
max absolute difference: 1.15e-06
dynamic sequence input: verified
```

## Re-export Segmenter ONNX

The exporter is self-contained so it does not need the upstream segmentation
package's Linux-only `ssq` import:

```bash
pip install -r requirements-pytorch-export.txt
python pytorch/export_segmenter_onnx.py
```

On Windows, if the newest `onnx` wheel hits long-path issues, use `onnx==1.16.2`
or install ONNX into a short temporary path for export.

## Attribution

This project builds on the prior local pipeline and three upstream projects:

- Tang Hong's signal-quality assessment code for heart-sound signals
- SiyuLou's Automatic Heart Sound Classification CRNN
- Alvaro Gaona's heart-sounds-segmentation model

See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
