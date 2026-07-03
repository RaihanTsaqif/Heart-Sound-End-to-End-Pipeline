# Third-Party Notices and Attributions

This project builds on three upstream projects:

- Tang Hong's signal-quality assessment code for heart-sound signals.
- SiyuLou's AutomaticHeartSoundClassification project.
- Alvaro Gaona's heart-sounds-segmentation project.

The two neural-network upstream projects are released under the **MIT License**.
Their architectures, trained weights, ONNX exports, and related scripts in this
repository are derived works of those architectures. Full MIT license texts are
reproduced below.

The Tang signal-quality assessment repository describes academic use on its
GitHub page, but no OSI-style license file was observed there at the time this
notice was written. Treat the Tang-derived `sqa_gate/` code as attribution-
required academic/research-use material unless you obtain clearer permission
from the original author.

---

## 1. Heart-Sound Signal-Quality Assessment Gate

- **Used for:** Tang SQA feature extraction used by the heart/no-heart front
  gate.
- **Files in this repo derived from it:** `sqa_gate/features.py`,
  `sqa_gate/preprocessing.py`, and the feature names/logic used by
  `pipeline.py`.
- **Original project:** tanghongdlut / signal-quality-assessment-of-heart-sound-signal
  <https://github.com/tanghongdlut/signal-quality-assessment-of-heart-sound-signal>
- **Original author notice:** Hong Tang, School of Biomedical Engineering,
  Dalian University of Technology. The upstream README describes the MATLAB code
  as assessing heart-sound signal quality, extracting multi-domain features, and
  performing binary/triple quality classification for academic use.
- **License note:** no explicit MIT/BSD/Apache/GPL-style license file was
  observed in the GitHub repository. Before publishing this repo publicly, review
  whether the Tang-derived Python port should be included, replaced with a
  clean-room implementation, or distributed only with explicit permission.

---

## 2. Murmur Classification Model - CRNN

- **Used for:** the heart-murmur Present/Absent CRNN architecture and the
  training/evaluation framework it is based on.
- **Files in this repo derived from it:** `models/murmur_crnn_circor.pth`,
  `onnx/murmur_crnn_circor.onnx`, the CRNN config in `configs/`, and the murmur
  inference path in `pipeline.py`.
- **Original project:** SiyuLou / AutomaticHeartSoundClassification
  <https://github.com/SiyuLou/AutomaticHeartSoundClassification>
- **Note:** that project is itself based on the `pytorch-template` by victoresque
  (<https://github.com/victoresque/pytorch-template>, MIT License).

```text
MIT License

Copyright (c) 2022 SiyuLou

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 3. Heart-Sound Segmentation Model - LSTM + CRF

- **Used for:** the S1/Systole/S2/Diastole segmentation architecture, the FSST
  feature transform, and the training/evaluation framework.
- **Files in this repo derived from it:**
  `models/segmenter_finetuned_circor_datascale1200_ep30.pth`,
  `onnx/segmenter_emissions_datascale1200_ep30.onnx`,
  `onnx/segmenter_crf_transitions_datascale1200_ep30.npz`,
  `pytorch/export_segmenter_onnx.py`, and the segmentation/timing path in
  `pipeline.py`.
- **Original project:** alvgaona / heart-sounds-segmentation
  <https://github.com/alvgaona/heart-sounds-segmentation>

```text
MIT License

Copyright (c) 2019 Alvaro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Datasets

The models were trained/evaluated on the following public datasets, each governed
by its own data-use terms:

- **CirCor DigiScope Phonocardiogram Dataset** - Oliveira et al., via PhysioNet.
  <https://physionet.org/content/circor-heart-sound/>
- **David Springer heart-sound segmentation data** - used by the segmentation
  project above for training the segmenter.

Datasets are not included in this repository; download them from the sources
above under their respective licenses.

## Other Dependencies

This project also uses standard open-source Python packages including
onnxruntime, NumPy, SciPy, librosa, ssqueezepy, scikit-learn, PyWavelets, and
soundfile. See each package's distribution for details.
