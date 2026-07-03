# ONNX Assets

| File | Purpose |
|---|---|
| `murmur_crnn_circor.onnx` | Murmur Present/Absent CRNN |
| `segmenter_emissions_datascale1200_ep30.onnx` | Segmenter emission network for S1/Systole/S2/Diastole |
| `segmenter_crf_transitions_datascale1200_ep30.npz` | CRF transition tables used by Python Viterbi decoding |

The segmenter's CRF decoder is not inside the ONNX graph. The runtime gets
emissions from ONNX and then runs Viterbi using the saved transition tables.

The heart/no-heart gate is not a neural network. It uses Tang SQA features and
two scikit-learn classifiers trained at startup from
`../sqa_gate/heart_no_heart_training_features.csv`.
