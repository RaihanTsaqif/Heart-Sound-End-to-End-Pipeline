# Model Assets

This folder stores the PyTorch checkpoints used by the full pipeline.

| File | Purpose |
|---|---|
| `murmur_crnn_circor.pth` | CirCor murmur classifier checkpoint from the previous pipeline |
| `segmenter_finetuned_circor_datascale1200_ep30.pth` | 30-epoch segmenter fine-tuned on 1,200 CirCor windows |

The default runtime uses the ONNX files in `../onnx/`. These PyTorch files are
kept for provenance and re-export.
