"""Export the 30-epoch/1200-recording segmenter checkpoint to ONNX.

This script is intentionally self-contained. The upstream segmentation package
imports the Linux-only `ssq` module at package import time, but exporting the
emission network only needs the model architecture and weights.
"""
import argparse
import os

import numpy as np
import torch
from torch import nn


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DEFAULT_CKPT = os.path.join(ROOT, "models", "segmenter_finetuned_circor_datascale1200_ep30.pth")
DEFAULT_OUTDIR = os.path.join(ROOT, "onnx")


class CRF(nn.Module):
    def __init__(self, num_tags=4):
        super().__init__()
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))


class SegmenterEmissionNet(nn.Module):
    def __init__(self, input_size=44, batch_size=1, hidden_size=240):
        super().__init__()
        self.register_buffer("h0", torch.zeros(2, batch_size, hidden_size))
        self.register_buffer("c0", torch.zeros(2, batch_size, hidden_size))
        self.lstm_1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                              bidirectional=True, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size,
                              bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size * 2, 4)
        self.crf = CRF(4)

    def forward(self, x):
        output, (hn, cn) = self.lstm_1(x, (self.h0, self.c0))
        output = self.relu(output)
        output = self.dropout(output)
        output, _ = self.lstm_2(output, (hn, cn))
        output = self.relu(output)
        output = self.dropout(output)
        return self.linear(output)


def load_state_dict(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    stripped = {}
    for key, value in state.items():
        if key.startswith("model."):
            key = key[len("model."):]
        if key in {"h0", "c0"}:
            continue
        stripped[key] = value
    return stripped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    model = SegmenterEmissionNet().eval()
    state = load_state_dict(args.checkpoint)
    model.load_state_dict(state, strict=False)

    os.makedirs(args.outdir, exist_ok=True)
    onnx_path = os.path.join(args.outdir, "segmenter_emissions_datascale1200_ep30.onnx")
    crf_path = os.path.join(args.outdir, "segmenter_crf_transitions_datascale1200_ep30.npz")

    dummy = torch.randn(1, 2000, 44)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["fsst"], output_names=["emissions"],
        dynamic_axes={"fsst": {1: "seq"}, "emissions": {1: "seq"}},
        opset_version=17, dynamo=False,
    )

    np.savez(
        crf_path,
        start_transitions=model.crf.start_transitions.detach().numpy(),
        end_transitions=model.crf.end_transitions.detach().numpy(),
        transitions=model.crf.transitions.detach().numpy(),
    )

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(onnx_path))
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out_onnx = sess.run(None, {"fsst": dummy.numpy()})[0]
    with torch.no_grad():
        out_torch = model(dummy).numpy()
    d2 = torch.randn(1, 1234, 44)
    out2 = sess.run(None, {"fsst": d2.numpy()})[0]

    print(f"emissions PyTorch vs ONNX max abs diff: {np.abs(out_onnx - out_torch).max():.2e}")
    print(f"dynamic-seq check ok: input {tuple(d2.shape)} -> emissions {out2.shape}")
    print(f"checkpoint -> {args.checkpoint}")
    print(f"saved -> {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.2f} MB)")
    print(f"saved -> {crf_path}")


if __name__ == "__main__":
    main()
