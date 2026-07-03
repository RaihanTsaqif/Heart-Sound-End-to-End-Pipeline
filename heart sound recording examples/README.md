# Heart Sound Recording Examples

Sample phonocardiogram (PCG) recordings you can download and run through the
pipeline to try it out. Upload any of these in the browser interface
(`python app/server.py`, then http://127.0.0.1:8010) or pass one on the command
line:

```bash
python run_pipeline.py "heart sound recording examples/85336_MV.wav"
```

All files are mono 16-bit WAV. The pipeline resamples internally, so the original
sample rate does not matter.

## Files

### CirCor recordings (4000 Hz)

From the CirCor DigiScope dataset. The name is `<patient-id>_<location>.wav`,
where the location is the auscultation site on the chest:

| File | Length | Auscultation site |
|---|---:|---|
| `85197_TV.wav` | 11.6s | Tricuspid valve |
| `85213_TV.wav` | 21.4s | Tricuspid valve |
| `85249_PV.wav` | 25.0s | Pulmonary valve |
| `85285_AV.wav` | 21.5s | Aortic valve |
| `85328_PV.wav` | 20.1s | Pulmonary valve |
| `85336_MV.wav` | 17.6s | Mitral valve |
| `85349_AV.wav` | 19.8s | Aortic valve |

Location codes: `AV` aortic, `MV` mitral, `PV` pulmonary, `TV` tricuspid.

### StetoQ recordings (4500 Hz)

Recorded locally with a StetoQ digital stethoscope — a different device from the
CirCor corpus, useful for checking that the pipeline generalizes beyond its
training data.

| File | Length |
|---|---:|
| `StetoQ_1.wav` | 30.0s |
| `StetoQ_2.wav` | 30.0s |
| `StetoQ_3.wav` | 30.0s |

These are provided for demonstration only. See the main
[README](../README.md) for how each stage makes its decision, and
`THIRD_PARTY_NOTICES.md` for dataset attribution.
