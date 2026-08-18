# Feature Extractors

PathLearn embeds each patch with a *feature extractor*. The built-in **Apple
Vision** feature print (2048-dim) needs no setup. Pathology foundation models
usually separate tissue classes much better and can be installed as Core ML
extractors.

## Installing an extractor

Place two files in the app's extractors folder
(`~/Library/Application Support/<bundle-id>/Extractors/`, opened via
**File ▸ Extractors…** in the app):

1. `<name>.mlpackage` — the Core ML model. Input: a `[1, 3, H, W]` Float
   tensor (normalized pixels, NCHW). Output: the embedding vector.
2. `<name>.paninextractor.json` — a descriptor telling the app how to drive it:

```json
{
  "identity" : { "kind" : "coreml", "name" : "phikon-v1", "revision" : 1 },
  "kind" : "coreml",
  "inputWidth" : 224,
  "inputHeight" : 224,
  "featureDim" : 768,
  "pixelNormalization" : {
    "meanRGB" : [0.485, 0.456, 0.406],
    "stdRGB" : [0.229, 0.224, 0.225],
    "scale" : 0.00392156862745098
  },
  "modelFilename" : "phikon.mlpackage"
}
```

The app applies `(pixel * scale - mean) / std` itself, so the model should
expect already-normalized input. Rescan with the ↻ button in any extractor
picker; the model is compiled once and cached.

The `identity` is stamped on every patch and classifier, so banks/models from
different extractors can't be mixed accidentally. If you replace a model file,
bump `revision` (or re-extract) to avoid silently mixing feature spaces.

## Ready-made downloads

The GitHub **Releases** page of this repo provides:

- **VISTA segmentation models** (3 × ~1 MB) — for Analysis ▸ Classify with
  VISTA; these go in the `VISTA/` folder next to `Extractors/`, not in it.
- **Phikon** (`phikon.mlpackage` + descriptor, ~164 MB) — Owkin's iBOT
  ViT-B/16 pathology encoder, 768-dim. Non-commercial research use.

## Converting a model yourself

Gated models (**UNI**, **UNI2**, **Virchow**) cannot be redistributed — you
must request access (e.g. on Hugging Face), then convert your own copy.

`Phikon/convert_phikon.py` is the working template: load the model in
PyTorch, wrap it so its forward returns the embedding (CLS token), trace with
`torch.jit.trace`, convert with `coremltools` using a
`TensorType(shape=(1, 3, 224, 224))` input, and verify the Core ML output
matches PyTorch. For a timm-based gated model like UNI the load step becomes:

```python
import timm
model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True,
                          init_values=1e-5, dynamic_img_size=True)
model.eval()
# forward already returns the embedding; wrap/trace/convert as in convert_phikon.py
```

Then write the descriptor with the model's dimensions (UNI: 1024-dim, UNI2:
1536-dim; both 224×224, ImageNet normalization) and drop both files into the
Extractors folder.

Environment used for conversion (see `Phikon/`):
Python 3.11, `torch`, `timm`/`transformers`, `coremltools`.

## Licenses

| Model | Source | Redistribution |
|-------|--------|----------------|
| Apple Vision | built into macOS | n/a |
| Phikon | [owkin/phikon](https://huggingface.co/owkin/phikon) | non-commercial research |
| UNI / UNI2 | Mahmood Lab (gated) | **no** — convert your own copy |
| Virchow | Paige (gated) | **no** — convert your own copy |
| CTransPath | public weights | convert per its license |
| VISTA / MicePan | [gelatinfrogs/MicePan-Segmentation](https://github.com/gelatinfrogs/MicePan-Segmentation) | published by authors; cite Ternes et al. 2020 |
