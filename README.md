# PathLearn

A native macOS app for digital pathology. It combines whole-slide image (WSI)
viewing, region annotation, and a complete on-device machine-learning
pipeline — from drawing labels through training a classifier to rendering
prediction heatmaps.

## Overview

PathLearn loads gigapixel pathology slides via **OpenSlide**, lets you draw
polygon/lasso annotations that round-trip with **QuPath** as GeoJSON, and
provides a complete, offline ML workflow: sample feature patches from
annotated regions, extract embeddings with Apple **Vision** or an installed
**Core ML** pathology model, train a classifier, and run inference visualized
directly on the slide.

Everything runs locally on your Mac — no cloud, no data leaves the machine.

## Features

- **Whole-slide viewing** — smooth tiled rendering of `.svs`, `.tif`, `.ndpi`,
  `.mrxs`, `.scn`, and other OpenSlide formats, with zoom, pan, and fit-to-window.
- **Annotation** — freehand lasso and click-to-place polygon tools, per-class
  colors, and a sortable, editable annotation sidebar. Annotations auto-save to a
  `.geojson` sidecar next to each slide.
- **QuPath interoperability** — import and export QuPath-compatible GeoJSON
  (with correct Y-axis handling), plus export annotation regions as PNG/JPEG tiles.
- **Classification profiles** — manage reusable sets of classes (labels + colors),
  including **Exclude / null** classes (e.g. lumens) whose look-alike patches are
  filtered out of training and prediction.
- **Feature extraction** — pluggable extractors: built-in Apple Vision feature
  prints or installable Core ML pathology foundation models (UNI, UNI2, Virchow,
  CTransPath, Phikon, …). See [EXTRACTORS.md](EXTRACTORS.md).
- **Patch bank** — sample patches on a configurable grid (size / stride / level)
  with white-background and minimum-nuclei filters; every patch stores its
  nucleus count; save, load, and merge banks of feature vectors.
- **Classifier training** — multinomial logistic regression (Accelerate/BLAS),
  per-patch or pooled per-annotation, one-vs-rest binarization, and null-class
  similarity exclusion; t-SNE embedding with a nearest-centroid classifier; and
  a geometry model trained on architectural descriptors (nuclear crowding,
  luminal complexity, …).
- **t-SNE visualization** — interactive 2D embedding plot with per-class
  centroids, white-patch and minimum-nuclei filters.
- **Prediction** — multi-pass, multi-scale inference over an annotation with
  per-patch or whole-annotation (mean / worst-focus) verdicts, minimum-nuclei
  gating, per-class heatmap visibility, saving predictions back as annotations,
  and region proposal + grading.
- **VISTA tissue segmentation** — runs the published MicePan 3-UNet ensemble
  (mouse pancreas H&E: neoplasia / metaplasia / normal acinar / stroma),
  converted to Core ML, as an independent comparison against your own classifier.
- **In-app help** — a Help window (⌘?) documenting the full workflow.

## Requirements

- macOS with Xcode (built with SwiftUI, SwiftData, Vision, Core ML, and Accelerate).
- [OpenSlide](https://openslide.org) installed via [Homebrew](https://brew.sh):
  ```sh
  brew install openslide
  ```

## Building

Open `PathLearn.xcodeproj` in Xcode and build/run the `PathLearn` target.
The project uses file-system–synchronized groups, so source files are picked up
automatically.

## Models

Model binaries are **not** stored in this repo. The
[Releases](../../releases) page provides the three VISTA segmentation models
and the Phikon extractor as downloads. Gated models (UNI, UNI2, Virchow) are
not redistributed — convert your own licensed copies using the instructions in
[EXTRACTORS.md](EXTRACTORS.md) and the scripts in `VISTA/` and `Phikon/`.

## Workflow at a glance

1. **Open a slide** — File ▸ Open Slide… (⌘O).
2. **Annotate** — draw your regions of choice with the Lasso or Polygon tool
   and assign a class.
3. **Extract patches** — File ▸ Extract Patches… (⇧⌘X) to build a feature bank.
4. **Train** — Analysis ▸ Regression Classifier… (⌥⌘T) or t-SNE Classifier… (⌥⌘A).
5. **Predict** — select an annotation, then Analysis ▸ Classify on Annotation…
   (⌥⌘R) to overlay a prediction heatmap, or Classify with VISTA (⌥⌘V).

See the in-app **Help** window (⌘?) for detailed, step-by-step instructions.

## Repository layout

- `PathLearn/` — the macOS app (SwiftUI views, models, OpenSlide bridge).
- `PathologyFeatures/` — a standalone Swift package for pluggable feature
  extractors (Vision + Core ML), reusable outside the app.
- `VISTA/` — Python scripts that reproduce and convert the MicePan/VISTA
  segmentation models to Core ML.
- `Phikon/` — Python script that converts the Phikon pathology encoder to
  Core ML.

## Attribution

- **VISTA / MicePan** segmentation models:
  [gelatinfrogs/MicePan-Segmentation](https://github.com/gelatinfrogs/MicePan-Segmentation)
  — Ternes et al., *Scientific Reports* (2020).
- **Phikon**: [Owkin](https://huggingface.co/owkin/phikon), non-commercial
  research use.
- **UNI / UNI2**: Mahmood Lab (gated; weights not redistributed here).

---

*Research tooling for digital pathology. Not a medical device; not for diagnostic use.*
