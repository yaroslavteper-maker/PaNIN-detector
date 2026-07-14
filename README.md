# PaNIN Detector

A native macOS app for pancreatic pathology. It combines whole-slide image (WSI)
viewing, region annotation, and an on-device machine-learning pipeline to help
detect **PaNIN** (Pancreatic Intraepithelial Neoplasia) lesions — from drawing
labels through training a classifier to rendering prediction heatmaps.

## Overview

PaNIN Detector loads gigapixel pathology slides via **OpenSlide**, lets you draw
polygon/lasso annotations that round-trip with **QuPath** as GeoJSON, and provides
a complete, offline ML workflow: sample feature patches from annotated regions,
extract embeddings with Apple **Vision** or a bundled **Core ML** model, train a
classifier, and run multi-pass inference visualized directly on the slide.

Everything runs locally on your Mac — no cloud, no data leaves the machine.

## Features

- **Whole-slide viewing** — smooth tiled rendering of `.svs`, `.tif`, `.ndpi`,
  `.mrxs`, `.scn`, and other OpenSlide formats, with zoom, pan, and fit-to-window.
- **Annotation** — freehand lasso and click-to-place polygon tools, per-class
  colors, and a sortable, editable annotation sidebar. Annotations auto-save to a
  `.geojson` sidecar next to each slide.
- **QuPath interoperability** — import and export QuPath-compatible GeoJSON
  (with correct Y-axis handling), plus export annotation regions as PNG/JPEG tiles.
- **Classification profiles** — manage reusable sets of classes (labels + colors);
  the default profile covers PaNIN-1/2/3, Normal, and Stroma.
- **Feature extraction** — pluggable extractors: built-in Apple Vision feature
  prints or installable Core ML pathology models (UNI, Virchow, CTransPath, …).
- **Patch bank** — sample patches on a configurable grid (size / stride / level)
  with white-background filtering; save, load, and merge banks of feature vectors.
- **Classifier training** — multinomial logistic regression (Accelerate/BLAS) or
  t-SNE embedding with a nearest-centroid classifier, complete with accuracy and
  per-class precision/recall/F1 metrics.
- **t-SNE visualization** — interactive 2D embedding plot with per-class centroids.
- **Prediction** — multi-pass, multi-scale inference over an annotation, rendered
  as a color-coded confidence heatmap on the slide.
- **In-app help** — a Help window (⌘?) documenting the full workflow.

## Requirements

- macOS with Xcode (built with SwiftUI, SwiftData, Vision, Core ML, and Accelerate).
- [OpenSlide](https://openslide.org) installed via [Homebrew](https://brew.sh):
  ```sh
  brew install openslide
  ```

## Building

Open `PaNIN detector.xcodeproj` in Xcode and build/run the `PaNIN detector` target.
The project uses file-system–synchronized groups, so source files are picked up
automatically.

## Workflow at a glance

1. **Open a slide** — File ▸ Open Slide… (⌘O).
2. **Annotate** — draw regions with the Lasso or Polygon tool and assign a class.
3. **Extract patches** — File ▸ Extract Patches… (⇧⌘X) to build a feature bank.
4. **Train** — Analysis ▸ Regression Classifier… (⌥⌘T) or t-SNE Classifier… (⌥⌘A).
5. **Predict** — select an annotation, then Analysis ▸ Classify on Annotation…
   (⌥⌘R) to overlay a prediction heatmap.

See the in-app **Help** window (⌘?) for detailed, step-by-step instructions.

## Repository layout

- `PaNIN detector/` — the macOS app (SwiftUI views, models, OpenSlide bridge).
- `PathologyFeatures/` — a standalone Swift package for pluggable feature
  extractors (Vision + Core ML), reusable outside the app.

---

*Research tooling for digital pathology. Not a medical device; not for diagnostic use.*
