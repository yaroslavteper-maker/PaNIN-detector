# PaNIN Detector — Project Summary

A macOS whole-slide-image (WSI) annotation and machine-learning pathology tool for
pancreatic tissue analysis (PaNIN — Pancreatic Intraepithelial Neoplasia). The app
combines real-time slide viewing (OpenSlide), polygon annotation with QuPath-compatible
GeoJSON, feature extraction (Vision or Core ML), patch sampling, classifier training,
t-SNE embedding visualization, and multi-pass inference — all in a single native app.

> **Status:** The entire application was built out in the current working session on top
> of the initial commit. All ~50 source files, the `PathologyFeatures` Swift package, and
> project configuration are staged and ready for review.

---

## 1. Application Overview

| Area | Description |
|------|-------------|
| **Platform** | macOS, SwiftUI + AppKit hybrid |
| **Purpose** | View whole-slide pathology images, annotate regions, train ML classifiers, and run inference to detect PaNIN lesions |
| **Persistence** | SwiftData (patch feature store), JSON sidecars (annotations, banks, models, profiles), UserDefaults (current profile) |
| **Compute** | Accelerate/BLAS (logistic regression, t-SNE), Vision & Core ML (feature extraction), OpenSlide (WSI decoding) |

**Key entry points**

- `PaNIN_detectorApp.swift` — app lifecycle, SwiftData container for `MLPatch`, menu commands (File / Analysis), window groups for the Annotation Preview and t-SNE Plot.
- `ContentView.swift` — main UI orchestration; binds the slide canvas, annotation sidebar, and ML panels; dispatches menu commands via `NotificationCenter`.
- `SlideDocumentView.swift` — the AppKit NSView hierarchy: tiled slide layer + annotation overlay.

**Architecture highlights**

- SwiftUI for panels/sheets/sidebars; AppKit (`NSScrollView`, `CATiledLayer`, custom `NSView`) for low-level zoom/pan and drawing.
- `@Observable` stores drive reactive UI updates.
- Heavy work (training, t-SNE, prediction, patch extraction) runs in detached tasks with progress callbacks; SwiftData writes hop back to the main actor.
- Loose coupling via `Notification.Name` events for menu dispatch.

---

## 2. Slide Viewing & OpenSlide Integration

| File | Role |
|------|------|
| `Models/SlideImage.swift` | Thread-safe wrapper over the OpenSlide C library. Opens `.svs`, `.tif`, `.ndpi`, `.mrxs`, `.scn`, and other WSI formats. Exposes level count, downsamples, level dimensions, properties, and `readRegion(...)` returning a BGRA sRGB `CGImage`. |
| `OpenSlide/COpenSlideShim.h`, `OpenSlide/module.modulemap` | C shim exposing OpenSlide to Swift. Dependency installed via Homebrew. |
| `Views/SlideCanvasView.swift` | SwiftUI `NSViewRepresentable` bridging to the AppKit scroll view. |
| `Views/SlideScrollView.swift` | `NSScrollView` with magnification (≈0.005×–20×), ⌘+scroll zoom, native pan. |
| `Views/SlideDocumentView.swift` | Hosts the `CATiledLayer` slide renderer + annotation overlay; drives tile reads at the level matching the current zoom. |
| `Models/RenderSettings.swift` | Stroke thickness/color, fill opacity, selected-annotation emphasis. |

**Rendering:** a `CATiledLayer` (~512×512 tiles) reads OpenSlide regions on a background queue, picking the resolution level appropriate to the zoom scale.

**⚠️ Y-flip caveat:** the OpenSlide data origin and CoreAnimation tile context are Y-up, while the annotation overlay is Y-down (flipped `NSView`). The tile layer flips Y (`slideH - y`) when reading, and all coordinate boundaries (GeoJSON import/export, patch sampling, annotation export) mirror Y to stay consistent. Annotations live in mirrored space relative to QuPath.

---

## 3. Annotation System

**Data model**

| File | Role |
|------|------|
| `Models/Annotation.swift` | Polygon in base-level slide pixels; classification label, RGB color, optional name, visibility. Computes area (shoelace), bounding box, display strings. |
| `Models/AnnotationStore.swift` | `@Observable` store bound to a per-slide `.geojson` sidecar — auto-loads on open, saves on mutation. Add / update / remove / rename / reclassify / set visibility. |
| `Models/GeoJSON.swift` | Encodes/decodes a **QuPath-dialect** GeoJSON `FeatureCollection` (classification name + color, name, visibility). Y is flipped at the import/export boundary so QuPath's top-left origin aligns with the app. |
| `Models/Classification.swift` | `Classification` (id, name, color) and `ClassificationProfile`; default "Pancreatic Pathology" profile with PaNIN-1/2/3, Normal, Stroma. |
| `Models/AnnotationExporter.swift` | Exports each annotation region as PNG/JPEG to `<baseDir>/<slide>/<class>/<name>.<ext>`; bounding-box or square (white-padded) crop; async with progress report. |

**UI**

| File | Role |
|------|------|
| `Views/AnnotationOverlayView.swift` | Mouse input + drawing. Three tools: **Pan**, **Lasso** (free-form drag), **Polygon** (click vertices; Return/Esc/Backspace/close). Renders filled annotations and the prediction heatmap. |
| `Views/AnnotationSidebar.swift` | Tabbed sidebar: Annotations / Classes / Bank / Classifier. |
| `Views/AnnotationEditorSheet.swift`, `Views/AnnotationPreviewWindow.swift` | Edit classification/name/color; preview an annotation's cropped region. |
| `Views/LabelPickerSheet.swift` | Choose/create a classification when committing a shape. |
| `Views/ClassesListView.swift`, `Views/ClassEditorSheet.swift` | Manage the classification profile. |
| `Views/ExportOptionsSheet.swift` | PNG/JPEG, crop shape, level, padding for annotation export. |

---

## 4. Machine-Learning Pipeline

The largest subsystem: feature extraction → patch sampling/storage → embedding → training → inference.

### 4a. Feature Extraction

| File | Role |
|------|------|
| `Models/FeatureExtractor.swift` | Apple Vision `VNGenerateImageFeaturePrintRequest` (299×299 in, ~2048 floats). OS-tied revision stored per patch. Identity `vision:vision:r<rev>`. |
| `Models/CoreMLFeatureExtractor.swift` | Runs a user-supplied `.mlpackage` (UNI, Virchow, CTransPath, …) described by a JSON descriptor; compiles/caches `.mlmodelc`; configurable ImageNet-style normalization. Identity `coreml:<name>:r<rev>`. |
| `Models/AnyFeatureExtractor.swift` | Protocol abstracting Vision & Core ML extractors → `(data, dim, elementType, identity)`. |
| `Models/ExtractorRegistry.swift` | `@Observable` discovery of built-in Vision + user Core ML extractors (`*.paninextractor.json` sidecars in Application Support); caches instantiated extractors. |
| `Models/ExtractorDescriptor.swift` | JSON descriptor: identity, kind, input/output dims, feature dim, pixel normalization, model filename. |

**Extractor identity** (`kind:name:r<rev>`) is stored on every patch and classifier so incompatible feature spaces can never be mixed.

### 4b. Patch Sampling & Extraction

| File | Role |
|------|------|
| `Models/MLPatch.swift` | SwiftData `@Model`: classification, origin (x,y,level), size, feature `Data`, feature dim, element type, extractor identity/revision, white fraction. Legacy patches fall back to a Vision identity. |
| `Models/PatchSampler.swift` | Turns a polygon into grid patch origins (point-in-polygon ray casting) in level-0 data-Y space; cheap bbox estimator for UI preview. |
| `Models/PatchExtractionPipeline.swift` | Samples patches per annotation, reads from OpenSlide, computes white fraction, skips over-white patches, extracts features, writes `MLPatch` rows; async with a per-class saved/skipped report. |

### 4c. Embedding & Classifiers

| File | Role |
|------|------|
| `Models/TSNE.swift` | Symmetric t-SNE (van der Maaten 2008) over patch features via Accelerate/BLAS; configurable perplexity, iterations, early exaggeration, learning rate, momentum, seed. Returns 2D points + per-class centroids. |
| `Models/Centroids.swift` | Class centroids (mean feature + plot position + count); nearest-centroid classification; conversion to an `MLClassifier` via temperature-scaled softmax weights. |
| `Models/EmbeddingStore.swift`, `Models/EmbeddingSnapshot.swift` | Holds the current embedding + centroids + progress; `EmbeddingSnapshot` serializes the plot for embedding-aware classifiers. |
| `Models/LogisticRegression.swift` | Multinomial softmax via batch gradient descent (Accelerate `sgemm`, stable softmax, L2, cross-entropy logging). |
| `Models/ClassifierTrainer.swift` | Loads patches (filters by white fraction, enabled classes, extractor), optional one-vs-rest, stratified train/val split, validates uniformity, trains, returns an `MLClassifier` with metrics. |
| `Models/MLClassifier.swift` | Trained model: weights (D×K) + biases, class labels, feature dim, metrics, kind (logistic/centroid), extractor identity, optional embedding snapshot; `predict(...)` → label + softmax probs. |
| `Models/TrainingMetrics.swift` | Train/val accuracy, per-class precision/recall/F1, confusion matrix, counts, final loss, timestamp, revision. |

### 4d. Prediction (Inference)

| File | Role |
|------|------|
| `Models/PatchPrediction.swift` | One classifier output: data-space origin, size, predicted label, probabilities, max probability, pass config. |
| `Models/PredictionPipeline.swift` | **Multi-pass** inference — classes grouped by `(patchSize, stride, level)`; each unique tuple is a pass. Samples, extracts, classifies, keeps predictions ≥ min confidence; async with progress + per-pass counts. |
| `Models/PredictionStore.swift` | `@Observable` prediction results, source annotation, per-class colors, min probability, progress. |
| `Views/PredictAnnotationSheet.swift` | Configure and launch inference over an annotation. |

Predictions are rendered as a per-class alpha-blended heatmap in the annotation overlay.

### 4e. Serialization & Stores

| File | Role |
|------|------|
| `Models/BankSerializer.swift` | JSON snapshot of the patch bank (base64 features); replace/append import; versioned with legacy fallback. |
| `Models/ModelSerializer.swift` | JSON classifier file (`.cl`, legacy `.paninmodel.json`): weights/biases (base64), metrics, extractor identity, kind, embedding snapshot. |
| `Models/MLBankStore.swift` | `@Observable` bank stats (per-class, per-extractor counts); refresh/clear from SwiftData. |
| `Models/MLClassifierStore.swift` | `@Observable` active classifier + training state. |
| `Models/ProfileStore.swift` | Load/save classification profile (JSON + UserDefaults); manage classes. |

### 4f. t-SNE & ML Views

| File | Role |
|------|------|
| `Views/TSNEConfigSheet.swift`, `Views/TSNEPlotWindow.swift` | Configure t-SNE (extractor, sample cap, white-fraction cap, hyperparams) and view the interactive color-coded scatter + centroids. |
| `Views/TrainModelSheet.swift` | Choose extractor, enabled classes, one-vs-rest, hyperparameters. |
| `Views/ExtractPatchesSheet.swift` | Choose level, patch size, stride, extractor, scope. |
| `Views/ExtractorsListSheet.swift` | Browse available Vision + Core ML extractors. |
| `Views/MLBankPanel.swift`, `Views/MLModelPanel.swift` | Bank summary counts; active classifier info, metrics, and actions. |

---

## 5. PathologyFeatures Swift Package

A standalone package mirroring the app's extractor types for reuse: a public `FeatureExtractor` protocol, `VisionFeatureExtractor`, `CoreMLFeatureExtractor`, an `ExtractorRegistry` factory (`builtInVision()`, `load(descriptor:)`), and a public `ExtractorDescriptor`.

---

## 6. Notable Implementation Details

- **Y-coordinate mirroring** consistently applied across tiling, overlay, GeoJSON, patch sampling, and export (see the caveat in §2).
- **Backward compatibility:** optional `extractorIdentity`/`kind` fields fall back to legacy Vision/logistic defaults; versioned bank/model files still load older formats.
- **Feature element types** stored as raw `VNElementType` (float32/float16); decoders validate uniformity.
- **Whiteness filtering** (R,G,B ≥ 220) computed per patch, used to skip background during extraction/training/t-SNE.
- **Multi-pass prediction** enables multi-scale strategies (e.g. lesion vs. architecture) by grouping classes by patch geometry.
- **Centroid → softmax conversion** turns nearest-centroid classification into a standard `MLClassifier` via temperature-scaled logits.
- **Async off-main-actor** compute with progress callbacks; SwiftData mutations marshalled back to the main actor.

---

## 7. Dependencies

- **OpenSlide** (Homebrew) — whole-slide image decoding.
- **SwiftData** — patch feature persistence.
- **Accelerate / BLAS** — logistic regression, t-SNE.
- **Vision** — `VNGenerateImageFeaturePrintRequest`.
- **Core ML** — user-supplied pathology foundation models.
- **SwiftUI / AppKit / Foundation** — app UI and platform.

---

## 8. Capabilities Delivered

- ✅ Load, zoom, and pan multi-resolution slides (OpenSlide + `CATiledLayer`)
- ✅ Draw polygon/lasso annotations with QuPath-compatible GeoJSON import/export
- ✅ Manage classification profiles (classes, colors, names)
- ✅ Extract feature patches on a configurable grid (size / stride / level)
- ✅ Pluggable feature extractors (built-in Vision + user Core ML models)
- ✅ Train logistic-regression and centroid classifiers with full metrics
- ✅ Visualize feature embeddings via t-SNE
- ✅ Run multi-pass, multi-scale inference and render prediction heatmaps
- ✅ Persist profiles, patch banks, and classifiers to disk
- ✅ Export annotation regions as image tiles
