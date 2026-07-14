# PathologyFeatures

Pluggable image-feature extraction for histopathology workflows on macOS. A
single `FeatureExtractor` protocol with two stock implementations:

- **`VisionFeatureExtractor`** — Apple's built-in `VNGenerateImageFeaturePrintRequest`.
  No model file required, runs on the Neural Engine. Good baseline.

- **`CoreMLFeatureExtractor`** — Drives any `.mlpackage` whose output is a
  feature vector. Plug in a pathology-pretrained encoder (UNI, Virchow,
  CTransPath, RetCCL, …) once you've converted it to Core ML.

The package is intentionally **standalone**: no dependencies on the
PaNIN-detector app, no SwiftData, no UI. Bring your own pixels, get back a
feature vector.

## Usage

```swift
import PathologyFeatures

// Built-in (no model file needed)
let extractor: any FeatureExtractor = ExtractorRegistry.builtInVision()

// Or load a Core ML extractor from a sidecar JSON
let descriptorURL = URL(fileURLWithPath: "/path/to/uni-v1.paninextractor.json")
let extractor = try ExtractorRegistry.load(descriptor: descriptorURL)

let result = try await extractor.extract(cgImage)
print(result.dim, result.elementType, result.data.count)
print(extractor.identity.stringForm)
```

## The sidecar file (`*.paninextractor.json`)

Describes how to drive a Core ML model:

```json
{
  "identity": { "kind": "coreml", "name": "uni-v1", "revision": 1 },
  "kind": "coreml",
  "inputWidth": 224,
  "inputHeight": 224,
  "featureDim": 1024,
  "pixelNormalization": {
    "meanRGB": [0.485, 0.456, 0.406],
    "stdRGB":  [0.229, 0.224, 0.225],
    "scale":    0.00392156862745098
  },
  "modelFilename": "uni.mlpackage"
}
```

`modelFilename` is resolved relative to the descriptor file's directory.

## Converting a pretrained encoder

`Tools/convert_to_coreml.py` is a reference script for turning a PyTorch
encoder checkpoint (UNI / Virchow / CTransPath) into a Core ML `.mlpackage`
plus a matching descriptor JSON.

```sh
python3 -m pip install torch coremltools transformers huggingface_hub
python3 Tools/convert_to_coreml.py --model uni --output /path/to/output
```

## Bank-compatibility identity

Every extractor exposes an `ExtractorIdentity` with `(kind, name, revision)`.
If you persist feature vectors, **store the identity string alongside each
row** and refuse to mix vectors from different identities — features from
different encoders are not comparable.

## Platforms

macOS 14+.
