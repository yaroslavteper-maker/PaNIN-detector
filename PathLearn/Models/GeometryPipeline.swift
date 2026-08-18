import Foundation

/// Off-main orchestration for geometry features: compute descriptors for a set
/// of annotations (extraction) and classify one annotation with a geometry
/// model (prediction). Prediction reuses the existing `PredictionPipeline.Report`
/// / `PatchPrediction` types so results flow through the same store, heatmap,
/// and panel as every other prediction.
nonisolated enum GeometryPipeline {

    enum PipelineError: Error, CustomStringConvertible {
        case describeFailed
        case featureDimMismatch(expected: Int, got: Int)
        var description: String {
            switch self {
            case .describeFailed:
                return "Could not compute geometry for this annotation (too little tissue or too few nuclei)."
            case .featureDimMismatch(let e, let g):
                return "Geometry dimension mismatch: model expects \(e), got \(g). Re-train the geometry model."
            }
        }
    }

    /// Compute descriptors for each annotation. Annotations that can't be
    /// described (too small / sparse) are skipped. Runs off-main.
    static func extract(slide: SlideImage,
                        slidePath: String,
                        slideName: String,
                        annotations: [Annotation],
                        progress: @Sendable (Int, Int) -> Void) -> [GeometryFeatureStore.Record] {
        var out: [GeometryFeatureStore.Record] = []
        let total = annotations.count
        for (i, ann) in annotations.enumerated() {
            if let features = PathologyGeometry.describe(slide: slide, annotation: ann) {
                out.append(GeometryFeatureStore.Record(
                    slidePath: slidePath,
                    slideName: slideName,
                    annotationID: ann.id,
                    classification: ann.classification,
                    features: features,
                    version: PathologyGeometry.version
                ))
            } else {
                print("[geometry] skipped annotation \(ann.id) (\(ann.classification)) — not describable")
            }
            progress(i + 1, total)
        }
        return out
    }

    /// Classify one annotation with a geometry model — just the label + probs,
    /// no painting. Used for batch grading of proposed regions.
    static func grade(slide: SlideImage,
                      annotation: Annotation,
                      classifier: MLClassifier) throws -> (label: String, probs: [String: Float]) {
        guard let features = PathologyGeometry.describe(slide: slide, annotation: annotation) else {
            throw PipelineError.describeFailed
        }
        guard features.count == classifier.featureDim else {
            throw PipelineError.featureDimMismatch(expected: classifier.featureDim, got: features.count)
        }
        return classifier.predict(features)
    }

    /// Classify one annotation with a geometry model and paint the verdict over
    /// a grid of tiles inside it (reusing the sheet's sampling config).
    static func predict(slide: SlideImage,
                        annotation: Annotation,
                        classifier: MLClassifier,
                        sampling: PredictionPipeline.PassConfig) throws -> PredictionPipeline.Report {
        guard let features = PathologyGeometry.describe(slide: slide, annotation: annotation) else {
            throw PipelineError.describeFailed
        }
        guard features.count == classifier.featureDim else {
            throw PipelineError.featureDimMismatch(expected: classifier.featureDim, got: features.count)
        }
        let (label, probs) = classifier.predict(features)
        let maxProb = probs.values.max() ?? 0

        // Diagnostic: raw descriptors + resulting probabilities. Compare the
        // features against this annotation's row in geometry_bank.json — if they
        // match but the label is wrong, the ACTIVE model is stale (retrain).
        let featStr = zip(PathologyGeometry.descriptorNames, features)
            .map { "\($0)=\(String(format: "%.4g", $1))" }.joined(separator: " ")
        let probStr = probs.sorted { $0.value > $1.value }
            .map { "\($0.key):\(String(format: "%.2f", $0.value))" }.joined(separator: " ")
        print("[geometry] predict ann=\(annotation.id) source=\(annotation.classification)")
        print("[geometry]   features: \(featStr)")
        print("[geometry]   probs: \(probStr) → \(label)")
        if classifier.featureMean == nil {
            print("[geometry]   ⚠️ classifier has NO standardization params — not a geometry model / stale. Retrain.")
        }

        let levelIdx = max(0, min(Int(sampling.level), Int(slide.levelCount) - 1))
        let lds = slide.levelDownsamples[levelIdx]
        var origins = PatchSampler.samplePatches(
            polygonOverlayY: annotation.points,
            slideDimensions: slide.dimensions,
            patchSizeLevel: sampling.patchSizeLevel,
            strideLevel: sampling.strideLevel,
            levelDownsample: lds
        )
        let patchLevel0 = Int((Double(sampling.patchSizeLevel) * lds).rounded())

        // Fallback: if the grid came up empty (tiny annotation / large stride),
        // paint a single tile at the annotation's mirrored bounding box.
        if origins.isEmpty {
            let slideH = slide.dimensions.height
            let data = annotation.points.map { CGPoint(x: $0.x, y: slideH - $0.y) }
            var minX = data[0].x, minY = data[0].y
            for p in data.dropFirst() { minX = min(minX, p.x); minY = min(minY, p.y) }
            origins = [(x: Int64(minX.rounded(.down)), y: Int64(minY.rounded(.down)))]
        }

        let predictions = origins.map { origin in
            PatchPrediction(
                dataX: origin.x, dataY: origin.y,
                sizeLevel0: max(patchLevel0, 1),
                predictedLabel: label, probabilities: probs, maxProbability: maxProb,
                passPatchSize: sampling.patchSizeLevel, passStride: sampling.strideLevel
            )
        }
        let passInfo = [PredictionPassInfo(
            patchSizeLevel: sampling.patchSizeLevel, strideLevel: sampling.strideLevel,
            level: Int32(levelIdx), levelDownsample: lds, classes: [label]
        )]
        print("[geometry] predict → \(label) @ \(maxProb) (\(predictions.count) tiles)")
        return PredictionPipeline.Report(
            predictions: predictions, perClass: [label: predictions.count],
            passes: passInfo, total: predictions.count, skipped: 0
        )
    }
}
