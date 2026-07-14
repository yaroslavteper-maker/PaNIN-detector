import Foundation
import Accelerate

/// Per-class centroid in both the original feature space (for downstream
/// nearest-centroid classification) and the 2D t-SNE plot space (for
/// visualisation of where the class "centre of mass" sits).
struct Centroid: Sendable, Identifiable {
    let label: String
    /// Mean of the feature vectors of patches in this class (length = featureDim).
    /// This is what a Phase-2 nearest-centroid classifier scores against.
    let featureCentroid: [Float]
    /// Mean of the t-SNE coordinates of class members. Used only for the
    /// plot overlay — t-SNE is non-linear, so this is the visual centre of
    /// the cluster, not the projection of `featureCentroid`.
    let plotPosition: SIMD2<Float>
    let count: Int

    var id: String { label }
}

/// Set of per-class centroids computed from one embedding run. Tied to the
/// extractor identity so we never mix centroids from incompatible feature
/// spaces. Carries a `TrainingMetrics` snapshot of how well the centroids
/// classify the bank that produced them — used as the `metrics` field when
/// promoted to an `MLClassifier`.
struct CentroidSet: Sendable {
    let centroids: [Centroid]
    let extractorIdentity: ExtractorIdentity
    let featureDim: Int
    let metrics: TrainingMetrics
    let createdAt: Date

    var labels: [String] { centroids.map(\.label) }
    var trainAccuracy: Float { metrics.trainAccuracy }

    /// Nearest-centroid classification in feature space. Returns the label of
    /// the closest centroid by squared Euclidean distance, plus the raw
    /// per-class distances (squared) in label order.
    /// Phase-2 prediction pipeline will call this.
    func classify(_ feature: [Float]) -> (label: String, distances: [Float])? {
        guard feature.count == featureDim else { return nil }
        var bestIdx = 0
        var bestDist: Float = .infinity
        var dists = [Float](repeating: 0, count: centroids.count)
        for (i, c) in centroids.enumerated() {
            var d: Float = 0
            for k in 0..<featureDim {
                let v = feature[k] - c.featureCentroid[k]
                d += v * v
            }
            dists[i] = d
            if d < bestDist {
                bestDist = d
                bestIdx = i
            }
        }
        return (centroids[bestIdx].label, dists)
    }
}

enum CentroidComputer {
    /// Compute per-class centroids from a flat row-major feature matrix and a
    /// matching set of t-SNE plot points. Also runs nearest-centroid
    /// classification over the same patches to produce a `TrainingMetrics`
    /// snapshot — useful both as a separability diagnostic and to populate
    /// the metrics field when promoted to an MLClassifier.
    static func compute(features: [Float],
                       dim: Int,
                       labels: [String],
                       plotPoints: [SIMD2<Float>],
                       extractorIdentity: ExtractorIdentity) -> CentroidSet {
        precondition(features.count == labels.count * dim,
                     "feature count must equal labels × dim")
        precondition(plotPoints.count == labels.count,
                     "plot points and labels must align")

        // 1. Group by class.
        var byClass: [String: [Int]] = [:]
        for (i, lbl) in labels.enumerated() {
            byClass[lbl, default: []].append(i)
        }

        // 2. Build centroid for each class (sorted for stable ordering).
        var result: [Centroid] = []
        result.reserveCapacity(byClass.count)
        for label in byClass.keys.sorted() {
            let indices = byClass[label]!
            var fcMean = [Float](repeating: 0, count: dim)
            for idx in indices {
                let base = idx * dim
                for k in 0..<dim { fcMean[k] += features[base + k] }
            }
            let invN: Float = 1.0 / Float(indices.count)
            var s = invN
            fcMean.withUnsafeMutableBufferPointer { p in
                vDSP_vsmul(p.baseAddress!, 1, &s, p.baseAddress!, 1, vDSP_Length(dim))
            }
            var px: Float = 0, py: Float = 0
            for idx in indices {
                px += plotPoints[idx].x
                py += plotPoints[idx].y
            }
            px *= invN; py *= invN

            result.append(Centroid(
                label: label,
                featureCentroid: fcMean,
                plotPosition: SIMD2<Float>(px, py),
                count: indices.count
            ))
        }

        // 3. Run nearest-centroid classification on the same bank to fill in
        // the metrics field. O(N × K × D) but only once per t-SNE run.
        let metrics = computeMetrics(
            features: features, dim: dim,
            labels: labels,
            centroids: result,
            extractorRevision: extractorIdentity.revision
        )

        return CentroidSet(
            centroids: result,
            extractorIdentity: extractorIdentity,
            featureDim: dim,
            metrics: metrics,
            createdAt: Date()
        )
    }

    private static func computeMetrics(features: [Float],
                                      dim: Int,
                                      labels: [String],
                                      centroids: [Centroid],
                                      extractorRevision: Int) -> TrainingMetrics {
        let classes = centroids.map(\.label)
        let K = classes.count
        var labelToIdx: [String: Int] = [:]
        for (i, lbl) in classes.enumerated() { labelToIdx[lbl] = i }

        var confusion = Array(repeating: Array(repeating: 0, count: K), count: K)
        var correct = 0
        for i in 0..<labels.count {
            guard let trueIdx = labelToIdx[labels[i]] else { continue }
            // Find nearest centroid in feature space.
            var bestK = 0
            var bestD: Float = .infinity
            let base = i * dim
            for k in 0..<K {
                var d: Float = 0
                let cv = centroids[k].featureCentroid
                for r in 0..<dim {
                    let v = features[base + r] - cv[r]
                    d += v * v
                }
                if d < bestD { bestD = d; bestK = k }
            }
            confusion[trueIdx][bestK] += 1
            if bestK == trueIdx { correct += 1 }
        }
        let acc = labels.isEmpty ? 0 : Float(correct) / Float(labels.count)

        var precision: [String: Float] = [:]
        var recall: [String: Float] = [:]
        var f1: [String: Float] = [:]
        for i in 0..<K {
            let tp = Float(confusion[i][i])
            var fp: Float = 0, fn: Float = 0
            for j in 0..<K where j != i {
                fp += Float(confusion[j][i])
                fn += Float(confusion[i][j])
            }
            let p = (tp + fp) > 0 ? tp / (tp + fp) : 0
            let r = (tp + fn) > 0 ? tp / (tp + fn) : 0
            precision[classes[i]] = p
            recall[classes[i]] = r
            f1[classes[i]] = (p + r) > 0 ? 2 * p * r / (p + r) : 0
        }

        return TrainingMetrics(
            trainAccuracy: acc,
            valAccuracy: acc,
            perClassPrecision: precision,
            perClassRecall: recall,
            perClassF1: f1,
            confusionMatrix: confusion,
            classLabels: classes,
            trainCount: labels.count,
            valCount: 0,
            finalLoss: 0,
            trainedAt: Date(),
            featureExtractorRevision: extractorRevision,
            iterations: 0
        )
    }
}

extension CentroidSet {
    /// Convert this centroid set into a softmax `MLClassifier` whose argmax
    /// returns the nearest centroid by feature-space squared Euclidean
    /// distance. Equivalence: argmin_k ||x − c_k||² ≡ argmax_k (W_k·x + b_k)
    /// with W_k = 2·c_k/T and b_k = −‖c_k‖²/T. The temperature `T` scales
    /// softmax confidences; larger T → softer probabilities. Default scales
    /// by feature dim so logits stay in a reasonable range for typical
    /// foundation-model embeddings.
    func toMLClassifier(temperature: Float? = nil,
                       embedding: EmbeddingStore.Embedding? = nil) -> MLClassifier {
        let T = temperature ?? max(1, Float(featureDim))
        let K = centroids.count
        let D = featureDim
        var W = [Float](repeating: 0, count: D * K)
        var b = [Float](repeating: 0, count: K)
        for (k, c) in centroids.enumerated() {
            var sqNorm: Float = 0
            for d in 0..<D {
                let v = c.featureCentroid[d]
                W[d * K + k] = (2 * v) / T
                sqNorm += v * v
            }
            b[k] = -sqNorm / T
        }
        let snapshot = embedding.map {
            EmbeddingSnapshot(from: $0, centroids: self, temperature: T)
        }
        return MLClassifier(
            weights: W,
            biases: b,
            classLabels: centroids.map(\.label),
            featureDim: D,
            metrics: metrics,
            featureExtractorRevision: extractorIdentity.revision,
            extractorIdentity: extractorIdentity.stringForm,
            kind: MLClassifier.Kind.centroid.rawValue,
            embeddingSnapshot: snapshot
        )
    }
}
