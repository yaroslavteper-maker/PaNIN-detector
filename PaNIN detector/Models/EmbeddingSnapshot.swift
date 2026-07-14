import Foundation

/// Self-contained Codable record of a single t-SNE run, captured at the
/// moment its centroids are promoted to a classifier. When embedded in a
/// saved `.cl` file via `MLClassifier.embeddingSnapshot`, loading that file
/// restores the original scatter, class colours, and centroid markers so
/// the plot window can be reopened and looks identical to the live run.
struct EmbeddingSnapshot: Codable, Sendable {
    let pointsX: [Float]
    let pointsY: [Float]
    let labels: [String]
    let perplexity: Double
    let iterations: Int
    let totalAvailable: Int
    let extractorIdentity: String
    /// Softmax temperature used to derive the classifier weights from the
    /// centroids. Stored so reconstruction can recover the feature-space
    /// centroids from `W` if needed: c_k[d] = W[d*K + k] · T / 2.
    let temperature: Float
    /// label → [x, y] of that class's plot-space centroid.
    let centroidPositions: [String: [Float]]
    let centroidCounts: [String: Int]

    init(from embedding: EmbeddingStore.Embedding,
         centroids: CentroidSet,
         temperature: Float) {
        var xs = [Float](); xs.reserveCapacity(embedding.points.count)
        var ys = [Float](); ys.reserveCapacity(embedding.points.count)
        for p in embedding.points {
            xs.append(p.x)
            ys.append(p.y)
        }
        self.pointsX = xs
        self.pointsY = ys
        self.labels = embedding.labels
        self.perplexity = embedding.perplexity
        self.iterations = embedding.iterations
        self.totalAvailable = embedding.totalAvailable
        self.extractorIdentity = embedding.extractorIdentity.stringForm
        self.temperature = temperature
        var positions: [String: [Float]] = [:]
        var counts: [String: Int] = [:]
        for c in centroids.centroids {
            positions[c.label] = [c.plotPosition.x, c.plotPosition.y]
            counts[c.label] = c.count
        }
        self.centroidPositions = positions
        self.centroidCounts = counts
    }

    /// Rebuild the live `EmbeddingStore.Embedding` from this snapshot.
    func toEmbedding() -> EmbeddingStore.Embedding {
        let n = min(pointsX.count, pointsY.count, labels.count)
        var points: [SIMD2<Float>] = []
        points.reserveCapacity(n)
        for i in 0..<n {
            points.append(SIMD2<Float>(pointsX[i], pointsY[i]))
        }
        let truncatedLabels = Array(labels.prefix(n))
        let identity = ExtractorIdentity.parse(extractorIdentity)
            ?? .legacyVision(revision: 1)
        return EmbeddingStore.Embedding(
            points: points,
            labels: truncatedLabels,
            slideNames: Array(repeating: "", count: n),
            extractorIdentity: identity,
            perplexity: perplexity,
            iterations: iterations,
            totalAvailable: totalAvailable,
            createdAt: Date()
        )
    }

    /// Rebuild the `CentroidSet` from this snapshot + the classifier whose
    /// weights encoded the feature-space centroids. Each centroid's
    /// `featureCentroid` is recovered as W_k · T / 2 — exact inverse of
    /// `CentroidSet.toMLClassifier`.
    func toCentroidSet(using classifier: MLClassifier) -> CentroidSet {
        let D = classifier.featureDim
        let K = classifier.classCount
        let T = temperature
        var centroids: [Centroid] = []
        centroids.reserveCapacity(K)
        for (k, label) in classifier.classLabels.enumerated() {
            // Recover feature-space centroid: c_k[d] = W[d*K + k] · T / 2
            var feature = [Float](repeating: 0, count: D)
            for d in 0..<D {
                feature[d] = classifier.weights[d * K + k] * T / 2
            }
            let xy = centroidPositions[label] ?? [0, 0]
            let plotPos = SIMD2<Float>(xy[0], xy.count > 1 ? xy[1] : 0)
            let count = centroidCounts[label] ?? 0
            centroids.append(Centroid(
                label: label,
                featureCentroid: feature,
                plotPosition: plotPos,
                count: count
            ))
        }
        let identity = ExtractorIdentity.parse(extractorIdentity)
            ?? classifier.resolvedExtractorIdentity
        return CentroidSet(
            centroids: centroids,
            extractorIdentity: identity,
            featureDim: D,
            metrics: classifier.metrics,
            createdAt: classifier.createdAt
        )
    }
}
