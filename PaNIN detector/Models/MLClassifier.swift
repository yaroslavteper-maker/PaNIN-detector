import Foundation

/// A trained softmax classifier: weights + biases + the ordered class labels.
struct MLClassifier: Sendable {
    let id: UUID
    let weights: [Float]    // D * K, row-major
    let biases: [Float]     // K
    let classLabels: [String]
    let featureDim: Int
    let metrics: TrainingMetrics
    let featureExtractorRevision: Int
    /// Stable identity of the extractor whose features this classifier was
    /// trained on. Optional so models written before this field migrate
    /// cleanly — a `nil` value is treated as a legacy Vision identity at
    /// `featureExtractorRevision`.
    let extractorIdentity: String?
    /// Provenance: `"logistic"` (gradient-trained) or `"centroid"` (derived
    /// from class means via `CentroidSet.toMLClassifier`). Optional for
    /// backward compat — `nil` is treated as `.logistic`.
    let kind: String?
    /// For centroid classifiers: the t-SNE embedding that produced these
    /// centroids, captured at promote-time so the plot survives save/load.
    /// `nil` for logistic classifiers and for legacy centroid files.
    let embeddingSnapshot: EmbeddingSnapshot?
    let createdAt: Date

    var classCount: Int { classLabels.count }

    enum Kind: String, Codable, Sendable, CaseIterable, Identifiable {
        case logistic
        case centroid
        var id: String { rawValue }
        var displayName: String {
            switch self {
            case .logistic: return "Logistic"
            case .centroid: return "Centroid"
            }
        }
    }

    /// Resolved kind. Legacy classifiers (saved before `kind` existed) are
    /// treated as logistic.
    var classifierKind: Kind {
        if let k = kind, let parsed = Kind(rawValue: k) { return parsed }
        return .logistic
    }

    /// Resolved identity. Falls back to a legacy Vision identity for older
    /// models that predate the `extractorIdentity` field.
    var resolvedExtractorIdentity: ExtractorIdentity {
        if let s = extractorIdentity, let id = ExtractorIdentity.parse(s) {
            return id
        }
        return .legacyVision(revision: featureExtractorRevision)
    }

    init(
        id: UUID = UUID(),
        weights: [Float],
        biases: [Float],
        classLabels: [String],
        featureDim: Int,
        metrics: TrainingMetrics,
        featureExtractorRevision: Int,
        extractorIdentity: String? = nil,
        kind: String? = nil,
        embeddingSnapshot: EmbeddingSnapshot? = nil,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.weights = weights
        self.biases = biases
        self.classLabels = classLabels
        self.featureDim = featureDim
        self.metrics = metrics
        self.featureExtractorRevision = featureExtractorRevision
        self.extractorIdentity = extractorIdentity
        self.kind = kind
        self.embeddingSnapshot = embeddingSnapshot
        self.createdAt = createdAt
    }

    func predict(_ features: [Float]) -> (label: String, probs: [String: Float]) {
        let (idx, probs) = LogisticRegression.predict(
            features: features,
            weights: weights, biases: biases,
            classCount: classLabels.count
        )
        var map: [String: Float] = [:]
        for (i, label) in classLabels.enumerated() {
            map[label] = probs[i]
        }
        return (classLabels[idx], map)
    }
}
