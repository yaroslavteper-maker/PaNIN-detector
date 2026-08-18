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
    /// Feature-aggregation scheme this classifier was trained on. `nil` = the
    /// legacy per-patch model (one decision per tile). A non-nil value (e.g.
    /// `FeaturePooling.meanMaxStdID`) means the classifier expects a single
    /// pooled descriptor per annotation, and `featureDim` is the *pooled*
    /// dimension (e.g. 3× the raw extractor dimension).
    let aggregation: String?
    /// Per-dimension feature standardization (z-scoring) captured at train
    /// time. `nil` for models trained without standardization (all legacy
    /// per-patch models). When present, `predict` applies `(x - mean) / std`
    /// before the softmax — essential for pooled models, whose concatenated
    /// mean/max/std blocks have very different magnitudes.
    let featureMean: [Float]?
    let featureStd: [Float]?
    /// What the feature vector represents. `nil` = image-embedding features from
    /// an extractor (the default patch/pooled models). `"geometry"` = handcrafted
    /// architectural descriptors (`PathologyGeometry`), which are computed from
    /// the annotation region at predict time instead of the patch bank.
    let featureSource: String?
    /// Reference feature vectors of the "null / exclude" classes (e.g. PaNIN
    /// lumens), captured at train time (raw, pre-standardization). Used to
    /// nullify candidate patches that resemble them. `nil`/empty = no null
    /// filtering. Kept small (capped/subsampled) so it serializes cheaply.
    let nullReference: [[Float]]?
    /// Cosine-similarity cutoff: a patch whose nearest-null similarity is
    /// `>= nullThreshold` is treated as null and excluded. `nil` = disabled.
    let nullThreshold: Float?
    let createdAt: Date

    var classCount: Int { classLabels.count }

    /// True when a candidate patch's raw features look like a null/lumen patch
    /// and should be excluded from training/prediction.
    func isNullLike(_ features: [Float]) -> Bool {
        guard let refs = nullReference, let t = nullThreshold, !refs.isEmpty else { return false }
        for r in refs where r.count == features.count {
            if MLClassifier.cosine(features, r) >= t { return true }
        }
        return false
    }

    /// Cosine similarity of two equal-length vectors (0 if either is degenerate).
    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<a.count { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        let d = (na.squareRoot()) * (nb.squareRoot())
        return d > 0 ? dot / d : 0
    }

    /// True when the classifier makes one decision for a whole annotation from
    /// a pooled bag of patch features, rather than one decision per patch.
    var isPooled: Bool { aggregation != nil }

    /// True for a handcrafted-geometry classifier.
    var isGeometry: Bool { featureSource == "geometry" }

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
        aggregation: String? = nil,
        featureMean: [Float]? = nil,
        featureStd: [Float]? = nil,
        featureSource: String? = nil,
        nullReference: [[Float]]? = nil,
        nullThreshold: Float? = nil,
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
        self.aggregation = aggregation
        self.featureMean = featureMean
        self.featureStd = featureStd
        self.featureSource = featureSource
        self.nullReference = nullReference
        self.nullThreshold = nullThreshold
        self.createdAt = createdAt
    }

    func predict(_ features: [Float]) -> (label: String, probs: [String: Float]) {
        let (idx, probs) = LogisticRegression.predict(
            features: standardized(features),
            weights: weights, biases: biases,
            classCount: classLabels.count
        )
        var map: [String: Float] = [:]
        for (i, label) in classLabels.enumerated() {
            map[label] = probs[i]
        }
        return (classLabels[idx], map)
    }

    /// Apply the stored z-scoring, if any. A no-op for models without stored
    /// standardization params or on a dimension mismatch.
    private func standardized(_ features: [Float]) -> [Float] {
        guard let m = featureMean, let s = featureStd,
              m.count == features.count, s.count == features.count else {
            return features
        }
        var out = features
        for i in 0..<out.count { out[i] = (features[i] - m[i]) / s[i] }
        return out
    }
}
