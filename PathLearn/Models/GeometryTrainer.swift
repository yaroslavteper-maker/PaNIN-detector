import Foundation

/// Trains a logistic-regression classifier on the compact handcrafted geometry
/// descriptors (one example per annotation). Because the feature vector is tiny
/// (`PathologyGeometry.dimension`), a few dozen annotations per class is enough
/// — this sidesteps the data-starvation that sinks feature pooling. Features are
/// always z-scored (they span densities, µm², fractions), and the params ride on
/// the classifier so prediction standardizes identically.
nonisolated enum GeometryTrainer {

    struct Config: Sendable {
        var iterations: Int = 400
        var learningRate: Float = 0.2
        var l2: Float = 0.01
        var randomSeed: UInt64 = 1
        /// Folds for stratified cross-validation. Clamped down to the smallest
        /// class count so every fold has ≥1 example per class. At small bank
        /// sizes CV gives a far more stable accuracy estimate than one split.
        var kFolds: Int = 5
    }

    enum TrainError: Error, CustomStringConvertible {
        case notEnoughData(Int)
        case notEnoughClasses(Int)
        case classNeedsMoreSamples(String, Int)
        case staleRecords(Int)
        var description: String {
            switch self {
            case .staleRecords(let n):
                return "All \(n) geometry record(s) were computed with an older descriptor version. Clear the geometry bank and re-extract on your slides."
            case .notEnoughData(let n):
                return "Need at least 4 geometry examples to train (have \(n)). Extract geometry on more annotations."
            case .notEnoughClasses(let n):
                return "Need at least 2 classes to train (have \(n))."
            case .classNeedsMoreSamples(let label, let n):
                return "Class “\(label)” has only \(n) annotation(s); need at least 2."
            }
        }
    }

    static func train(records: [GeometryFeatureStore.Record],
                      config: Config = Config(),
                      progress: (@Sendable (Int, Int, String) -> Void)? = nil) throws -> MLClassifier {
        let dim = PathologyGeometry.dimension
        let currentVersion = records.filter { $0.features.count == dim && $0.version == PathologyGeometry.version }
        // If there are records but all are an older descriptor version, say so.
        if currentVersion.isEmpty && !records.isEmpty {
            throw TrainError.staleRecords(records.count)
        }
        let usable = currentVersion
        guard usable.count >= 4 else { throw TrainError.notEnoughData(usable.count) }

        let classes = Array(Set(usable.map(\.classification))).sorted()
        guard classes.count >= 2 else { throw TrainError.notEnoughClasses(classes.count) }
        let classIndex = Dictionary(uniqueKeysWithValues: classes.enumerated().map { ($1, $0) })

        var perClassCount: [String: Int] = [:]
        for r in usable { perClassCount[r.classification, default: 0] += 1 }
        for (label, c) in perClassCount where c < 2 {
            throw TrainError.classNeedsMoreSamples(label, c)
        }

        let K = classes.count
        // Raw (unstandardized) examples — standardization happens per fold for
        // the CV metrics, and globally for the final shipped model.
        let examples = usable.map {
            TrainingExample(features: $0.features, classIndex: classIndex[$0.classification]!)
        }

        // Stratified folds: shuffle each class's indices, deal round-robin.
        let minClassCount = perClassCount.values.min() ?? 0
        let k = max(2, min(config.kFolds, minClassCount))
        var rng = SeededRNG(seed: config.randomSeed == 0 ? 1 : config.randomSeed)
        var byClass: [Int: [Int]] = [:]
        for (i, ex) in examples.enumerated() { byClass[ex.classIndex, default: []].append(i) }
        var folds = Array(repeating: [Int](), count: k)
        for (_, idxs) in byClass {
            var shuffled = idxs
            shuffled.shuffle(using: &rng)
            for (j, idx) in shuffled.enumerated() { folds[j % k].append(idx) }
        }

        // Cross-validation: aggregate a confusion matrix over held-out folds.
        var confusion = Array(repeating: Array(repeating: 0, count: K), count: K)
        var cvCorrect = 0, cvTotal = 0
        for f in 0..<k {
            let valIdx = folds[f]
            let trainIdx = folds.enumerated().filter { $0.offset != f }.flatMap { $0.element }
            guard !valIdx.isEmpty, !trainIdx.isEmpty else { continue }
            let (mean, std) = standardize(examples, indices: trainIdx, dim: dim)
            let (w, b) = fit(examples, indices: trainIdx, mean: mean, std: std,
                             dim: dim, K: K, config: config, progress: nil)
            for idx in valIdx {
                let x = apply(examples[idx].features, mean: mean, std: std)
                let (p, _) = LogisticRegression.predict(features: x, weights: w, biases: b, classCount: K)
                confusion[examples[idx].classIndex][p] += 1
                if p == examples[idx].classIndex { cvCorrect += 1 }
                cvTotal += 1
            }
            progress?(f + 1, k, "Cross-validating fold \(f + 1)/\(k)")
        }
        let cvAccuracy = cvTotal > 0 ? Float(cvCorrect) / Float(cvTotal) : 0

        // Per-class precision/recall/F1 from the aggregated CV confusion.
        var precision: [String: Float] = [:], recall: [String: Float] = [:], f1: [String: Float] = [:]
        for i in 0..<K {
            let tp = Float(confusion[i][i]); var fp: Float = 0; var fn: Float = 0
            for j in 0..<K where j != i { fp += Float(confusion[j][i]); fn += Float(confusion[i][j]) }
            let p = (tp + fp) > 0 ? tp / (tp + fp) : 0
            let r = (tp + fn) > 0 ? tp / (tp + fn) : 0
            precision[classes[i]] = p; recall[classes[i]] = r
            f1[classes[i]] = (p + r) > 0 ? 2 * p * r / (p + r) : 0
        }

        // Final shipped model: standardize + train on ALL data.
        let allIdx = Array(0..<examples.count)
        let (gMean, gStd) = standardize(examples, indices: allIdx, dim: dim)
        progress?(0, config.iterations, "Training final model…")
        let (weights, biases) = fit(examples, indices: allIdx, mean: gMean, std: gStd,
                                    dim: dim, K: K, config: config,
                                    progress: { c, t in progress?(c, t, "Training iter \(c)/\(t)") })
        var trainCorrect = 0
        for idx in allIdx {
            let x = apply(examples[idx].features, mean: gMean, std: gStd)
            let (p, _) = LogisticRegression.predict(features: x, weights: weights, biases: biases, classCount: K)
            if p == examples[idx].classIndex { trainCorrect += 1 }
        }
        let trainAccuracy = Float(trainCorrect) / Float(examples.count)

        let metrics = TrainingMetrics(
            trainAccuracy: trainAccuracy, valAccuracy: cvAccuracy,
            perClassPrecision: precision, perClassRecall: recall, perClassF1: f1,
            confusionMatrix: confusion, classLabels: classes,
            trainCount: examples.count, valCount: cvTotal,
            finalLoss: 0, trainedAt: Date(),
            featureExtractorRevision: 0, iterations: config.iterations
        )

        print("[geometry] \(K) classes, \(usable.count) annotations — \(k)-fold CV accuracy \(Int(cvAccuracy * 100))% (resub \(Int(trainAccuracy * 100))%)")
        return MLClassifier(
            weights: weights, biases: biases,
            classLabels: classes, featureDim: dim, metrics: metrics,
            featureExtractorRevision: 0,
            extractorIdentity: nil,
            kind: MLClassifier.Kind.logistic.rawValue,
            featureMean: gMean, featureStd: gStd,
            featureSource: "geometry"
        )
    }

    // MARK: - Helpers

    /// Per-dimension mean/std over the given example indices (std floored).
    private static func standardize(_ examples: [TrainingExample],
                                    indices: [Int], dim: Int) -> (mean: [Float], std: [Float]) {
        let n = Float(indices.count)
        var mean = [Float](repeating: 0, count: dim)
        for i in indices { let f = examples[i].features; for d in 0..<dim { mean[d] += f[d] } }
        for d in 0..<dim { mean[d] /= n }
        var std = [Float](repeating: 0, count: dim)
        for i in indices { let f = examples[i].features; for d in 0..<dim { let e = f[d] - mean[d]; std[d] += e * e } }
        for d in 0..<dim { std[d] = max((std[d] / n).squareRoot(), 1e-6) }
        return (mean, std)
    }

    private static func apply(_ f: [Float], mean: [Float], std: [Float]) -> [Float] {
        var o = f
        for d in 0..<f.count { o[d] = (f[d] - mean[d]) / std[d] }
        return o
    }

    /// Train logistic-regression weights on `indices`, standardizing with the
    /// supplied mean/std.
    private static func fit(_ examples: [TrainingExample],
                            indices: [Int], mean: [Float], std: [Float],
                            dim: Int, K: Int, config: Config,
                            progress: (@Sendable (Int, Int) -> Void)?) -> (weights: [Float], biases: [Float]) {
        let N = indices.count
        var flatX = [Float](repeating: 0, count: N * dim)
        var labels = [Int](repeating: 0, count: N)
        for (i, idx) in indices.enumerated() {
            let x = apply(examples[idx].features, mean: mean, std: std)
            for d in 0..<dim { flatX[i * dim + d] = x[d] }
            labels[i] = examples[idx].classIndex
        }
        let trained = LogisticRegression.train(
            flatX: flatX, labels: labels, N: N, D: dim, K: K,
            iterations: config.iterations, learningRate: config.learningRate, l2: config.l2,
            progress: progress
        )
        return (trained.weights, trained.biases)
    }
}
