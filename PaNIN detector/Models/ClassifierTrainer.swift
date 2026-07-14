import Foundation
import SwiftData

/// Pipeline: fetch all `MLPatch` rows from the bank → decode features →
/// stratified train/val split → train via `LogisticRegression` →
/// evaluate → return an `MLClassifier`.
enum ClassifierTrainer {

    struct Config: Sendable {
        var iterations: Int = 200
        var learningRate: Float = 0.1
        var l2: Float = 0.001
        var valFraction: Float = 0.2
        /// 0 means a fresh random shuffle each run.
        var randomSeed: UInt64 = 0
        /// If set, patches whose stored `whiteFraction` exceeds this are
        /// excluded from training. Legacy patches (whiteFraction = 0) always
        /// pass — re-extract them to assign a real score.
        var maxWhiteFraction: Float? = nil
        /// If set, only patches whose `classification` is in this set are
        /// used for training. `nil` = use every class in the bank.
        var enabledClasses: Set<String>? = nil
        /// When `true` and `enabledClasses` has exactly one entry, all *other*
        /// patches in the bank are kept and relabeled as `"Background"` so the
        /// trainer produces a 2-class one-vs-rest classifier. Solves the
        /// "I only have one class but want to detect it" use case without
        /// requiring softmax to be applied to a single class.
        var binarizeAgainstRest: Bool = false
        /// If set, only patches whose resolved extractor identity string
        /// matches will be used. Lets a mixed bank (e.g. Vision + UNI patches
        /// over the same annotations) train one classifier per extractor
        /// without needing to clear the others.
        var extractorIdentity: String? = nil
    }

    /// Class label used for the synthetic negative class when
    /// `Config.binarizeAgainstRest` is true.
    public static let backgroundClassLabel = "Background"

    struct Loaded: Sendable {
        let examples: [TrainingExample]
        let classLabels: [String]
        let featureDim: Int
        let extractorRevision: Int
        let extractorIdentity: ExtractorIdentity
    }

    enum TrainError: Error, CustomStringConvertible {
        case notEnoughData(Int)
        case notEnoughClasses(Int)
        case mixedFeatureDims
        case unsupportedElementType(Int)
        case mixedExtractorRevisions
        case mixedExtractorIdentities
        case classNeedsMoreSamples(String, Int)

        var description: String {
            switch self {
            case .notEnoughData(let n):
                return "Need at least 4 patches to train (have \(n))."
            case .notEnoughClasses(let n):
                return "Need at least 2 distinct classes to train (have \(n))."
            case .mixedFeatureDims:
                return "Patches have different feature dimensions. Clear the bank and re-extract."
            case .unsupportedElementType(let e):
                return "Unsupported feature element type (\(e)). Clear and re-extract."
            case .mixedExtractorRevisions:
                return "Patches were extracted with different feature-print revisions. Clear and re-extract."
            case .mixedExtractorIdentities:
                return "Patches were extracted with different feature extractors. Filter the bank to a single extractor before training."
            case .classNeedsMoreSamples(let label, let n):
                return "Class “\(label)” has only \(n) patch(es); need at least 2 per class."
            }
        }
    }

    // MARK: Load (MainActor — SwiftData fetch)

    @MainActor
    static func loadExamples(context: ModelContext,
                             maxWhiteFraction: Float? = nil,
                             enabledClasses: Set<String>? = nil,
                             binarizeAgainstRest: Bool = false,
                             extractorIdentity: String? = nil) throws -> Loaded {
        let descriptor = FetchDescriptor<MLPatch>()
        var patches = try context.fetch(descriptor)
        if let cutoff = maxWhiteFraction {
            let before = patches.count
            patches = patches.filter { $0.whiteFraction <= cutoff }
            print("[trainer] whiteness filter ≤\(cutoff): \(before) → \(patches.count) patches")
        }
        if let extractorIdentity {
            let before = patches.count
            patches = patches.filter { $0.resolvedExtractorIdentity.stringForm == extractorIdentity }
            print("[trainer] extractor filter \(extractorIdentity): \(before) → \(patches.count) patches")
        }

        // Decide whether we're running in one-vs-rest mode. Requires the
        // caller to have explicitly enabled exactly one class.
        let useBinarize = binarizeAgainstRest && (enabledClasses?.count == 1)
        let target: String? = useBinarize ? enabledClasses?.first : nil

        // Apply normal class filter (skip when binarizing — we *want* every
        // other patch to land in the synthetic "Background" class).
        if !useBinarize, let enabled = enabledClasses {
            let before = patches.count
            patches = patches.filter { enabled.contains($0.classification) }
            print("[trainer] class filter (\(enabled.count) classes): \(before) → \(patches.count) patches")
        }

        // The effective label each patch contributes to training. Either the
        // patch's stored classification, or the binarize remap.
        func effectiveLabel(_ patch: MLPatch) -> String {
            if let target = target {
                return patch.classification == target ? target : backgroundClassLabel
            }
            return patch.classification
        }

        guard patches.count >= 4 else {
            throw TrainError.notEnoughData(patches.count)
        }

        let classes = Array(Set(patches.map(effectiveLabel))).sorted()
        guard classes.count >= 2 else {
            throw TrainError.notEnoughClasses(classes.count)
        }

        let dim = patches[0].featureDim
        guard patches.allSatisfy({ $0.featureDim == dim }) else {
            throw TrainError.mixedFeatureDims
        }
        let rev = patches[0].extractorRevision
        guard patches.allSatisfy({ $0.extractorRevision == rev }) else {
            throw TrainError.mixedExtractorRevisions
        }
        let resolvedIdentity = patches[0].resolvedExtractorIdentity
        guard patches.allSatisfy({ $0.resolvedExtractorIdentity == resolvedIdentity }) else {
            throw TrainError.mixedExtractorIdentities
        }
        print("[trainer] bank extractor: \(resolvedIdentity.stringForm)")
        let classIndex = Dictionary(uniqueKeysWithValues: classes.enumerated().map { ($1, $0) })

        if useBinarize, let target = target {
            let positiveCount = patches.filter { $0.classification == target }.count
            print("[trainer] binarize vs rest: target=\"\(target)\" (\(positiveCount) pts) vs \"\(backgroundClassLabel)\" (\(patches.count - positiveCount) pts)")
        }

        var examples: [TrainingExample] = []
        examples.reserveCapacity(patches.count)
        for patch in patches {
            let label = effectiveLabel(patch)
            guard let cIdx = classIndex[label] else { continue }
            let features = try decodeFeatures(patch: patch, dim: dim)
            examples.append(TrainingExample(features: features, classIndex: cIdx))
        }

        // Ensure every class has >=2 samples (so stratified split is sane).
        var perClassCount: [Int: Int] = [:]
        for e in examples { perClassCount[e.classIndex, default: 0] += 1 }
        for (i, c) in perClassCount where c < 2 {
            throw TrainError.classNeedsMoreSamples(classes[i], c)
        }

        return Loaded(
            examples: examples,
            classLabels: classes,
            featureDim: dim,
            extractorRevision: rev,
            extractorIdentity: resolvedIdentity
        )
    }

    @MainActor
    private static func decodeFeatures(patch: MLPatch, dim: Int) throws -> [Float] {
        switch patch.featureElementType {
        case 1: // VNElementType.float — 32-bit
            let count = patch.featureData.count / MemoryLayout<Float>.size
            var out = [Float](repeating: 0, count: count)
            _ = out.withUnsafeMutableBytes { dst in
                patch.featureData.copyBytes(to: dst)
            }
            return out
        case 2: // VNElementType.float16 — 16-bit
            let count = patch.featureData.count / MemoryLayout<Float16>.size
            var f16 = [Float16](repeating: 0, count: count)
            _ = f16.withUnsafeMutableBytes { dst in
                patch.featureData.copyBytes(to: dst)
            }
            return f16.map { Float($0) }
        default:
            throw TrainError.unsupportedElementType(patch.featureElementType)
        }
    }

    // MARK: Train (nonisolated — heavy work, off-main)

    nonisolated static func train(
        loaded: Loaded,
        config: Config,
        progress: @Sendable @escaping (Int, Int, String) -> Void
    ) -> MLClassifier {
        let examples = loaded.examples
        let classes = loaded.classLabels
        let D = loaded.featureDim
        let K = classes.count

        // Stratified split
        progress(0, config.iterations, "Splitting train/val…")
        var perClass: [Int: [Int]] = [:]
        for (i, ex) in examples.enumerated() {
            perClass[ex.classIndex, default: []].append(i)
        }
        var rng = SeededRNG(seed: config.randomSeed)
        var trainIdx: [Int] = []
        var valIdx: [Int] = []
        for (_, indices) in perClass {
            var shuffled = indices
            shuffled.shuffle(using: &rng)
            let valCount = max(1, Int(Float(indices.count) * config.valFraction))
            valIdx.append(contentsOf: shuffled.prefix(valCount))
            trainIdx.append(contentsOf: shuffled.dropFirst(valCount))
        }

        // Build flat training matrix
        let Ntrain = trainIdx.count
        var flatX = [Float](repeating: 0, count: Ntrain * D)
        var labels = [Int](repeating: 0, count: Ntrain)
        for (i, idx) in trainIdx.enumerated() {
            let ex = examples[idx]
            for d in 0..<D { flatX[i * D + d] = ex.features[d] }
            labels[i] = ex.classIndex
        }

        progress(0, config.iterations, "Training…")
        let trained = LogisticRegression.train(
            flatX: flatX,
            labels: labels,
            N: Ntrain, D: D, K: K,
            iterations: config.iterations,
            learningRate: config.learningRate,
            l2: config.l2,
            progress: { current, total in
                progress(current, total, "Training iter \(current)/\(total)")
            }
        )

        // Evaluate
        progress(config.iterations, config.iterations, "Evaluating…")
        let trainAcc = accuracy(over: trainIdx, examples: examples, trained: trained)
        let valAcc = accuracy(over: valIdx, examples: examples, trained: trained)

        // Confusion matrix on the val set
        var confusion = Array(repeating: Array(repeating: 0, count: K), count: K)
        for idx in valIdx {
            let ex = examples[idx]
            let (predIdx, _) = LogisticRegression.predict(
                features: ex.features,
                weights: trained.weights, biases: trained.biases,
                classCount: K
            )
            confusion[ex.classIndex][predIdx] += 1
        }

        // Per-class precision / recall / F1 on val
        var precision: [String: Float] = [:]
        var recall: [String: Float] = [:]
        var f1: [String: Float] = [:]
        for i in 0..<K {
            let tp = Float(confusion[i][i])
            var fp: Float = 0
            var fn: Float = 0
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

        let metrics = TrainingMetrics(
            trainAccuracy: trainAcc,
            valAccuracy: valAcc,
            perClassPrecision: precision,
            perClassRecall: recall,
            perClassF1: f1,
            confusionMatrix: confusion,
            classLabels: classes,
            trainCount: trainIdx.count,
            valCount: valIdx.count,
            finalLoss: trained.finalLoss,
            trainedAt: Date(),
            featureExtractorRevision: loaded.extractorRevision,
            iterations: config.iterations
        )

        return MLClassifier(
            weights: trained.weights,
            biases: trained.biases,
            classLabels: classes,
            featureDim: D,
            metrics: metrics,
            featureExtractorRevision: loaded.extractorRevision,
            extractorIdentity: loaded.extractorIdentity.stringForm,
            kind: MLClassifier.Kind.logistic.rawValue
        )
    }

    nonisolated private static func accuracy(
        over indices: [Int],
        examples: [TrainingExample],
        trained: LogisticRegression.Trained
    ) -> Float {
        guard !indices.isEmpty else { return 0 }
        var correct = 0
        for idx in indices {
            let ex = examples[idx]
            let (predIdx, _) = LogisticRegression.predict(
                features: ex.features,
                weights: trained.weights, biases: trained.biases,
                classCount: trained.classCount
            )
            if predIdx == ex.classIndex { correct += 1 }
        }
        return Float(correct) / Float(indices.count)
    }
}

struct TrainingExample: Sendable {
    let features: [Float]
    let classIndex: Int
}

/// Seeded RNG so training runs are reproducible when a seed is supplied.
struct SeededRNG: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) {
        // seed == 0 → roll a fresh seed
        self.state = seed == 0 ? UInt64.random(in: 1...UInt64.max) : seed
    }
    mutating func next() -> UInt64 {
        state &*= 6364136223846793005
        state &+= 1442695040888963407
        return state
    }
}
