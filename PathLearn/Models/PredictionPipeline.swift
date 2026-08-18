import Foundation
import CoreGraphics

/// Multi-pass patch-by-patch inference. Classes are grouped by their
/// (patch, stride, level) tuple; each unique tuple runs as one pass that
/// samples + extracts + classifies, and we only keep predictions whose
/// predicted label is assigned to that pass. This way "small lesions" can
/// be scanned at one scale and "tissue architecture" at another, without
/// wasted work when several classes share the same scale.
enum PredictionPipeline {

    /// One sampling configuration. Multiple classes can share one config.
    struct PassConfig: Sendable, Hashable {
        let patchSizeLevel: Int
        let strideLevel: Int
        let level: Int32
    }

    /// How per-patch predictions are combined into the displayed result.
    enum Aggregation: String, Sendable, CaseIterable, Identifiable, Codable {
        /// Legacy behavior: every tile keeps its own independent label.
        case perPatch
        /// Late fusion — average the per-patch probability vectors across the
        /// whole annotation into one verdict ("overall character of lesion").
        case meanProbability
        /// Late fusion — take the per-class max probability across all patches,
        /// then argmax ("worst-focus wins" — if any focus strongly reads as a
        /// class, the whole annotation is called that class).
        case maxProbability

        var id: String { rawValue }
        var displayName: String {
            switch self {
            case .perPatch:        return "Per patch"
            case .meanProbability: return "Whole annotation (mean)"
            case .maxProbability:  return "Whole annotation (worst focus)"
            }
        }
        /// True for the annotation-level (late-fusion) modes.
        var isPooled: Bool { self != .perPatch }
    }

    /// Per-class pass assignment plus how results are aggregated.
    struct MultiPassConfig: Sendable {
        let perClass: [String: PassConfig]
        var aggregation: Aggregation = .perPatch
        /// If set, a sampled patch with fewer than this many detected nuclei is
        /// skipped (no classification / no heatmap cell) — the classify-time
        /// analog of the extraction "require visible nuclei" filter.
        var minNuclei: Int? = nil
    }

    struct Report: Sendable {
        let predictions: [PatchPrediction]
        let perClass: [String: Int]
        let passes: [PredictionPassInfo]
        let total: Int
        let skipped: Int
    }

    enum PipelineError: Error, CustomStringConvertible {
        case noPatches
        case featureDimMismatch(expected: Int, got: Int)
        case unsupportedElementType(Int)
        case extractorUnavailable(String, String)
        var description: String {
            switch self {
            case .noPatches: return "No patches sampled — check patch sizes / strides / level."
            case .featureDimMismatch(let exp, let got):
                return "Feature dimension mismatch: classifier expects \(exp), extractor returned \(got). Re-train with the current extractor."
            case .unsupportedElementType(let e):
                return "Unsupported feature element type (\(e))."
            case .extractorUnavailable(let id, let msg):
                return "Required extractor \(id) is not installed: \(msg)"
            }
        }
    }

    nonisolated static func run(
        slide: SlideImage,
        annotation: Annotation,
        classifier: MLClassifier,
        multiPass: MultiPassConfig,
        progress: @Sendable @escaping (Int, Int) -> Void
    ) async throws -> Report {

        // Group classes by their PassConfig.
        var passClasses: [PassConfig: Set<String>] = [:]
        for (label, cfg) in multiPass.perClass {
            passClasses[cfg, default: []].insert(label)
        }
        guard !passClasses.isEmpty else { throw PipelineError.noPatches }

        // Pre-sample each pass so we can show a real "N of total" progress bar.
        struct PassPlan {
            let cfg: PassConfig
            let level: Int32
            let lds: Double
            let patchSizeLevel0: Int
            let origins: [(x: Int64, y: Int64)]
            let allowedClasses: Set<String>
        }
        var plans: [PassPlan] = []
        var totalWork = 0
        for (cfg, classes) in passClasses {
            let levelIdx = max(0, min(Int(cfg.level), Int(slide.levelCount) - 1))
            let lds = slide.levelDownsamples[levelIdx]
            let origins = PatchSampler.samplePatches(
                polygonOverlayY: annotation.points,
                slideDimensions: slide.dimensions,
                patchSizeLevel: cfg.patchSizeLevel,
                strideLevel: cfg.strideLevel,
                levelDownsample: lds
            )
            let patchLevel0 = Int((Double(cfg.patchSizeLevel) * lds).rounded())
            plans.append(PassPlan(
                cfg: cfg,
                level: Int32(levelIdx),
                lds: lds,
                patchSizeLevel0: patchLevel0,
                origins: origins,
                allowedClasses: classes
            ))
            totalWork += origins.count
        }
        guard totalWork > 0 else { throw PipelineError.noPatches }

        // Resolve the extractor the classifier was trained against. Legacy
        // models without a stored identity fall back to a current-revision
        // Vision identity via `resolvedExtractorIdentity`.
        let identity = classifier.resolvedExtractorIdentity
        let extractor: any AnyFeatureExtractor
        do {
            extractor = try await MainActor.run {
                try ExtractorRegistry.shared.extractor(for: identity)
            }
        } catch {
            throw PipelineError.extractorUnavailable(identity.stringForm, String(describing: error))
        }
        print("[prediction] using extractor \(identity.stringForm)")

        // Annotation-level (pooled) classifiers make ONE decision for the whole
        // annotation from a pooled bag of patch features. Sample with a single
        // config (the annotation's own class pass, else the first pass), pool,
        // classify once, then paint every sampled tile with that single verdict
        // so the existing heatmap / panel render it unchanged.
        if classifier.isPooled {
            let plan = plans.first(where: { $0.allowedClasses.contains(annotation.classification) })
                ?? plans[0]
            let rawDim = classifier.featureDim / FeaturePooling.meanMaxStdMultiplier
            var bag: [[Float]] = []
            bag.reserveCapacity(plan.origins.count)
            var skipped = 0
            let total = plan.origins.count
            for (i, origin) in plan.origins.enumerated() {
                try Task.checkCancellation()
                guard let img = slide.readRegion(
                    x: origin.x, y: origin.y, level: plan.level,
                    width: plan.cfg.patchSizeLevel, height: plan.cfg.patchSizeLevel
                ) else {
                    skipped += 1
                    progress(i + 1, total)
                    continue
                }
                if let minN = multiPass.minNuclei,
                   PatchExtractionPipeline.nucleusCount(of: img) < minN {
                    skipped += 1
                    progress(i + 1, total)
                    continue
                }
                let extracted: FeatureExtractionResult
                do {
                    extracted = try await extractor.extract(img)
                } catch {
                    skipped += 1
                    progress(i + 1, total)
                    continue
                }
                let features = try decode(extracted)
                guard features.count == rawDim else {
                    throw PipelineError.featureDimMismatch(expected: rawDim, got: features.count)
                }
                // Keep null/lumen-like patches out of the pooled bag.
                if classifier.isNullLike(features) {
                    skipped += 1
                    progress(i + 1, total)
                    continue
                }
                bag.append(features)
                progress(i + 1, total)
            }

            guard let pooled = FeaturePooling.meanMaxStd(bag) else { throw PipelineError.noPatches }
            guard pooled.count == classifier.featureDim else {
                throw PipelineError.featureDimMismatch(expected: classifier.featureDim, got: pooled.count)
            }
            let (label, probs) = classifier.predict(pooled)
            let maxProb = probs.values.max() ?? 0

            // Paint every sampled tile with the single annotation-level verdict.
            let predictions = plan.origins.map { origin in
                PatchPrediction(
                    dataX: origin.x,
                    dataY: origin.y,
                    sizeLevel0: plan.patchSizeLevel0,
                    predictedLabel: label,
                    probabilities: probs,
                    maxProbability: maxProb,
                    passPatchSize: plan.cfg.patchSizeLevel,
                    passStride: plan.cfg.strideLevel
                )
            }
            let passInfo = [PredictionPassInfo(
                patchSizeLevel: plan.cfg.patchSizeLevel,
                strideLevel: plan.cfg.strideLevel,
                level: plan.level,
                levelDownsample: plan.lds,
                classes: [label]
            )]
            print("[prediction] pooled bag=\(bag.count) → \(label) @ \(maxProb)")
            return Report(
                predictions: predictions,
                perClass: [label: predictions.count],
                passes: passInfo,
                total: predictions.count,
                skipped: skipped
            )
        }

        // Late fusion — classify every tile with the (per-patch) model, then
        // combine the per-patch probability vectors into one annotation verdict
        // (mean or per-class max). Repaint all tiles with that verdict. Uses all
        // training data, so it avoids the feature-pooling data-starvation trap.
        if multiPass.aggregation.isPooled {
            let plan = plans.first(where: { $0.allowedClasses.contains(annotation.classification) })
                ?? plans[0]
            let labels = classifier.classLabels
            var accum = [Float](repeating: multiPass.aggregation == .maxProbability ? 0 : 0, count: labels.count)
            var counted = 0
            var skipped = 0
            let total = plan.origins.count
            for (i, origin) in plan.origins.enumerated() {
                try Task.checkCancellation()
                guard let img = slide.readRegion(
                    x: origin.x, y: origin.y, level: plan.level,
                    width: plan.cfg.patchSizeLevel, height: plan.cfg.patchSizeLevel
                ) else {
                    skipped += 1
                    progress(i + 1, total)
                    continue
                }
                if let minN = multiPass.minNuclei,
                   PatchExtractionPipeline.nucleusCount(of: img) < minN {
                    skipped += 1
                    progress(i + 1, total)
                    continue
                }
                let extracted: FeatureExtractionResult
                do {
                    extracted = try await extractor.extract(img)
                } catch {
                    skipped += 1
                    progress(i + 1, total)
                    continue
                }
                let features = try decode(extracted)
                guard features.count == classifier.featureDim else {
                    throw PipelineError.featureDimMismatch(expected: classifier.featureDim, got: features.count)
                }
                // Skip null/lumen-like patches before accumulating probabilities.
                if classifier.isNullLike(features) {
                    skipped += 1
                    progress(i + 1, total)
                    continue
                }
                let (_, probs) = classifier.predict(features)
                for (k, label) in labels.enumerated() {
                    let p = probs[label] ?? 0
                    if multiPass.aggregation == .maxProbability {
                        accum[k] = max(accum[k], p)
                    } else {
                        accum[k] += p
                    }
                }
                counted += 1
                progress(i + 1, total)
            }

            guard counted > 0 else { throw PipelineError.noPatches }
            // Mean: divide by count. Max: renormalize to a distribution so the
            // displayed probability reads as a fraction.
            let sum: Float
            if multiPass.aggregation == .maxProbability {
                sum = accum.reduce(0, +)
            } else {
                for k in accum.indices { accum[k] /= Float(counted) }
                sum = accum.reduce(0, +)
            }
            if sum > 0 { for k in accum.indices { accum[k] /= sum } }

            var probsMap: [String: Float] = [:]
            for (k, label) in labels.enumerated() { probsMap[label] = accum[k] }
            var bestIdx = 0
            for k in accum.indices where accum[k] > accum[bestIdx] { bestIdx = k }
            let label = labels[bestIdx]
            let maxProb = accum[bestIdx]

            let predictions = plan.origins.map { origin in
                PatchPrediction(
                    dataX: origin.x,
                    dataY: origin.y,
                    sizeLevel0: plan.patchSizeLevel0,
                    predictedLabel: label,
                    probabilities: probsMap,
                    maxProbability: maxProb,
                    passPatchSize: plan.cfg.patchSizeLevel,
                    passStride: plan.cfg.strideLevel
                )
            }
            let passInfo = [PredictionPassInfo(
                patchSizeLevel: plan.cfg.patchSizeLevel,
                strideLevel: plan.cfg.strideLevel,
                level: plan.level,
                levelDownsample: plan.lds,
                classes: [label]
            )]
            print("[prediction] late-fusion \(multiPass.aggregation.rawValue) over \(counted) patches → \(label) @ \(maxProb)")
            return Report(
                predictions: predictions,
                perClass: [label: predictions.count],
                passes: passInfo,
                total: predictions.count,
                skipped: skipped
            )
        }

        var allPredictions: [PatchPrediction] = []
        var perClassCounts: [String: Int] = [:]
        var skipped = 0
        var globalCurrent = 0

        for plan in plans {
            for origin in plan.origins {
                try Task.checkCancellation()

                guard let img = slide.readRegion(
                    x: origin.x, y: origin.y, level: plan.level,
                    width: plan.cfg.patchSizeLevel, height: plan.cfg.patchSizeLevel
                ) else {
                    skipped += 1
                    globalCurrent += 1
                    progress(globalCurrent, totalWork)
                    continue
                }

                if let minN = multiPass.minNuclei,
                   PatchExtractionPipeline.nucleusCount(of: img) < minN {
                    skipped += 1
                    globalCurrent += 1
                    progress(globalCurrent, totalWork)
                    continue
                }

                let extracted: FeatureExtractionResult
                do {
                    extracted = try await extractor.extract(img)
                } catch {
                    skipped += 1
                    globalCurrent += 1
                    progress(globalCurrent, totalWork)
                    continue
                }

                let features: [Float]
                switch extracted.elementType {
                case 1:
                    let count = extracted.data.count / MemoryLayout<Float>.size
                    var out = [Float](repeating: 0, count: count)
                    _ = out.withUnsafeMutableBytes { dst in
                        extracted.data.copyBytes(to: dst)
                    }
                    features = out
                case 2:
                    let count = extracted.data.count / MemoryLayout<Float16>.size
                    var f16 = [Float16](repeating: 0, count: count)
                    _ = f16.withUnsafeMutableBytes { dst in
                        extracted.data.copyBytes(to: dst)
                    }
                    features = f16.map { Float($0) }
                default:
                    throw PipelineError.unsupportedElementType(extracted.elementType)
                }

                guard features.count == classifier.featureDim else {
                    throw PipelineError.featureDimMismatch(expected: classifier.featureDim, got: features.count)
                }

                // Nullify patches that resemble the null/lumen reference.
                if classifier.isNullLike(features) {
                    skipped += 1
                    globalCurrent += 1
                    progress(globalCurrent, totalWork)
                    continue
                }

                let (label, probs) = classifier.predict(features)

                // Only keep this prediction if its predicted label is owned by
                // the current pass.
                if plan.allowedClasses.contains(label) {
                    let maxProb = probs.values.max() ?? 0
                    allPredictions.append(PatchPrediction(
                        dataX: origin.x,
                        dataY: origin.y,
                        sizeLevel0: plan.patchSizeLevel0,
                        predictedLabel: label,
                        probabilities: probs,
                        maxProbability: maxProb,
                        passPatchSize: plan.cfg.patchSizeLevel,
                        passStride: plan.cfg.strideLevel
                    ))
                    perClassCounts[label, default: 0] += 1
                }

                globalCurrent += 1
                progress(globalCurrent, totalWork)
            }
        }

        let passInfo = plans.map { plan in
            PredictionPassInfo(
                patchSizeLevel: plan.cfg.patchSizeLevel,
                strideLevel: plan.cfg.strideLevel,
                level: plan.level,
                levelDownsample: plan.lds,
                classes: Array(plan.allowedClasses).sorted()
            )
        }

        return Report(
            predictions: allPredictions,
            perClass: perClassCounts,
            passes: passInfo,
            total: allPredictions.count,
            skipped: skipped
        )
    }

    /// Decode a raw feature-print blob into `[Float]`, handling the float and
    /// float16 element types Vision / Core ML extractors emit.
    nonisolated private static func decode(_ extracted: FeatureExtractionResult) throws -> [Float] {
        switch extracted.elementType {
        case 1:
            let count = extracted.data.count / MemoryLayout<Float>.size
            var out = [Float](repeating: 0, count: count)
            _ = out.withUnsafeMutableBytes { extracted.data.copyBytes(to: $0) }
            return out
        case 2:
            let count = extracted.data.count / MemoryLayout<Float16>.size
            var f16 = [Float16](repeating: 0, count: count)
            _ = f16.withUnsafeMutableBytes { extracted.data.copyBytes(to: $0) }
            return f16.map { Float($0) }
        default:
            throw PipelineError.unsupportedElementType(extracted.elementType)
        }
    }
}
