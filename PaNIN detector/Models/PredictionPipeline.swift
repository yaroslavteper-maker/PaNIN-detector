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

    /// Per-class pass assignment.
    struct MultiPassConfig: Sendable {
        let perClass: [String: PassConfig]
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
}
