import Foundation
import CoreGraphics
import SwiftData
import Vision

/// Coordinates patch sampling → feature extraction → SwiftData inserts.
/// Caller invokes from a `Task.detached`; only the SwiftData writes hop
/// back to the MainActor.
enum PatchExtractionPipeline {

    struct Config: Sendable {
        var patchSizeLevel: Int = 224
        var strideLevel: Int = 224
        var level: Int32 = 0
        /// If set, patches whose pixel-whiteness exceeds this fraction are
        /// skipped during extraction (their features are never computed).
        var maxWhiteFraction: Float? = nil
        /// Which feature extractor to use. `nil` selects the built-in Vision
        /// extractor at the current revision (legacy behaviour).
        var extractorIdentity: ExtractorIdentity? = nil
    }

    struct Report: Sendable {
        let saved: Int
        let skipped: Int
        let perClass: [String: Int]
    }

    enum PipelineError: Error, CustomStringConvertible {
        case noAnnotations
        var description: String {
            switch self {
            case .noAnnotations: return "No annotations selected for extraction."
            }
        }
    }

    private struct ReadyPatch: Sendable {
        let annotationID: UUID
        let classification: String
        let x: Int64
        let y: Int64
        let data: Data
        let dim: Int
        let elementType: Int
        let revision: Int
        let identityString: String
        let whiteFraction: Float
    }

    /// Fraction of pixels in the patch where R, G, B are all ≥ threshold.
    /// Cheap linear scan; only called once per accepted patch.
    nonisolated static func whiteFraction(of cgImage: CGImage,
                                          threshold: UInt8 = 220) -> Float {
        guard let data = cgImage.dataProvider?.data,
              let bytes = CFDataGetBytePtr(data) else { return 0 }
        let w = cgImage.width
        let h = cgImage.height
        let bpp = cgImage.bitsPerPixel / 8
        let bpr = cgImage.bytesPerRow
        guard w > 0, h > 0, bpp >= 3, bpr > 0 else { return 0 }

        var whiteCount = 0
        // OpenSlide → CGImage is BGRA premultipliedFirst byteOrder32Little:
        //   bytes in memory are B G R A per pixel.
        for y in 0..<h {
            let row = y * bpr
            for x in 0..<w {
                let off = row + x * bpp
                let b = bytes[off]
                let g = bytes[off + 1]
                let r = bytes[off + 2]
                if r >= threshold, g >= threshold, b >= threshold {
                    whiteCount += 1
                }
            }
        }
        return Float(whiteCount) / Float(w * h)
    }

    nonisolated static func run(
        slide: SlideImage,
        slidePath: String,
        slideName: String,
        annotations: [Annotation],
        config: Config,
        modelContainer: ModelContainer,
        progress: @Sendable (Int, Int) -> Void
    ) async throws -> Report {
        guard !annotations.isEmpty else { throw PipelineError.noAnnotations }

        let slideDim = slide.dimensions
        let levelIdx = max(0, min(Int(config.level), Int(slide.levelCount) - 1))
        let lds = slide.levelDownsamples[levelIdx]
        let level = Int32(levelIdx)

        // Pre-sample all patch origins per annotation.
        struct PatchTask {
            let annotationID: UUID
            let classification: String
            let origin: (x: Int64, y: Int64)
        }
        var tasks: [PatchTask] = []
        for ann in annotations {
            let origins = PatchSampler.samplePatches(
                polygonOverlayY: ann.points,
                slideDimensions: slideDim,
                patchSizeLevel: config.patchSizeLevel,
                strideLevel: config.strideLevel,
                levelDownsample: lds
            )
            for o in origins {
                tasks.append(PatchTask(
                    annotationID: ann.id,
                    classification: ann.classification,
                    origin: o
                ))
            }
        }
        let total = tasks.count
        guard total > 0 else {
            return Report(saved: 0, skipped: 0, perClass: [:])
        }

        print("[extraction] sampled \(total) patches across \(annotations.count) annotation(s)")

        let identity = config.extractorIdentity
            ?? .legacyVision(revision: VNGenerateImageFeaturePrintRequest.currentRevision)
        let extractor: any AnyFeatureExtractor
        do {
            extractor = try await MainActor.run {
                try ExtractorRegistry.shared.extractor(for: identity)
            }
        } catch {
            print("[extraction] could not resolve extractor \(identity.stringForm): \(error)")
            throw error
        }
        print("[extraction] using extractor \(identity.stringForm)")
        var batch: [ReadyPatch] = []
        var savedCount = 0
        var skippedCount = 0
        var perClass: [String: Int] = [:]
        let batchSize = 50

        for (i, task) in tasks.enumerated() {
            try Task.checkCancellation()

            guard let img = slide.readRegion(
                x: task.origin.x,
                y: task.origin.y,
                level: level,
                width: config.patchSizeLevel,
                height: config.patchSizeLevel
            ) else {
                skippedCount += 1
                progress(i + 1, total)
                continue
            }

            let whiteFrac = whiteFraction(of: img)
            if let cutoff = config.maxWhiteFraction, whiteFrac > cutoff {
                skippedCount += 1
                progress(i + 1, total)
                continue
            }

            let extracted: FeatureExtractionResult
            do {
                extracted = try await extractor.extract(img)
            } catch {
                skippedCount += 1
                progress(i + 1, total)
                continue
            }

            batch.append(ReadyPatch(
                annotationID: task.annotationID,
                classification: task.classification,
                x: task.origin.x,
                y: task.origin.y,
                data: extracted.data,
                dim: extracted.dim,
                elementType: extracted.elementType,
                revision: extracted.identity.revision,
                identityString: extracted.identity.stringForm,
                whiteFraction: whiteFrac
            ))

            if batch.count >= batchSize {
                let toFlush = batch
                batch = []
                try await flush(
                    toFlush,
                    slidePath: slidePath,
                    slideName: slideName,
                    level: level,
                    sizeLevel: config.patchSizeLevel,
                    container: modelContainer
                )
                savedCount += toFlush.count
                for p in toFlush { perClass[p.classification, default: 0] += 1 }
            }

            progress(i + 1, total)
        }

        if !batch.isEmpty {
            let toFlush = batch
            try await flush(
                toFlush,
                slidePath: slidePath,
                slideName: slideName,
                level: level,
                sizeLevel: config.patchSizeLevel,
                container: modelContainer
            )
            savedCount += toFlush.count
            for p in toFlush { perClass[p.classification, default: 0] += 1 }
        }

        print("[extraction] done: saved=\(savedCount) skipped=\(skippedCount)")
        return Report(saved: savedCount, skipped: skippedCount, perClass: perClass)
    }

    @MainActor
    private static func flush(
        _ batch: [ReadyPatch],
        slidePath: String,
        slideName: String,
        level: Int32,
        sizeLevel: Int,
        container: ModelContainer
    ) throws {
        let context = ModelContext(container)
        for p in batch {
            let row = MLPatch(
                slidePath: slidePath,
                slideName: slideName,
                annotationID: p.annotationID,
                classification: p.classification,
                patchX: p.x,
                patchY: p.y,
                patchLevel: level,
                patchSizeLevel: sizeLevel,
                featureData: p.data,
                featureDim: p.dim,
                featureElementType: p.elementType,
                extractorRevision: p.revision,
                extractorIdentity: p.identityString,
                whiteFraction: p.whiteFraction
            )
            context.insert(row)
        }
        try context.save()
    }
}
