import Foundation
import CoreGraphics

/// Runs the VISTA / MicePan tissue segmenter over an annotation and produces one
/// `PatchPrediction` per 512×512 tile (dominant tissue type + composition), so
/// the existing heatmap overlay, Model panel, and per-annotation persistence all
/// render it unchanged — VISTA slots in beside `PredictionPipeline`.
///
/// Tiles are read at level 0 (native scan magnification, matching how the paper
/// ran the models on full-resolution scans). To reproduce the validated Python
/// behavior, Reinhard normalization is computed per intermediate crop (not per
/// tile): in-polygon tiles are bucketed into crop-sized regions, each region is
/// read and normalized once, then its tiles are segmented.
nonisolated enum VISTAPipeline {

    static let tile = VISTASegmenter.tileSize   // 512
    static let crop = 2048                        // intermediate normalization crop

    struct Report: Sendable {
        let predictions: [PatchPrediction]
        let perClass: [String: Int]
        let passes: [PredictionPassInfo]
        let total: Int
        let skipped: Int
    }

    enum PipelineError: Error, CustomStringConvertible {
        case noTiles
        var description: String {
            switch self {
            case .noTiles: return "No 512×512 tiles fit inside this annotation at level 0."
            }
        }
    }

    nonisolated static func run(
        slide: SlideImage,
        annotation: Annotation,
        segmenter: VISTASegmenter,
        progress: @Sendable @escaping (Int, Int) -> Void
    ) async throws -> Report {

        // In-polygon tile origins in level-0 data-Y coords (same convention and
        // Y-mirror as the patch pipeline).
        let origins = PatchSampler.samplePatches(
            polygonOverlayY: annotation.points,
            slideDimensions: slide.dimensions,
            patchSizeLevel: tile,
            strideLevel: tile,
            levelDownsample: 1.0
        )
        guard !origins.isEmpty else { throw PipelineError.noTiles }
        let total = origins.count

        // Bucket tiles into CROP-aligned blocks so each normalization region is
        // read once and shared by the tiles inside it.
        var buckets: [BucketKey: [(x: Int64, y: Int64)]] = [:]
        for o in origins {
            let key = BucketKey(bx: Int(o.x) / crop, by: Int(o.y) / crop)
            buckets[key, default: []].append(o)
        }

        let slideW = Int(slide.dimensions.width)
        let slideH = Int(slide.dimensions.height)

        var predictions: [PatchPrediction] = []
        var perClass: [String: Int] = [:]
        var skipped = 0
        var done = 0

        for (_, tiles) in buckets {
            try Task.checkCancellation()

            // Region covering all tiles in this bucket (clamped to the slide).
            var minX = Int.max, minY = Int.max, maxX = Int.min, maxY = Int.min
            for t in tiles {
                minX = min(minX, Int(t.x)); minY = min(minY, Int(t.y))
                maxX = max(maxX, Int(t.x) + tile); maxY = max(maxY, Int(t.y) + tile)
            }
            let rx = max(0, minX), ry = max(0, minY)
            let rw = min(maxX, slideW) - rx
            let rh = min(maxY, slideH) - ry
            guard rw >= tile, rh >= tile,
                  let cg = slide.readRegion(x: Int64(rx), y: Int64(ry), level: 0,
                                            width: rw, height: rh),
                  let regionRGBA = rgba(from: cg, width: rw, height: rh) else {
                skipped += tiles.count
                done += tiles.count
                progress(done, total)
                continue
            }

            let normalized = VISTAColor.normalizeCrop(rgba: regionRGBA, width: rw, height: rh)

            for t in tiles {
                try Task.checkCancellation()
                let ox = Int(t.x) - rx
                let oy = Int(t.y) - ry
                defer { done += 1; progress(done, total) }
                guard ox >= 0, oy >= 0, ox + tile <= rw, oy + tile <= rh,
                      let tileRGBA = subTile(normalized, regionW: rw, ox: ox, oy: oy) else {
                    skipped += 1
                    continue
                }
                do {
                    guard let res = try segmenter.segment(tileRGBA: tileRGBA) else {
                        skipped += 1
                        continue
                    }
                    predictions.append(PatchPrediction(
                        dataX: t.x, dataY: t.y, sizeLevel0: tile,
                        predictedLabel: res.dominant,
                        probabilities: res.fractions,
                        maxProbability: res.maxFraction,
                        passPatchSize: tile, passStride: tile
                    ))
                    perClass[res.dominant, default: 0] += 1
                } catch {
                    skipped += 1
                }
            }
        }

        let passInfo = [PredictionPassInfo(
            patchSizeLevel: tile, strideLevel: tile, level: 0,
            levelDownsample: 1.0, classes: Array(perClass.keys).sorted()
        )]
        return Report(predictions: predictions, perClass: perClass,
                      passes: passInfo, total: predictions.count, skipped: skipped)
    }

    private struct BucketKey: Hashable { let bx: Int; let by: Int }

    // MARK: - Pixel helpers

    /// Draw a CGImage into a tightly-packed RGBA8 buffer (deviceRGB, straight
    /// alpha), regardless of the source's byte order/alpha.
    private static func rgba(from cg: CGImage, width: Int, height: Int) -> [UInt8]? {
        var buf = [UInt8](repeating: 0, count: width * height * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        let ok: Bool = buf.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(
                data: raw.baseAddress,
                width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width * 4,
                space: cs,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return false }
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        return ok ? buf : nil
    }

    /// Copy a `tile`×`tile` RGBA block out of a larger region buffer.
    private static func subTile(_ region: [UInt8], regionW: Int, ox: Int, oy: Int) -> [UInt8]? {
        var out = [UInt8](repeating: 0, count: tile * tile * 4)
        for row in 0..<tile {
            let srcStart = ((oy + row) * regionW + ox) * 4
            let dstStart = row * tile * 4
            let rowBytes = tile * 4
            guard srcStart + rowBytes <= region.count else { return nil }
            out.replaceSubrange(dstStart..<(dstStart + rowBytes),
                                with: region[srcStart..<(srcStart + rowBytes)])
        }
        return out
    }
}
