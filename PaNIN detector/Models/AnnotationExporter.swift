import Foundation
import ImageIO
import UniformTypeIdentifiers
import CoreGraphics

/// Writes each annotation's bounding-box region (read from OpenSlide at the
/// chosen pyramid level) into:
///   <baseDirectory>/<slideName>/<classification>/<name or id>.<ext>
///
/// Runs nonisolated so the caller can dispatch it off the main actor; the
/// progress callback fires on whatever queue the exporter happens to be on.
enum AnnotationExporter {

    enum Format: Sendable {
        case png
        case jpeg(quality: Double)

        var fileExtension: String {
            switch self {
            case .png:  return "png"
            case .jpeg: return "jpg"
            }
        }

        var typeIdentifier: CFString {
            switch self {
            case .png:  return UTType.png.identifier as CFString
            case .jpeg: return UTType.jpeg.identifier as CFString
            }
        }
    }

    enum CropShape: String, Sendable, CaseIterable, Identifiable {
        /// Tight axis-aligned rectangle around the polygon points (varying aspect).
        case boundingBox
        /// Square centered on the polygon's bounding box, side = max(w, h).
        /// Areas outside the slide are filled with white so the output is
        /// always a perfect square — convenient for ML training where models
        /// expect square inputs.
        case square

        var id: String { rawValue }
        var label: String {
            switch self {
            case .boundingBox: return "Bounding box"
            case .square:      return "Square"
            }
        }
    }

    struct Options: Sendable {
        var format: Format = .png
        var level: Int32 = 0
        /// Pixels (in level-0 slide coords) to expand the bounding box by on
        /// every side. Useful when you want context around the polygon.
        var paddingLevel0: Int = 0
        var cropShape: CropShape = .square
    }

    struct Report: Sendable {
        let savedCount: Int
        let skippedCount: Int
        let outputDirectory: URL
    }

    enum ExportError: Error, CustomStringConvertible {
        case writeFailed(String)
        case noAnnotations
        var description: String {
            switch self {
            case .writeFailed(let s): return "Could not write image: \(s)"
            case .noAnnotations:      return "No annotations to export"
            }
        }
    }

    nonisolated static func export(
        annotations: [Annotation],
        slide: SlideImage,
        slideName: String,
        baseDirectory: URL,
        options: Options,
        progress: (@Sendable (Int, Int) -> Void)? = nil
    ) async throws -> Report {
        guard !annotations.isEmpty else { throw ExportError.noAnnotations }

        let fm = FileManager.default
        let slideDir = baseDirectory.appending(path: sanitize(slideName))
        try fm.createDirectory(at: slideDir, withIntermediateDirectories: true)

        let levelIndex = max(0, min(Int(options.level), Int(slide.levelCount) - 1))
        let lds   = slide.levelDownsamples[levelIndex]
        let level = Int32(levelIndex)

        var savedCount = 0
        var skippedCount = 0
        let total = annotations.count

        let slideRect = CGRect(origin: .zero, size: slide.dimensions)
        let pad = CGFloat(options.paddingLevel0)
        let slideH = slide.dimensions.height

        for (i, ann) in annotations.enumerated() {
            try Task.checkCancellation()

            // Mirror polygon Y to slide-data-Y (see renderPreview for details).
            let dataPoints = ann.points.map {
                CGPoint(x: $0.x, y: slideH - $0.y)
            }
            let rawBBox = boundingBox(of: dataPoints)
            let paddedBBox = rawBBox.insetBy(dx: -pad, dy: -pad)

            // Desired output rectangle in slide-pixel coordinates (Y-down).
            // For `.square`, this may extend past the slide — we fill the
            // missing area with white on the final canvas.
            let outputBBox: CGRect
            switch options.cropShape {
            case .boundingBox:
                outputBBox = paddedBBox.intersection(slideRect)
            case .square:
                let side = max(paddedBBox.width, paddedBBox.height)
                let cx = paddedBBox.midX, cy = paddedBBox.midY
                outputBBox = CGRect(x: cx - side / 2, y: cy - side / 2,
                                    width: side, height: side)
            }

            // Sub-rect that's actually within the slide and readable.
            let readBBox = outputBBox.intersection(slideRect)
            guard outputBBox.width > 0, outputBBox.height > 0,
                  readBBox.width > 0, readBBox.height > 0 else {
                print("[exporter] skip ann \(i+1)/\(total): empty bbox (raw=\(rawBBox))")
                skippedCount += 1
                progress?(i + 1, total)
                continue
            }

            let readX = Int64(floor(readBBox.minX))
            let readY = Int64(floor(readBBox.minY))
            let readW = max(1, Int(ceil(readBBox.width  / lds)))
            let readH = max(1, Int(ceil(readBBox.height / lds)))

            // Final output dimensions in level pixels (the canvas size).
            let canvasW = max(1, Int(ceil(outputBBox.width  / lds)))
            let canvasH = max(1, Int(ceil(outputBBox.height / lds)))

            let polygonArea = ann.areaInSlidePixels
            let bboxArea = Double(rawBBox.width) * Double(rawBBox.height)
            let outputArea = Double(outputBBox.width) * Double(outputBBox.height)
            let readPixelArea = Double(readW) * Double(readH)
            print(String(format:
                "[exporter] ann %d/%d class=%@ pts=%d\n" +
                "    polygon area: %.0f slide-px² (≈%.0f x %.0f if square)\n" +
                "    raw bbox    : (x=%.0f y=%.0f) %.0f x %.0f slide-px  (area=%.0f)\n" +
                "    output bbox : (x=%.0f y=%.0f) %.0f x %.0f slide-px  (area=%.0f, shape=%@)\n" +
                "    read region : %d x %d level-%d px        (lds=%.4f, area=%.0f level-px²)\n" +
                "    canvas      : %d x %d level-px",
                i+1, total, ann.classification, ann.points.count,
                polygonArea, sqrt(polygonArea), sqrt(polygonArea),
                rawBBox.origin.x, rawBBox.origin.y, rawBBox.width, rawBBox.height, bboxArea,
                outputBBox.origin.x, outputBBox.origin.y, outputBBox.width, outputBBox.height, outputArea, String(describing: options.cropShape),
                readW, readH, level, lds, readPixelArea,
                canvasW, canvasH
            ))

            guard let readImage = slide.readRegion(x: readX, y: readY, level: level,
                                                   width: readW, height: readH) else {
                print("[exporter]   readRegion returned nil for ann \(i+1)")
                skippedCount += 1
                progress?(i + 1, total)
                continue
            }
            print("[exporter]   got CGImage \(readImage.width)x\(readImage.height)")

            // Compose onto a canvas if the desired output is larger than what
            // we could read (i.e. the square extended past the slide edge),
            // otherwise use the read image directly.
            let cgImage: CGImage
            if abs(outputBBox.minX - readBBox.minX) < 0.5
                && abs(outputBBox.minY - readBBox.minY) < 0.5
                && abs(outputBBox.width  - readBBox.width)  < 0.5
                && abs(outputBBox.height - readBBox.height) < 0.5 {
                cgImage = readImage
            } else {
                let offsetX = (readBBox.minX - outputBBox.minX) / lds
                let offsetY = (readBBox.minY - outputBBox.minY) / lds
                if let canvas = renderOntoWhiteCanvas(
                    source: readImage,
                    canvasSize: CGSize(width: canvasW, height: canvasH),
                    offset: CGPoint(x: offsetX, y: offsetY)
                ) {
                    print("[exporter]   composed onto \(canvasW)x\(canvasH) canvas at offset (\(offsetX), \(offsetY))")
                    cgImage = canvas
                } else {
                    print("[exporter]   canvas compose failed, using raw read")
                    cgImage = readImage
                }
            }

            let classDir = slideDir.appending(path: sanitize(ann.classification))
            try? fm.createDirectory(at: classDir, withIntermediateDirectories: true)

            let baseName = sanitize(annotationBaseName(ann))
            let outURL   = classDir.appending(path: "\(baseName).\(options.format.fileExtension)")

            do {
                try writeImage(cgImage, to: outURL, format: options.format)
                savedCount += 1
            } catch {
                skippedCount += 1
                print("[exporter] failed to write \(outURL.lastPathComponent): \(error)")
            }

            progress?(i + 1, total)
        }

        return Report(
            savedCount: savedCount,
            skippedCount: skippedCount,
            outputDirectory: slideDir
        )
    }

    struct PreviewResult: Sendable {
        let image: CGImage
        /// The slide-pixel rectangle the image represents (Y-down). Use this
        /// to map polygon points → pixel positions on the preview image:
        ///   px_x = (p.x - outputBBox.minX) * (image.width  / outputBBox.width)
        ///   px_y = (p.y - outputBBox.minY) * (image.height / outputBBox.height)
        let outputBBox: CGRect
    }

    /// Renders a single annotation's crop (same geometry as `export`) into
    /// a CGImage without writing to disk. Auto-picks a pyramid level so the
    /// returned image has its longest side near `targetMaxDim` — keeps the
    /// preview fast even for huge polygons.
    nonisolated static func renderPreview(
        for annotation: Annotation,
        slide: SlideImage,
        cropShape: CropShape = .square,
        paddingLevel0: Int = 0,
        targetMaxDim: Int = 1024
    ) async -> PreviewResult? {
        let slideRect = CGRect(origin: .zero, size: slide.dimensions)
        let pad = CGFloat(paddingLevel0)
        let slideH = slide.dimensions.height

        // Polygon points are stored in overlay Y-down coords. The CATiledLayer
        // renders with a Y-flip that puts slide-data-Y=K at view-Y=(slideH-K)
        // (i.e. the slide is mirrored vertically on screen, which is hard to
        // notice on a pathology slide). To pull the actual visible content,
        // mirror Y on the polygon before computing the read region.
        let dataPoints = annotation.points.map {
            CGPoint(x: $0.x, y: slideH - $0.y)
        }
        let rawBBox = boundingBox(of: dataPoints)
        guard rawBBox.width > 0, rawBBox.height > 0 else { return nil }
        let paddedBBox = rawBBox.insetBy(dx: -pad, dy: -pad)

        let outputBBox: CGRect
        switch cropShape {
        case .boundingBox:
            outputBBox = paddedBBox.intersection(slideRect)
        case .square:
            let side = max(paddedBBox.width, paddedBBox.height)
            let cx = paddedBBox.midX, cy = paddedBBox.midY
            outputBBox = CGRect(x: cx - side / 2, y: cy - side / 2,
                                width: side, height: side)
        }
        let readBBox = outputBBox.intersection(slideRect)
        guard outputBBox.width > 0, outputBBox.height > 0,
              readBBox.width > 0, readBBox.height > 0 else { return nil }

        // Pick a level so the longest side of the output is around targetMaxDim.
        let maxDim = max(outputBBox.width, outputBBox.height)
        let desiredDownsample = max(1.0, Double(maxDim) / Double(targetMaxDim))
        let level = slide.bestLevel(forDownsample: desiredDownsample)
        let lds = slide.levelDownsamples[Int(level)]

        let readX = Int64(floor(readBBox.minX))
        let readY = Int64(floor(readBBox.minY))
        let readW = max(1, Int(ceil(readBBox.width  / lds)))
        let readH = max(1, Int(ceil(readBBox.height / lds)))
        let canvasW = max(1, Int(ceil(outputBBox.width  / lds)))
        let canvasH = max(1, Int(ceil(outputBBox.height / lds)))

        print(String(format:
            "[preview] class=%@ rawBBox=(%.0f,%.0f,%.0f,%.0f) outputBBox=(%.0f,%.0f,%.0f,%.0f) " +
            "level=%d lds=%.4f read=%dx%d canvas=%dx%d shape=%@",
            annotation.classification,
            rawBBox.origin.x, rawBBox.origin.y, rawBBox.width, rawBBox.height,
            outputBBox.origin.x, outputBBox.origin.y, outputBBox.width, outputBBox.height,
            level, lds, readW, readH, canvasW, canvasH,
            String(describing: cropShape)))

        guard let readImage = slide.readRegion(x: readX, y: readY, level: level,
                                               width: readW, height: readH) else {
            return nil
        }

        let resultImage: CGImage
        if abs(outputBBox.minX - readBBox.minX) < 0.5
            && abs(outputBBox.minY - readBBox.minY) < 0.5
            && abs(outputBBox.width  - readBBox.width)  < 0.5
            && abs(outputBBox.height - readBBox.height) < 0.5 {
            resultImage = readImage
        } else {
            let offsetX = (readBBox.minX - outputBBox.minX) / lds
            let offsetY = (readBBox.minY - outputBBox.minY) / lds
            guard let canvas = renderOntoWhiteCanvas(
                source: readImage,
                canvasSize: CGSize(width: canvasW, height: canvasH),
                offset: CGPoint(x: offsetX, y: offsetY)
            ) else { return nil }
            resultImage = canvas
        }
        return PreviewResult(image: resultImage, outputBBox: outputBBox)
    }

    nonisolated private static func annotationBaseName(_ ann: Annotation) -> String {
        if let n = ann.name, !n.trimmingCharacters(in: .whitespaces).isEmpty {
            return n
        }
        return ann.id.uuidString
    }

    nonisolated private static func renderOntoWhiteCanvas(
        source: CGImage,
        canvasSize: CGSize,
        offset: CGPoint     // Y-down, in canvas pixel coords
    ) -> CGImage? {
        let w = Int(canvasSize.width)
        let h = Int(canvasSize.height)
        guard w > 0, h > 0 else { return nil }

        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo: CGBitmapInfo = [
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue),
            .byteOrder32Little
        ]
        guard let ctx = CGContext(
            data: nil, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: w * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else { return nil }

        // White background fills any area outside the slide.
        ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))

        // Flip CTM to Y-down so the offset semantics match slide coords.
        ctx.translateBy(x: 0, y: CGFloat(h))
        ctx.scaleBy(x: 1, y: -1)

        let drawRect = CGRect(
            x: offset.x, y: offset.y,
            width: CGFloat(source.width),
            height: CGFloat(source.height)
        )
        ctx.draw(source, in: drawRect)

        return ctx.makeImage()
    }

    nonisolated private static func boundingBox(of points: [CGPoint]) -> CGRect {
        guard let first = points.first else { return .zero }
        var minX = first.x, maxX = first.x
        var minY = first.y, maxY = first.y
        for p in points.dropFirst() {
            minX = min(minX, p.x); maxX = max(maxX, p.x)
            minY = min(minY, p.y); maxY = max(maxY, p.y)
        }
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    nonisolated private static func writeImage(_ image: CGImage,
                                               to url: URL,
                                               format: Format) throws {
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL,
            format.typeIdentifier,
            1, nil
        ) else {
            throw ExportError.writeFailed("destination creation failed")
        }
        var props: CFDictionary?
        if case .jpeg(let q) = format {
            props = [kCGImageDestinationLossyCompressionQuality: q] as CFDictionary
        }
        CGImageDestinationAddImage(dest, image, props)
        if !CGImageDestinationFinalize(dest) {
            throw ExportError.writeFailed("finalize failed")
        }
    }

    nonisolated private static func sanitize(_ name: String) -> String {
        let invalid: Set<Character> = ["/", "\\", ":", "*", "?", "\"", "<", ">", "|"]
        let cleaned = String(name.map { invalid.contains($0) ? "_" : $0 })
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return cleaned.isEmpty ? "untitled" : cleaned
    }
}
