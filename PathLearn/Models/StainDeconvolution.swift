import Foundation
import CoreGraphics

/// Ruifrok–Johnston color deconvolution for H&E. Separates an RGB image into
/// per-pixel hematoxylin (nuclei) and eosin (cytoplasm/stroma) concentrations
/// in optical-density space. Pure CPU, no external models.
///
/// This is the classical backend behind the geometry descriptors; a learned
/// stain normalizer / segmenter can replace it without changing callers.
nonisolated enum StainDeconvolution {

    /// Decoded stain concentrations for a region, one scalar per pixel.
    struct Result: Sendable {
        let width: Int
        let height: Int
        /// Hematoxylin concentration per pixel (row-major). Higher = more nuclei.
        let hematoxylin: [Float]
        /// Eosin concentration per pixel (row-major). Higher = more cytoplasm.
        let eosin: [Float]
        /// True where the pixel is near-white background (not tissue).
        let isBackground: [Bool]
    }

    // Standard normalized H&E stain vectors in RGB optical-density space
    // (Ruifrok & Johnston 2001). Rows: hematoxylin, eosin, residual.
    private static let stainMatrix: [[Double]] = normalizedRows([
        [0.650, 0.704, 0.286],   // hematoxylin
        [0.072, 0.990, 0.105],   // eosin
        [0.268, 0.570, 0.776],   // residual (DAB-ish) — keeps the matrix invertible
    ])

    /// Inverse of `stainMatrix`, computed once. Maps an OD triple → stain
    /// concentrations.
    private static let inverseStain: [[Double]] = invert3x3(stainMatrix)

    /// Deconvolve a BGRA (premultipliedFirst, byteOrder32Little) region — the
    /// format `SlideImage.readRegion` returns.
    static func deconvolve(_ image: CGImage, backgroundThreshold: UInt8 = 220) -> Result? {
        let w = image.width, h = image.height
        guard w > 0, h > 0,
              let data = image.dataProvider?.data,
              let bytes = CFDataGetBytePtr(data) else { return nil }
        let bpp = image.bitsPerPixel / 8
        let bpr = image.bytesPerRow
        guard bpp >= 3, bpr > 0 else { return nil }

        var hema = [Float](repeating: 0, count: w * h)
        var eos = [Float](repeating: 0, count: w * h)
        var bg = [Bool](repeating: false, count: w * h)

        let inv = inverseStain
        for y in 0..<h {
            let row = y * bpr
            for x in 0..<w {
                let off = row + x * bpp
                // BGRA in memory.
                let b = bytes[off]
                let g = bytes[off + 1]
                let r = bytes[off + 2]
                let idx = y * w + x

                if r >= backgroundThreshold, g >= backgroundThreshold, b >= backgroundThreshold {
                    bg[idx] = true
                }

                // Optical density per channel: OD = -log10((I+1)/256).
                let odR = -log10((Double(r) + 1.0) / 256.0)
                let odG = -log10((Double(g) + 1.0) / 256.0)
                let odB = -log10((Double(b) + 1.0) / 256.0)

                // Concentration = inverse · OD (first two stains only).
                let cH = inv[0][0] * odR + inv[0][1] * odG + inv[0][2] * odB
                let cE = inv[1][0] * odR + inv[1][1] * odG + inv[1][2] * odB
                hema[idx] = Float(max(0, cH))
                eos[idx] = Float(max(0, cE))
            }
        }

        return Result(width: w, height: h, hematoxylin: hema, eosin: eos, isBackground: bg)
    }

    // MARK: - Small linear algebra

    private static func normalizedRows(_ m: [[Double]]) -> [[Double]] {
        m.map { row in
            let n = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).squareRoot()
            return n > 0 ? row.map { $0 / n } : row
        }
    }

    private static func invert3x3(_ m: [[Double]]) -> [[Double]] {
        let a = m[0][0], b = m[0][1], c = m[0][2]
        let d = m[1][0], e = m[1][1], f = m[1][2]
        let g = m[2][0], h = m[2][1], i = m[2][2]
        let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        guard abs(det) > 1e-12 else {
            // Fall back to identity — deconvolution degrades gracefully.
            return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        }
        let invDet = 1.0 / det
        return [
            [ (e * i - f * h) * invDet, (c * h - b * i) * invDet, (b * f - c * e) * invDet ],
            [ (f * g - d * i) * invDet, (a * i - c * g) * invDet, (c * d - a * f) * invDet ],
            [ (d * h - e * g) * invDet, (b * g - a * h) * invDet, (a * e - b * d) * invDet ],
        ]
    }
}
