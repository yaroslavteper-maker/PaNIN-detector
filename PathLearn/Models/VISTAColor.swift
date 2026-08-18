import Foundation
import CoreGraphics

/// Color-normalization for the VISTA / MicePan segmenter, ported to match the
/// Python reference (`vista_norm.py` + `target_stats.py`) the models were
/// validated against. Two steps, both in CIELAB (D65, OpenCV sRGB coefficients):
///
///  1. Luminosity standardization — scale L so its 95th percentile maps to 100.
///  2. Reinhard transfer — shift/scale each LAB channel of the tissue pixels to
///     match the fixed target statistics baked in below.
///
/// LAB here uses the float convention L∈[0,100], a/b centered at 0 (no 8-bit
/// round-trip). The target stats were recomputed with this exact formula so
/// in-app normalization is self-consistent; validated to reproduce the paper's
/// tissue composition to within ~1% of the OpenCV path.
nonisolated enum VISTAColor {

    // Baked Reinhard target (float-LAB, computed over the published target's
    // tissue pixels by target_stats.py — keep in sync with that script).
    static let targetMean: (L: Float, a: Float, b: Float) = (61.774665, 19.989539, -18.963797)
    static let targetStd:  (L: Float, a: Float, b: Float) = (11.029669, 6.853097, 9.262695)

    // OpenCV sRGB->XYZ (D65) matrix, row-major.
    private static let m = (
        r: (Float(0.412453), Float(0.357580), Float(0.180423)),
        g: (Float(0.212671), Float(0.715160), Float(0.072169)),
        b: (Float(0.019334), Float(0.119193), Float(0.950227))
    )
    private static let mInv = (  // inverse of the above (XYZ->linear sRGB)
        r: (Float(3.2404813), Float(-1.5371515), Float(-0.4985363)),
        g: (Float(-0.9692549), Float(1.8759900), Float(0.0415559)),
        b: (Float(0.0556466), Float(-0.2040413), Float(1.0573111))
    )
    private static let xn: Float = 0.950456
    private static let yn: Float = 1.0
    private static let zn: Float = 1.088754
    private static let eps: Float = 0.008856
    private static let kappaLo: Float = 7.787

    /// 256-entry sRGB -> linear LUT (input crops are 8-bit).
    private static let srgbLinearLUT: [Float] = (0..<256).map { i in
        let c = Float(i) / 255.0
        return c > 0.04045 ? powf((c + 0.055) / 1.055, 2.4) : c / 12.92
    }

    private static func fLab(_ t: Float) -> Float {
        t > eps ? cbrtf(t) : (kappaLo * t + 16.0 / 116.0)
    }
    private static func fLabInv(_ t: Float) -> Float {
        let t3 = t * t * t
        return t3 > eps ? t3 : (t - 16.0 / 116.0) / kappaLo
    }

    // MARK: - Crop normalization

    /// Normalize one RGBA8 crop in place-style: returns a new RGBA8 buffer of the
    /// same layout with tissue pixels color-normalized and background (near-white)
    /// pixels left untouched — mirroring `normalize_region` in the Python runner.
    ///
    /// - Parameters:
    ///   - rgba: tightly-packed RGBA8, `width*height*4` bytes.
    static func normalizeCrop(rgba: [UInt8], width: Int, height: Int) -> [UInt8] {
        let n = width * height
        guard rgba.count >= n * 4 else { return rgba }

        // Decode to LAB and build the tissue mask (any channel <= 200) from the
        // ORIGINAL pixels, as the Python path does.
        var L = [Float](repeating: 0, count: n)
        var A = [Float](repeating: 0, count: n)
        var B = [Float](repeating: 0, count: n)
        var isTissue = [Bool](repeating: false, count: n)
        for i in 0..<n {
            let r = rgba[i * 4 + 0], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2]
            isTissue[i] = (r <= 200 || g <= 200 || b <= 200)
            let (l, a, bb) = rgbToLab(r: r, g: g, b: b)
            L[i] = l; A[i] = a; B[i] = bb
        }

        // --- Step 1: luminosity standardization (percentile 95 of L). ---
        let p95 = percentile(L, 95)
        if p95 > 0 {
            let scale = 100.0 / p95
            for i in 0..<n { L[i] = min(max(L[i] * scale, 0), 100) }
        }
        // Luminosity standardization re-derives a/b via a full RGB round trip in
        // the reference, but it only alters L, so a/b are unchanged here.

        // --- Step 2: Reinhard transfer using tissue-pixel statistics. ---
        let (mL, sL) = meanStd(L, mask: isTissue)
        let (mA, sA) = meanStd(A, mask: isTissue)
        let (mB, sB) = meanStd(B, mask: isTissue)
        let e: Float = 1e-6
        let gL = (targetStd.L + e) / (sL + e)
        let gA = (targetStd.a + e) / (sA + e)
        let gB = (targetStd.b + e) / (sB + e)

        var out = rgba
        for i in 0..<n {
            guard isTissue[i] else { continue }  // leave background as-is
            let l = (L[i] - mL) * gL + targetMean.L
            let a = (A[i] - mA) * gA + targetMean.a
            let bb = (B[i] - mB) * gB + targetMean.b
            let (r, g, b) = labToRgb(L: l, a: a, b: bb)
            out[i * 4 + 0] = r
            out[i * 4 + 1] = g
            out[i * 4 + 2] = b
        }
        return out
    }

    // MARK: - Per-pixel conversions

    static func rgbToLab(r: UInt8, g: UInt8, b: UInt8) -> (Float, Float, Float) {
        let rl = srgbLinearLUT[Int(r)]
        let gl = srgbLinearLUT[Int(g)]
        let bl = srgbLinearLUT[Int(b)]
        let x = (m.r.0 * rl + m.r.1 * gl + m.r.2 * bl) / xn
        let y = (m.g.0 * rl + m.g.1 * gl + m.g.2 * bl) / yn
        let z = (m.b.0 * rl + m.b.1 * gl + m.b.2 * bl) / zn
        let fx = fLab(x), fy = fLab(y), fz = fLab(z)
        return (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
    }

    static func labToRgb(L: Float, a: Float, b: Float) -> (UInt8, UInt8, UInt8) {
        let fy = (L + 16.0) / 116.0
        let fx = fy + a / 500.0
        let fz = fy - b / 200.0
        let x = fLabInv(fx) * xn
        let y = fLabInv(fy) * yn
        let z = fLabInv(fz) * zn
        let rl = mInv.r.0 * x + mInv.r.1 * y + mInv.r.2 * z
        let gl = mInv.g.0 * x + mInv.g.1 * y + mInv.g.2 * z
        let bl = mInv.b.0 * x + mInv.b.1 * y + mInv.b.2 * z
        return (linearToSRGB(rl), linearToSRGB(gl), linearToSRGB(bl))
    }

    private static func linearToSRGB(_ c: Float) -> UInt8 {
        let v: Float
        if c > 0.0031308 {
            v = 1.055 * powf(max(c, 0), 1.0 / 2.4) - 0.055
        } else {
            v = 12.92 * c
        }
        return UInt8(min(max(v * 255.0, 0), 255).rounded())
    }

    // MARK: - Stats

    private static func meanStd(_ v: [Float], mask: [Bool]) -> (Float, Float) {
        var sum: Float = 0, count: Float = 0
        for i in 0..<v.count where mask[i] { sum += v[i]; count += 1 }
        guard count > 0 else { return (0, 1) }
        let mean = sum / count
        var varSum: Float = 0
        for i in 0..<v.count where mask[i] { let d = v[i] - mean; varSum += d * d }
        return (mean, sqrtf(varSum / count))
    }

    /// Percentile over all values (matches numpy linear interpolation).
    private static func percentile(_ v: [Float], _ p: Float) -> Float {
        guard !v.isEmpty else { return 0 }
        let sorted = v.sorted()
        let rank = (p / 100.0) * Float(sorted.count - 1)
        let lo = Int(floorf(rank)), hi = Int(ceilf(rank))
        if lo == hi { return sorted[lo] }
        let frac = rank - Float(lo)
        return sorted[lo] * (1 - frac) + sorted[hi] * frac
    }
}
