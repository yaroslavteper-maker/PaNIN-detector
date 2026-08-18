import Foundation
import CoreGraphics

/// Computes a compact vector of handcrafted architectural / cytologic
/// descriptors for a whole annotation — the features that separate PanIN grades
/// (cribriforming, luminal complexity, nuclear crowding, pleomorphism, loss of
/// order) that a small morphology tile can't see.
///
/// Every annotation is analyzed at a **fixed physical resolution** (target
/// µm/px) by tiling it into fixed-size windows and aggregating their stats. This
/// is essential: reading each annotation at "whatever level fits" made absolute
/// descriptors (nuclear area, spacing) scale with annotation size, which
/// confounded larger lesions (e.g. PanIN-3) and collapsed the classifier. Fixed
/// resolution also keeps nuclei resolvable (~several px) instead of sub-pixel.
///
/// Pipeline per window: color deconvolution → nucleus segmentation (swappable
/// backend) + lumen (whitespace) segmentation → pooled shape/topology stats.
nonisolated enum PathologyGeometry {

    /// Ordered descriptor names — indices match the produced vector.
    static let descriptorNames: [String] = [
        "nucleiPerMm2",           // 0  nuclear density
        "nuclearAreaMeanUm2",     // 1  nuclear size
        "nuclearAreaCV",          // 2  pleomorphism
        "nuclearPixelFraction",   // 3  crowding (nucleus px / tissue px)
        "nnDistMeanUm",           // 4  mean nearest-neighbor spacing
        "nnDistCV",               // 5  spacing disorder
        "lumenAreaFraction",      // 6  luminal space fraction
        "lumenCountPerMm2",       // 7  number of lumina (cribriforming proxy)
        "lumenSizeMeanUm2",       // 8  mean lumen size
        "lumenSizeCV",            // 9  lumen size variability
        "lumenCircularityMean",   // 10 low = irregular/cribriform boundaries
        "lumenCircularityCV",     // 11 lumen shape variability
        "eosinToHemaRatio",       // 12 cytoplasm vs nuclei (cellularity)
        "tissueFraction",         // 13 tissue px / inside-polygon px
    ]

    static var dimension: Int { descriptorNames.count }

    /// Bump when the descriptor set / semantics change so stale cached
    /// `GeometryFeature` rows can be detected. v2 = fixed-resolution windowing.
    static let version = 2

    struct Config: Sendable {
        /// Analysis resolution in microns per pixel. Every annotation is read at
        /// the pyramid level closest to this, so descriptors are comparable.
        /// ~1.0 keeps nuclei resolvable while a window still spans whole glands.
        var targetMPP: Double = 1.0
        /// Side length of each analysis window, in pixels at the analysis level.
        var windowPx: Int = 1024
        /// Max windows sampled per annotation (evenly spaced if more fit).
        var maxWindows: Int = 16
        /// Fallback microns-per-pixel at level 0 when the slide lacks
        /// `openslide.mpp-x`.
        var fallbackMPP: Double = 0.25
        /// Minimum lumen blob area (µm²) to count — filters inter-cell gaps.
        var minLumenAreaUm2: Double = 40
        /// Require at least this many nuclei total to emit a descriptor.
        var minNuclei: Int = 20
        /// Cap for the per-window O(n²) nearest-neighbor pass.
        var maxNucleiForNN: Int = 2000
    }

    static func describe(slide: SlideImage,
                         annotation: Annotation,
                         config: Config = Config(),
                         segmenter: NucleusSegmenting = ClassicalNucleusSegmenter()) -> [Float]? {
        guard annotation.points.count >= 3 else { return nil }

        // Mirror overlay-Y → slide-data-Y (same convention as PatchSampler).
        let slideH = slide.dimensions.height
        let dataPoints = annotation.points.map { CGPoint(x: Double($0.x), y: Double(slideH) - Double($0.y)) }

        var minX = dataPoints[0].x, maxX = dataPoints[0].x
        var minY = dataPoints[0].y, maxY = dataPoints[0].y
        for p in dataPoints.dropFirst() {
            minX = min(minX, p.x); maxX = max(maxX, p.x)
            minY = min(minY, p.y); maxY = max(maxY, p.y)
        }
        guard maxX > minX, maxY > minY else { return nil }

        // Fixed analysis level closest to the target resolution.
        let mpp0 = Double(slide.properties["openslide.mpp-x"] ?? "") ?? config.fallbackMPP
        let desiredDownsample = max(1.0, config.targetMPP / mpp0)
        let level = Int(slide.bestLevel(forDownsample: desiredDownsample))
        let lds = slide.levelDownsamples[level]
        let mpp = mpp0 * lds                 // µm per pixel at analysis level
        let pxAreaUm2 = mpp * mpp
        let slideW = Int64(slide.dimensions.width)
        let slideHpx = Int64(slide.dimensions.height)

        // Tile the bounding box into windows (level-0 coords), keeping those
        // whose center is inside the polygon.
        let windowLevel0 = Int64((Double(config.windowPx) * lds).rounded())
        guard windowLevel0 > 0 else { return nil }
        var windows: [(x: Int64, y: Int64)] = []
        var wy = Int64(minY.rounded(.down))
        let yEnd = Int64(maxY.rounded(.up))
        let xEnd = Int64(maxX.rounded(.up))
        while wy < yEnd {
            var wx = Int64(minX.rounded(.down))
            while wx < xEnd {
                let cx = Double(wx) + Double(windowLevel0) / 2
                let cy = Double(wy) + Double(windowLevel0) / 2
                if pointInPolygon(x: cx, y: cy, polygon: dataPoints) {
                    let ox = max(0, min(wx, slideW - windowLevel0))
                    let oy = max(0, min(wy, slideHpx - windowLevel0))
                    windows.append((ox, oy))
                }
                wx += windowLevel0
            }
            wy += windowLevel0
        }
        // Small annotation (no window center inside): analyze one clamped window.
        if windows.isEmpty {
            let ox = max(0, min(Int64(minX.rounded(.down)), slideW - windowLevel0))
            let oy = max(0, min(Int64(minY.rounded(.down)), slideHpx - windowLevel0))
            windows = [(ox, oy)]
        }
        // Cap to maxWindows, evenly spaced.
        if windows.count > config.maxWindows {
            let step = Double(windows.count) / Double(config.maxWindows)
            windows = (0..<config.maxWindows).map { windows[Int((Double($0) * step).rounded(.down))] }
        }

        // Accumulators across windows — all at the same fixed resolution.
        var nucAreasPx: [Int] = []
        var nnDistsUm: [Double] = []
        var nucPixelCount = 0
        var tissueCount = 0
        var insideCount = 0
        var lumenAreasPx: [Int] = []
        var lumenPerims: [Int] = []
        var lumenPixelCount = 0
        var lumenCount = 0
        var hemaSum = 0.0, eosSum = 0.0

        let minLumenPx = max(1, Int((config.minLumenAreaUm2 / pxAreaUm2).rounded()))

        for win in windows {
            guard let img = slide.readRegion(x: win.x, y: win.y, level: Int32(level),
                                             width: config.windowPx, height: config.windowPx),
                  let stain = StainDeconvolution.deconvolve(img) else { continue }
            let w = stain.width, h = stain.height
            let n = w * h
            var tissue = [Bool](repeating: false, count: n)
            var lumenRaw = [Bool](repeating: false, count: n)

            for py in 0..<h {
                for px in 0..<w {
                    let dx = Double(win.x) + (Double(px) + 0.5) * lds
                    let dy = Double(win.y) + (Double(py) + 0.5) * lds
                    guard pointInPolygon(x: dx, y: dy, polygon: dataPoints) else { continue }
                    let idx = py * w + px
                    insideCount += 1
                    if stain.isBackground[idx] {
                        lumenRaw[idx] = true
                    } else {
                        tissue[idx] = true
                        tissueCount += 1
                        hemaSum += Double(stain.hematoxylin[idx])
                        eosSum += Double(stain.eosin[idx])
                    }
                }
            }

            let nuc = segmenter.segment(stain: stain, tissueMask: tissue)
            nucPixelCount += nuc.nucleusPixelCount
            nucAreasPx.append(contentsOf: nuc.nuclei.map { $0.area })
            nnDistsUm.append(contentsOf: nnDistances(nuc.nuclei, mpp: mpp, cap: config.maxNucleiForNN))

            let blobs = ConnectedComponents.label(mask: lumenRaw, width: w, height: h, minArea: minLumenPx)
            lumenCount += blobs.count
            lumenPixelCount += blobs.reduce(0) { $0 + $1.area }
            lumenAreasPx.append(contentsOf: blobs.map { $0.area })
            lumenPerims.append(contentsOf: blobs.map { $0.perimeter })
        }

        guard insideCount > 0, tissueCount > 0, nucAreasPx.count >= config.minNuclei else { return nil }

        let tissueAreaMm2 = Double(tissueCount) * pxAreaUm2 / 1_000_000.0

        let areasUm2 = nucAreasPx.map { Double($0) * pxAreaUm2 }
        let (nucAreaMean, nucAreaStd) = meanStd(areasUm2)
        let nucAreaCV = nucAreaMean > 0 ? nucAreaStd / nucAreaMean : 0
        let nucleiPerMm2 = tissueAreaMm2 > 0 ? Double(nucAreasPx.count) / tissueAreaMm2 : 0
        let nuclearPixelFraction = tissueCount > 0 ? Double(nucPixelCount) / Double(tissueCount) : 0
        let (nnMean, nnStd) = meanStd(nnDistsUm)
        let nnCV = nnMean > 0 ? nnStd / nnMean : 0

        let lumenAreaFraction = insideCount > 0 ? Double(lumenPixelCount) / Double(insideCount) : 0
        let lumenCountPerMm2 = tissueAreaMm2 > 0 ? Double(lumenCount) / tissueAreaMm2 : 0
        let lumenAreasUm2 = lumenAreasPx.map { Double($0) * pxAreaUm2 }
        let (lumenSizeMean, lumenSizeStd) = meanStd(lumenAreasUm2)
        let lumenSizeCV = lumenSizeMean > 0 ? lumenSizeStd / lumenSizeMean : 0

        var circularities: [Double] = []
        circularities.reserveCapacity(lumenAreasPx.count)
        for (a, p) in zip(lumenAreasPx, lumenPerims) {
            let pd = Double(p)
            circularities.append(pd > 0 ? min(1.0, 4.0 * Double.pi * Double(a) / (pd * pd)) : 0)
        }
        let (circMean, circStd) = meanStd(circularities)
        let circCV = circMean > 0 ? circStd / circMean : 0

        let eosinToHema = hemaSum > 0 ? eosSum / hemaSum : 0
        let tissueFraction = insideCount > 0 ? Double(tissueCount) / Double(insideCount) : 0

        let out: [Double] = [
            nucleiPerMm2, nucAreaMean, nucAreaCV, nuclearPixelFraction,
            nnMean, nnCV,
            lumenAreaFraction, lumenCountPerMm2, lumenSizeMean, lumenSizeCV,
            circMean, circCV,
            eosinToHema, tissueFraction,
        ]
        return out.map { Float($0.isFinite ? $0 : 0) }
    }

    // MARK: - Helpers

    private static func meanStd(_ xs: [Double]) -> (mean: Double, std: Double) {
        guard !xs.isEmpty else { return (0, 0) }
        let m = xs.reduce(0, +) / Double(xs.count)
        let v = xs.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(xs.count)
        return (m, v.squareRoot())
    }

    /// Nearest-neighbor centroid distances (µm) for one window's nuclei.
    private static func nnDistances(_ nuclei: [DetectedNucleus],
                                    mpp: Double, cap: Int) -> [Double] {
        guard nuclei.count >= 2 else { return [] }
        let subset = nuclei.count > cap
            ? Array(nuclei.sorted { $0.area > $1.area }.prefix(cap))
            : nuclei
        var dists: [Double] = []
        dists.reserveCapacity(subset.count)
        for i in 0..<subset.count {
            var best = Double.greatestFiniteMagnitude
            let a = subset[i]
            for j in 0..<subset.count where j != i {
                let b = subset[j]
                let dx = a.cx - b.cx, dy = a.cy - b.cy
                let d2 = dx * dx + dy * dy
                if d2 < best { best = d2 }
            }
            if best.isFinite { dists.append(best.squareRoot() * mpp) }
        }
        return dists
    }

    private static func pointInPolygon(x: Double, y: Double, polygon: [CGPoint]) -> Bool {
        var inside = false
        var j = polygon.count - 1
        for i in 0..<polygon.count {
            let pi = polygon[i], pj = polygon[j]
            let yi = Double(pi.y), yj = Double(pj.y)
            let xi = Double(pi.x), xj = Double(pj.x)
            if (yi > y) != (yj > y),
               x < (xj - xi) * (y - yi) / (yj - yi) + xi {
                inside.toggle()
            }
            j = i
        }
        return inside
    }
}
