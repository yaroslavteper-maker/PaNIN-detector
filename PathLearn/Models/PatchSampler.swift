import Foundation
import CoreGraphics

/// Turns a polygon (in overlay Y-down coords) into a list of patch origins
/// (in level-0 slide-data-Y coords) on a regular grid. Only patches whose
/// center is inside the polygon and whose rect fits in the slide are emitted.
enum PatchSampler {
    nonisolated static func samplePatches(
        polygonOverlayY: [CGPoint],
        slideDimensions: CGSize,
        patchSizeLevel: Int,
        strideLevel: Int,
        levelDownsample: Double
    ) -> [(x: Int64, y: Int64)] {
        guard polygonOverlayY.count >= 3,
              patchSizeLevel > 0, strideLevel > 0,
              levelDownsample > 0 else { return [] }

        // Mirror Y to slide-data-Y (same convention as AnnotationExporter).
        let slideH = slideDimensions.height
        let dataPoints = polygonOverlayY.map { CGPoint(x: $0.x, y: slideH - $0.y) }

        // Bounding box of the data-Y polygon.
        var minX = dataPoints[0].x, maxX = dataPoints[0].x
        var minY = dataPoints[0].y, maxY = dataPoints[0].y
        for p in dataPoints.dropFirst() {
            minX = min(minX, p.x); maxX = max(maxX, p.x)
            minY = min(minY, p.y); maxY = max(maxY, p.y)
        }

        let lds = levelDownsample
        let patchLevel0 = Int64(round(Double(patchSizeLevel) * lds))
        let strideLevel0 = Int64(round(Double(strideLevel) * lds))
        guard patchLevel0 > 0, strideLevel0 > 0 else { return [] }

        let xStart = Int64(floor(minX))
        let yStart = Int64(floor(minY))
        let xEnd = Int64(ceil(maxX)) - patchLevel0
        let yEnd = Int64(ceil(maxY)) - patchLevel0
        let slideW = Int64(slideDimensions.width)
        let slideHpx = Int64(slideDimensions.height)

        var origins: [(x: Int64, y: Int64)] = []
        var y = yStart
        while y <= yEnd {
            var x = xStart
            while x <= xEnd {
                if x >= 0, y >= 0,
                   x + patchLevel0 <= slideW,
                   y + patchLevel0 <= slideHpx {
                    let cx = CGFloat(x) + CGFloat(patchLevel0) / 2
                    let cy = CGFloat(y) + CGFloat(patchLevel0) / 2
                    if pointInPolygon(CGPoint(x: cx, y: cy), polygon: dataPoints) {
                        origins.append((x, y))
                    }
                }
                x += strideLevel0
            }
            y += strideLevel0
        }
        return origins
    }

    /// Cheap upper-bound estimate (bbox / stride²) — for UI preview only.
    nonisolated static func estimatedPatchCount(
        polygonOverlayY: [CGPoint],
        patchSizeLevel: Int,
        strideLevel: Int,
        levelDownsample: Double
    ) -> Int {
        guard polygonOverlayY.count >= 3, strideLevel > 0, levelDownsample > 0 else { return 0 }
        var minX = polygonOverlayY[0].x, maxX = polygonOverlayY[0].x
        var minY = polygonOverlayY[0].y, maxY = polygonOverlayY[0].y
        for p in polygonOverlayY.dropFirst() {
            minX = min(minX, p.x); maxX = max(maxX, p.x)
            minY = min(minY, p.y); maxY = max(maxY, p.y)
        }
        let strideLevel0 = Double(strideLevel) * levelDownsample
        let nx = max(0, Int(floor((maxX - minX) / strideLevel0)))
        let ny = max(0, Int(floor((maxY - minY) / strideLevel0)))
        return nx * ny
    }

    /// Ray-casting point-in-polygon test.
    nonisolated private static func pointInPolygon(_ p: CGPoint, polygon: [CGPoint]) -> Bool {
        var inside = false
        var j = polygon.count - 1
        for i in 0..<polygon.count {
            let pi = polygon[i], pj = polygon[j]
            if (pi.y > p.y) != (pj.y > p.y),
               p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y) + pi.x {
                inside.toggle()
            }
            j = i
        }
        return inside
    }
}
