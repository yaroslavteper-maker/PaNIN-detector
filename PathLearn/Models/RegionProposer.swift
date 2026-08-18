import Foundation
import CoreGraphics

/// Groups a per-patch prediction's tiles into contiguous lesion regions and
/// emits a polygon outline for each — the "region proposal" step of the
/// detect-then-grade cascade. Tiles whose predicted label passes `isCandidate`
/// (e.g. any PanIN grade) are snapped to a grid, 4-connected-component labeled,
/// and traced into an orthogonal polygon ready to become an `Annotation`.
///
/// Output polygons are in level-0 **overlay-Y** coordinates (same space as
/// `Annotation.points`), so geometry grading and drawing line up.
nonisolated enum RegionProposer {

    struct Region: Sendable {
        /// Closed polygon in level-0 overlay-Y coordinates.
        let polygonOverlayY: [CGPoint]
        let tileCount: Int
    }

    static func propose(predictions: [PatchPrediction],
                        slideHeight: CGFloat,
                        isCandidate: (String) -> Bool,
                        minTiles: Int = 2) -> [Region] {
        let cands = predictions.filter { isCandidate($0.predictedLabel) }
        guard let first = cands.first else { return [] }

        // Grid pitch in level-0 px: stride·(sizeLevel0/patchSize).
        let pitch: Double = {
            guard first.passPatchSize > 0 else { return Double(first.sizeLevel0) }
            return Double(first.passStride) * Double(first.sizeLevel0) / Double(first.passPatchSize)
        }()
        guard pitch > 0 else { return [] }

        let minX = cands.map { Double($0.dataX) }.min()!
        let minY = cands.map { Double($0.dataY) }.min()!

        func key(_ gx: Int, _ gy: Int) -> Int64 { (Int64(gx) << 32) | (Int64(gy) & 0xffff_ffff) }
        var occupied = Set<Int64>()
        for p in cands {
            let gx = Int(((Double(p.dataX) - minX) / pitch).rounded())
            let gy = Int(((Double(p.dataY) - minY) / pitch).rounded())
            occupied.insert(key(gx, gy))
        }
        func decode(_ k: Int64) -> (Int, Int) { (Int(k >> 32), Int(Int32(truncatingIfNeeded: k))) }

        // 4-connected components.
        var visited = Set<Int64>()
        var regions: [Region] = []
        for start in occupied where !visited.contains(start) {
            var comp: [(Int, Int)] = []
            var stack = [start]
            visited.insert(start)
            while let k = stack.popLast() {
                let (gx, gy) = decode(k)
                comp.append((gx, gy))
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let nk = key(gx + dx, gy + dy)
                    if occupied.contains(nk), !visited.contains(nk) {
                        visited.insert(nk); stack.append(nk)
                    }
                }
            }
            guard comp.count >= minTiles else { continue }
            let cells = Set(comp.map { key($0.0, $0.1) })
            guard let loop = outline(cells: cells, keyOf: key, decode: decode) else { continue }
            // Grid corners → level-0 data → overlay-Y.
            let poly = loop.map { corner -> CGPoint in
                let dataX = minX + Double(corner.0) * pitch
                let dataY = minY + Double(corner.1) * pitch
                return CGPoint(x: dataX, y: Double(slideHeight) - dataY)
            }
            if poly.count >= 3 {
                regions.append(Region(polygonOverlayY: poly, tileCount: comp.count))
            }
        }
        // Largest lesions first.
        return regions.sorted { $0.tileCount > $1.tileCount }
    }

    /// Trace the orthogonal boundary of a 4-connected cell set into a single
    /// closed corner loop (the largest, ignoring holes). Cell (x,y) spans grid
    /// corners (x,y)…(x+1,y+1).
    private static func outline(cells: Set<Int64>,
                                keyOf: (Int, Int) -> Int64,
                                decode: (Int64) -> (Int, Int)) -> [(Int, Int)]? {
        // Directed boundary edges, interior kept consistently to one side so
        // they chain into loops. Corner encoded as key.
        func ckey(_ x: Int, _ y: Int) -> Int64 { (Int64(x) << 32) | (Int64(y) & 0xffff_ffff) }
        var next: [Int64: (Int, Int)] = [:]     // start corner → end corner
        func addEdge(_ ax: Int, _ ay: Int, _ bx: Int, _ by: Int) { next[ckey(ax, ay)] = (bx, by) }

        for k in cells {
            let (x, y) = decode(k)
            if !cells.contains(keyOf(x, y - 1)) { addEdge(x, y, x + 1, y) }         // top  →
            if !cells.contains(keyOf(x + 1, y)) { addEdge(x + 1, y, x + 1, y + 1) } // right ↓
            if !cells.contains(keyOf(x, y + 1)) { addEdge(x + 1, y + 1, x, y + 1) } // bottom ←
            if !cells.contains(keyOf(x - 1, y)) { addEdge(x, y + 1, x, y) }         // left ↑
        }
        guard !next.isEmpty else { return nil }

        // Walk edges into loops; keep the one with the largest |area|.
        var used = Set<Int64>()
        var best: [(Int, Int)]? = nil
        var bestArea = -1.0
        for startKey in next.keys where !used.contains(startKey) {
            var loop: [(Int, Int)] = []
            var curKey = startKey
            var guardCount = 0
            while let end = next[curKey], !used.contains(curKey), guardCount <= next.count {
                used.insert(curKey)
                let (sx, sy) = decode(curKey)
                loop.append((sx, sy))
                curKey = ckey(end.0, end.1)
                guardCount += 1
            }
            guard loop.count >= 4 else { continue }
            let simplified = removeCollinear(loop)
            let area = abs(shoelace(simplified))
            if area > bestArea { bestArea = area; best = simplified }
        }
        return best
    }

    private static func removeCollinear(_ pts: [(Int, Int)]) -> [(Int, Int)] {
        guard pts.count > 2 else { return pts }
        var out: [(Int, Int)] = []
        let n = pts.count
        for i in 0..<n {
            let a = pts[(i - 1 + n) % n], b = pts[i], c = pts[(i + 1) % n]
            let cross = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
            if cross != 0 { out.append(b) }   // keep only corners
        }
        return out.count >= 3 ? out : pts
    }

    private static func shoelace(_ pts: [(Int, Int)]) -> Double {
        var s = 0.0
        let n = pts.count
        for i in 0..<n {
            let a = pts[i], b = pts[(i + 1) % n]
            s += Double(a.0 * b.1 - b.0 * a.1)
        }
        return s / 2.0
    }
}
