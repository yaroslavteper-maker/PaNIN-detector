import Foundation

/// One connected region found in a boolean mask.
struct Blob: Sendable {
    let area: Int          // pixel count
    let cx: Double         // centroid x
    let cy: Double         // centroid y
    let perimeter: Int     // boundary pixel count (touches non-mask or edge)
}

/// Iterative flood-fill connected-components labeling over a row-major boolean
/// mask. Returns blobs with area, centroid, and perimeter. Used for both nuclei
/// (hematoxylin mask) and lumina (whitespace mask).
nonisolated enum ConnectedComponents {

    static func label(mask: [Bool],
                      width: Int,
                      height: Int,
                      minArea: Int = 1) -> [Blob] {
        guard width > 0, height > 0, mask.count == width * height else { return [] }
        var visited = [Bool](repeating: false, count: mask.count)
        var blobs: [Blob] = []
        var stack: [Int] = []

        for start in 0..<mask.count {
            if !mask[start] || visited[start] { continue }
            visited[start] = true
            stack.removeAll(keepingCapacity: true)
            stack.append(start)

            var area = 0
            var sumX = 0.0
            var sumY = 0.0
            var perimeter = 0

            while let idx = stack.popLast() {
                let x = idx % width
                let y = idx / width
                area += 1
                sumX += Double(x)
                sumY += Double(y)

                // 4-neighborhood: perimeter + traversal.
                var isBoundary = false
                // left
                if x > 0 {
                    let n = idx - 1
                    if mask[n] { if !visited[n] { visited[n] = true; stack.append(n) } }
                    else { isBoundary = true }
                } else { isBoundary = true }
                // right
                if x < width - 1 {
                    let n = idx + 1
                    if mask[n] { if !visited[n] { visited[n] = true; stack.append(n) } }
                    else { isBoundary = true }
                } else { isBoundary = true }
                // up
                if y > 0 {
                    let n = idx - width
                    if mask[n] { if !visited[n] { visited[n] = true; stack.append(n) } }
                    else { isBoundary = true }
                } else { isBoundary = true }
                // down
                if y < height - 1 {
                    let n = idx + width
                    if mask[n] { if !visited[n] { visited[n] = true; stack.append(n) } }
                    else { isBoundary = true }
                } else { isBoundary = true }

                if isBoundary { perimeter += 1 }
            }

            if area >= minArea {
                blobs.append(Blob(area: area,
                                  cx: sumX / Double(area),
                                  cy: sumY / Double(area),
                                  perimeter: max(perimeter, 1)))
            }
        }
        return blobs
    }
}
