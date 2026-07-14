import Foundation
import CoreGraphics

struct AnnotationColor: Hashable, Sendable, Codable {
    var r: Int
    var g: Int
    var b: Int

    static let defaultColor = AnnotationColor(r: 200, g: 60, b: 60)
}

struct Annotation: Identifiable, Hashable, Sendable {
    let id: UUID
    var points: [CGPoint]        // base-level slide pixel coordinates
    var classification: String
    var color: AnnotationColor
    var name: String?
    var isVisible: Bool

    init(id: UUID = UUID(),
         points: [CGPoint],
         classification: String,
         color: AnnotationColor,
         name: String? = nil,
         isVisible: Bool = true) {
        self.id = id
        self.points = points
        self.classification = classification
        self.color = color
        self.name = name
        self.isVisible = isVisible
    }

    /// What to show in tables/sidebars when no explicit name is set.
    var displayName: String { name ?? "" }

    /// Area enclosed by the polygon, in slide pixels squared (level-0).
    /// Uses the shoelace formula; assumes points form a simple polygon.
    var areaInSlidePixels: Double {
        guard points.count >= 3 else { return 0 }
        var sum: Double = 0
        for i in 0..<points.count {
            let j = (i + 1) % points.count
            sum += Double(points[i].x) * Double(points[j].y)
            sum -= Double(points[j].x) * Double(points[i].y)
        }
        return abs(sum) / 2.0
    }

    /// Axis-aligned bounding box of the polygon in slide-pixel coordinates.
    var boundingBoxInSlidePixels: CGRect {
        guard let first = points.first else { return .zero }
        var minX = first.x, maxX = first.x
        var minY = first.y, maxY = first.y
        for p in points.dropFirst() {
            minX = min(minX, p.x); maxX = max(maxX, p.x)
            minY = min(minY, p.y); maxY = max(maxY, p.y)
        }
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    /// Compact string like "1.2M" / "340K" / "120" for table display.
    var areaShort: String {
        let a = areaInSlidePixels
        if a >= 1_000_000 { return String(format: "%.1fM", a / 1_000_000) }
        if a >= 1_000     { return String(format: "%.0fK", a / 1_000) }
        return String(format: "%.0f", a)
    }
}
