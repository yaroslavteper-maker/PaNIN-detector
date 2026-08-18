import Foundation
import CoreGraphics

/// Encodes/decodes annotations as a QuPath-dialect GeoJSON FeatureCollection.
/// Coordinates are written as `[x, y]` pairs in base-level slide pixels.
/// `properties.classification = { name, color: [r,g,b] }` matches QuPath.
/// We also store `properties.name` (the annotation's display name) and
/// `properties.isVisible` (false to hide on the canvas); QuPath ignores both.
enum GeoJSON {
    static func encode(_ annotations: [Annotation]) throws -> Data {
        let features: [[String: Any]] = annotations.map { ann in
            var ring: [[Double]] = ann.points.map { [Double($0.x), Double($0.y)] }
            if let first = ring.first, ring.count >= 2, ring.last != first {
                ring.append(first)
            }
            var properties: [String: Any] = [
                "objectType": "annotation",
                "classification": [
                    "name": ann.classification,
                    "color": [ann.color.r, ann.color.g, ann.color.b]
                ]
            ]
            if let name = ann.name, !name.isEmpty {
                properties["name"] = name
            }
            if !ann.isVisible {
                properties["isVisible"] = false
            }
            return [
                "type": "Feature",
                "id": ann.id.uuidString,
                "geometry": [
                    "type": "Polygon",
                    "coordinates": [ring]
                ],
                "properties": properties
            ]
        }
        let fc: [String: Any] = [
            "type": "FeatureCollection",
            "features": features
        ]
        return try JSONSerialization.data(
            withJSONObject: fc,
            options: [.prettyPrinted, .sortedKeys]
        )
    }

    static func decode(_ data: Data) throws -> [Annotation] {
        let obj = try JSONSerialization.jsonObject(with: data)
        guard let dict = obj as? [String: Any] else { return [] }
        let features = (dict["features"] as? [[String: Any]]) ?? []

        var result: [Annotation] = []
        for f in features {
            guard let geom = f["geometry"] as? [String: Any] else { continue }
            let kind = geom["type"] as? String ?? ""
            var rings: [[[Double]]] = []
            if kind == "Polygon", let r = geom["coordinates"] as? [[[Double]]] {
                rings = r
            } else if kind == "MultiPolygon", let polys = geom["coordinates"] as? [[[[Double]]]] {
                rings = polys.flatMap { $0 }
            }
            guard let ring = rings.first else { continue }

            var points = ring.compactMap { pair -> CGPoint? in
                guard pair.count >= 2 else { return nil }
                return CGPoint(x: pair[0], y: pair[1])
            }
            if points.count > 1, points.first == points.last { points.removeLast() }
            guard points.count >= 3 else { continue }

            var classification = "Unlabeled"
            var color = AnnotationColor.defaultColor
            var name: String? = nil
            var isVisible = true
            if let props = f["properties"] as? [String: Any] {
                if let cls = props["classification"] as? [String: Any] {
                    if let n = cls["name"] as? String { classification = n }
                    if let c = cls["color"] as? [NSNumber], c.count >= 3 {
                        color = AnnotationColor(r: c[0].intValue, g: c[1].intValue, b: c[2].intValue)
                    }
                }
                if let n = props["name"] as? String, !n.isEmpty { name = n }
                if let v = props["isVisible"] as? Bool { isVisible = v }
            }

            var id = UUID()
            if let s = f["id"] as? String, let u = UUID(uuidString: s) { id = u }
            result.append(Annotation(
                id: id,
                points: points,
                classification: classification,
                color: color,
                name: name,
                isVisible: isVisible
            ))
        }
        return result
    }
}
