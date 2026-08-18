import Foundation

struct Classification: Identifiable, Hashable, Codable, Sendable {
    var id: UUID
    var name: String
    var color: AnnotationColor
    /// When true, annotations of this class are treated as a "null / exclude"
    /// reference (e.g. PaNIN lumens): their patches are never learned as a real
    /// class, and candidate patches that look like them are nullified from
    /// training and prediction.
    var isNull: Bool

    init(id: UUID = UUID(), name: String, color: AnnotationColor, isNull: Bool = false) {
        self.id = id
        self.name = name
        self.color = color
        self.isNull = isNull
    }

    // Custom decode so profiles saved before `isNull` existed still load.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        color = try c.decode(AnnotationColor.self, forKey: .color)
        isNull = try c.decodeIfPresent(Bool.self, forKey: .isNull) ?? false
    }
}

struct ClassificationProfile: Codable, Sendable {
    var name: String
    var classes: [Classification]

    static let `default` = ClassificationProfile(
        name: "Pancreatic Pathology",
        classes: [
            Classification(name: "PaNIN-1", color: AnnotationColor(r: 80,  g: 180, b: 80)),
            Classification(name: "PaNIN-2", color: AnnotationColor(r: 240, g: 180, b: 30)),
            Classification(name: "PaNIN-3", color: AnnotationColor(r: 220, g: 60,  b: 60)),
            Classification(name: "Normal",  color: AnnotationColor(r: 80,  g: 160, b: 220)),
            Classification(name: "Stroma",  color: AnnotationColor(r: 180, g: 130, b: 200)),
        ]
    )
}
