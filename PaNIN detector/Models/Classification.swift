import Foundation

struct Classification: Identifiable, Hashable, Codable, Sendable {
    var id: UUID
    var name: String
    var color: AnnotationColor

    init(id: UUID = UUID(), name: String, color: AnnotationColor) {
        self.id = id
        self.name = name
        self.color = color
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
