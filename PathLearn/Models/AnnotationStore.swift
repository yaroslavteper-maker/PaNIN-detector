import Foundation
import Observation

@Observable
final class AnnotationStore {
    var annotations: [Annotation] = []
    var selectedID: UUID?
    var currentLabel: String = "PaNIN-1"
    var currentColor: AnnotationColor = .defaultColor

    private(set) var sidecarURL: URL?

    /// `sidecarSuffix` lets a second store (e.g. predicted regions) persist to a
    /// separate companion file so it doesn't collide with the manual `.geojson`.
    func bind(toSlideURL slideURL: URL, sidecarSuffix: String = "geojson") {
        selectedID = nil
        let folder = slideURL.deletingLastPathComponent()
        let base = slideURL.deletingPathExtension().lastPathComponent
        sidecarURL = folder.appending(path: "\(base).\(sidecarSuffix)")
        loadIfPresent()
    }

    /// Detach from any open slide — flushes one last save then clears the
    /// in-memory annotations and the sidecar binding. Used by the Close Slide
    /// command so the next slide doesn't inherit stale state.
    func unbind() {
        save()
        annotations = []
        selectedID = nil
        sidecarURL = nil
    }

    func loadIfPresent() {
        guard let url = sidecarURL, FileManager.default.fileExists(atPath: url.path) else {
            annotations = []
            return
        }
        do {
            let data = try Data(contentsOf: url)
            annotations = try GeoJSON.decode(data)
        } catch {
            print("Failed to load annotations:", error)
            annotations = []
        }
    }

    func save() {
        guard let url = sidecarURL else { return }
        do {
            let data = try GeoJSON.encode(annotations)
            try data.write(to: url, options: [.atomic])
        } catch {
            print("Failed to save annotations:", error)
        }
    }

    func add(_ ann: Annotation) {
        annotations.append(ann)
        selectedID = ann.id
        print("[store] added \(ann.classification) [\(ann.points.count) pts] total=\(annotations.count)")
        save()
    }

    /// Append many annotations at once, saving the sidecar a single time.
    /// Used when materializing predicted regions into persistent annotations.
    func addBatch(_ incoming: [Annotation]) {
        guard !incoming.isEmpty else { return }
        annotations.append(contentsOf: incoming)
        print("[store] added batch of \(incoming.count), total=\(annotations.count)")
        save()
    }

    func update(_ ann: Annotation) {
        guard let i = annotations.firstIndex(where: { $0.id == ann.id }) else { return }
        annotations[i] = ann
        save()
    }

    func remove(id: UUID) {
        annotations.removeAll { $0.id == id }
        if selectedID == id { selectedID = nil }
        save()
    }

    /// Count of annotations whose visibility box is checked.
    var checkedCount: Int { annotations.lazy.filter { $0.isVisible }.count }

    /// Delete every annotation whose visibility box is checked. Lets the user
    /// uncheck the few they want to keep, then clear the rest in one action.
    func removeChecked() {
        let before = annotations.count
        annotations.removeAll { $0.isVisible }
        if let s = selectedID, !annotations.contains(where: { $0.id == s }) { selectedID = nil }
        print("[store] removed \(before - annotations.count) checked annotation(s); total=\(annotations.count)")
        save()
    }

    /// Delete every annotation on the slide and persist the empty sidecar.
    func removeAll() {
        let before = annotations.count
        annotations.removeAll()
        selectedID = nil
        print("[store] cleared all \(before) annotation(s)")
        save()
    }

    func rename(_ id: UUID, to name: String?) {
        guard let i = annotations.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = name?.trimmingCharacters(in: .whitespaces)
        annotations[i].name = (trimmed?.isEmpty ?? true) ? nil : trimmed
        save()
    }

    func setClassification(_ id: UUID, name: String, color: AnnotationColor) {
        guard let i = annotations.firstIndex(where: { $0.id == id }) else { return }
        annotations[i].classification = name
        annotations[i].color = color
        save()
    }

    func setVisibility(_ id: UUID, visible: Bool) {
        guard let i = annotations.firstIndex(where: { $0.id == id }) else { return }
        annotations[i].isVisible = visible
        save()
    }

    var existingLabels: [String] {
        Array(Set(annotations.map(\.classification))).sorted()
    }

    /// Merge a batch of annotations decoded from another GeoJSON source —
    /// typically a QuPath export. In `replace` mode the current set is
    /// discarded; in append mode the incoming polygons are added with fresh
    /// UUIDs so they don't collide with existing rows. The sidecar is saved
    /// after the merge so the imports survive a slide close.
    func importMerge(_ incoming: [Annotation], replace: Bool) {
        if replace {
            annotations = incoming
            selectedID = nil
        } else {
            let regenerated = incoming.map { ann in
                Annotation(
                    id: UUID(),
                    points: ann.points,
                    classification: ann.classification,
                    color: ann.color,
                    name: ann.name,
                    isVisible: ann.isVisible
                )
            }
            annotations.append(contentsOf: regenerated)
        }
        save()
    }
}
