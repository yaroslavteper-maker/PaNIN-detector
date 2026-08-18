import Foundation
import Observation

/// Accumulating bank of per-annotation geometry descriptors, persisted as a
/// single JSON file in Application Support so it survives across slides and
/// launches — mirroring how the patch bank accumulates, but self-contained
/// (no SwiftData schema change). Records are keyed by (slidePath, annotationID)
/// so re-extracting an annotation replaces its row.
@Observable
final class GeometryFeatureStore: @unchecked Sendable {

    struct Record: Codable, Sendable {
        var slidePath: String
        var slideName: String
        var annotationID: UUID
        var classification: String
        var features: [Float]
        var version: Int
    }

    private(set) var records: [Record] = []

    // Transient progress state (read by the UI).
    var isExtracting = false
    var extractCurrent = 0
    var extractTotal = 0
    var isTraining = false
    var trainingStatus = ""

    var total: Int { records.count }

    var perClass: [String: Int] {
        var out: [String: Int] = [:]
        for r in records { out[r.classification, default: 0] += 1 }
        return out
    }

    var perSlide: [String: Int] {
        var out: [String: Int] = [:]
        for r in records { out[r.slideName, default: 0] += 1 }
        return out
    }

    /// Records computed with an outdated descriptor version — must be
    /// re-extracted before they can be trained on.
    var staleCount: Int {
        records.filter { $0.version != PathologyGeometry.version }.count
    }

    /// Location of the JSON bank on disk (for Reveal in Finder).
    var bankURL: URL? { fileURL }

    /// Re-read the bank from disk (e.g. the sidebar Refresh button).
    func reload() { load() }

    init() { load() }

    /// Insert or replace records, matching on (slidePath, annotationID).
    func add(_ incoming: [Record]) {
        for rec in incoming {
            if let i = records.firstIndex(where: {
                $0.slidePath == rec.slidePath && $0.annotationID == rec.annotationID
            }) {
                records[i] = rec
            } else {
                records.append(rec)
            }
        }
        save()
    }

    func clear() {
        records = []
        save()
    }

    /// Remove every record for a given class label.
    func removeClass(_ label: String) {
        records.removeAll { $0.classification == label }
        save()
    }

    // MARK: - Persistence

    private var fileURL: URL? {
        guard let base = FileManager.default.urls(for: .applicationSupportDirectory,
                                                  in: .userDomainMask).first else { return nil }
        let dir = base.appending(path: "PathLearn", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appending(path: "geometry_bank.json")
    }

    private func load() {
        guard let url = fileURL, FileManager.default.fileExists(atPath: url.path) else { return }
        do {
            let data = try Data(contentsOf: url)
            records = try JSONDecoder().decode([Record].self, from: data)
            print("[geometry] loaded \(records.count) record(s)")
        } catch {
            print("[geometry] failed to load bank:", error)
        }
    }

    private func save() {
        guard let url = fileURL else { return }
        do {
            let data = try JSONEncoder().encode(records)
            try data.write(to: url, options: [.atomic])
        } catch {
            print("[geometry] failed to save bank:", error)
        }
    }
}
