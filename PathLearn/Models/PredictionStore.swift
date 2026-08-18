import Foundation
import Observation

/// One pass of the multi-pass prediction — records the patch/stride/level
/// settings plus which classes were "claimed" by this pass. Shown in the
/// Model panel so the user can see how the heatmap was generated.
struct PredictionPassInfo: Sendable, Hashable, Codable {
    let patchSizeLevel: Int
    let strideLevel: Int
    let level: Int32
    let levelDownsample: Double
    let classes: [String]
}

/// The complete prediction "module" for a single annotation — everything the
/// Classifier panel and the canvas overlay need to render that annotation's
/// analysis. Persisted per annotation so each one carries its own result.
struct PredictionResult: Sendable, Codable {
    var predictions: [PatchPrediction]
    var passes: [PredictionPassInfo]
    var perClassCount: [String: Int]
    /// Class label → color for painting the heatmap and legend swatches.
    var classColors: [String: AnnotationColor]
    /// Classification of the annotation this was run on (for the headline).
    var sourceAnnotationClass: String?
    /// Live display prefs the user can tweak — kept with the result so they
    /// survive selection changes and reloads.
    var minProbability: Float
    var hiddenClasses: Set<String>
    var createdAt: Date
}

/// Holds prediction results keyed by annotation, so selecting an annotation in
/// the sidebar shows its own analysis. Results persist to a companion file next
/// to the slide (`<slide>.predictions.json`) so they survive reopening. Lives at
/// the `ContentView` level; `bind`/`unbind` follow the open slide.
@Observable
final class PredictionStore: @unchecked Sendable {
    /// All saved results, keyed by annotation ID.
    private(set) var results: [UUID: PredictionResult] = [:]
    /// The annotation whose result is currently displayed (mirrors selection).
    var displayedID: UUID?

    /// Global on/off for painting the heatmap on the canvas (not per-annotation).
    var isVisible: Bool = true

    // Transient run state — only one prediction runs at a time.
    private var running: Bool = false
    private var runningID: UUID?
    var progressCurrent: Int = 0
    var progressTotal: Int = 0

    private var sidecarURL: URL?

    // MARK: Displayed-result accessors

    var displayedResult: PredictionResult? {
        displayedID.flatMap { results[$0] }
    }

    var predictions: [PatchPrediction] { displayedResult?.predictions ?? [] }
    var passes: [PredictionPassInfo] { displayedResult?.passes ?? [] }
    var perClassCount: [String: Int] { displayedResult?.perClassCount ?? [:] }
    var classColors: [String: AnnotationColor] { displayedResult?.classColors ?? [:] }
    var sourceAnnotationClass: String? { displayedResult?.sourceAnnotationClass }
    var hiddenClasses: Set<String> { displayedResult?.hiddenClasses ?? [] }

    /// Show "Running…" only when the in-flight run is for the displayed annotation.
    var isPredicting: Bool { running && runningID == displayedID }

    var minProbability: Float {
        get { displayedResult?.minProbability ?? 0 }
        set { mutateDisplayed(persist: false) { $0.minProbability = newValue } }
    }

    var progressFraction: Double {
        progressTotal > 0 ? Double(progressCurrent) / Double(progressTotal) : 0
    }

    /// The set of distinct predicted class labels for the displayed result.
    var predictedLabels: [String] {
        Set(predictions.map(\.predictedLabel)).sorted()
    }

    // MARK: Per-class visibility

    func isClassVisible(_ label: String) -> Bool {
        !(displayedResult?.hiddenClasses.contains(label) ?? false)
    }

    func setClassVisible(_ label: String, _ visible: Bool) {
        mutateDisplayed(persist: true) {
            if visible { $0.hiddenClasses.remove(label) }
            else { $0.hiddenClasses.insert(label) }
        }
    }

    func showAllClasses() {
        mutateDisplayed(persist: true) { $0.hiddenClasses.removeAll() }
    }

    func hideAllClasses() {
        mutateDisplayed(persist: true) { r in
            r.hiddenClasses = Set(r.predictions.map(\.predictedLabel))
        }
    }

    func setClassColor(_ label: String, _ color: AnnotationColor) {
        mutateDisplayed(persist: true) { $0.classColors[label] = color }
    }

    /// Persist after an interactive edit finishes (e.g. slider release).
    func commitEdits() { save() }

    // MARK: Run lifecycle

    func beginRun(annotationID: UUID) {
        running = true
        runningID = annotationID
        displayedID = annotationID
        progressCurrent = 0
        progressTotal = 0
    }

    func finishRun(_ result: PredictionResult, for id: UUID) {
        results[id] = result
        displayedID = id
        running = false
        runningID = nil
        save()
    }

    func failRun() {
        running = false
        runningID = nil
    }

    /// Clear the displayed annotation's stored result (trash / Clear Predictions).
    func reset() {
        guard let id = displayedID else { return }
        results[id] = nil
        save()
    }

    // MARK: Slide binding & persistence

    func bind(toSlideURL slideURL: URL) {
        let folder = slideURL.deletingLastPathComponent()
        let base = slideURL.deletingPathExtension().lastPathComponent
        sidecarURL = folder.appending(path: "\(base).predictions.json")
        displayedID = nil
        isVisible = true
        running = false
        runningID = nil
        progressCurrent = 0
        progressTotal = 0
        load()
    }

    /// Detach from the open slide — flush results, then clear in-memory state.
    func unbind() {
        save()
        results = [:]
        displayedID = nil
        running = false
        runningID = nil
        sidecarURL = nil
    }

    private func mutateDisplayed(persist: Bool,
                                 _ mutate: (inout PredictionResult) -> Void) {
        guard let id = displayedID, var r = results[id] else { return }
        mutate(&r)
        results[id] = r
        if persist { save() }
    }

    private func load() {
        results = [:]
        guard let url = sidecarURL,
              FileManager.default.fileExists(atPath: url.path) else { return }
        do {
            let data = try Data(contentsOf: url)
            let decoded = try JSONDecoder().decode([String: PredictionResult].self, from: data)
            var map: [UUID: PredictionResult] = [:]
            for (key, value) in decoded {
                if let u = UUID(uuidString: key) { map[u] = value }
            }
            results = map
            print("[predict] loaded \(results.count) saved result(s) from \(url.lastPathComponent)")
        } catch {
            print("[predict] failed to load results:", error)
        }
    }

    private func save() {
        guard let url = sidecarURL else { return }
        do {
            if results.isEmpty {
                // No results left — remove a stale companion file if present.
                if FileManager.default.fileExists(atPath: url.path) {
                    try FileManager.default.removeItem(at: url)
                }
                return
            }
            var out: [String: PredictionResult] = [:]
            for (key, value) in results { out[key.uuidString] = value }
            let data = try JSONEncoder().encode(out)
            try data.write(to: url, options: [.atomic])
        } catch {
            print("[predict] failed to save results:", error)
        }
    }
}
