import Foundation
import Observation

/// One pass of the multi-pass prediction — records the patch/stride/level
/// settings plus which classes were "claimed" by this pass. Shown in the
/// Model panel so the user can see how the heatmap was generated.
struct PredictionPassInfo: Sendable, Hashable {
    let patchSizeLevel: Int
    let strideLevel: Int
    let level: Int32
    let levelDownsample: Double
    let classes: [String]
}

/// Holds the predictions for the most recently scanned annotation, plus the
/// rendering knobs the overlay reads. Lives at the `ContentView` level and is
/// cleared whenever a different slide is opened.
@Observable
final class PredictionStore: @unchecked Sendable {
    var predictions: [PatchPrediction] = []
    var sourceAnnotationID: UUID?
    var sourceAnnotationClass: String?
    /// Class label → color, populated when a run kicks off so the overlay
    /// can paint each predicted class without depending on the profile store.
    var classColors: [String: AnnotationColor] = [:]
    var perClassCount: [String: Int] = [:]
    var passes: [PredictionPassInfo] = []

    var isVisible: Bool = true
    /// Class labels the user has hidden from the heatmap. A label absent from
    /// this set is drawn; membership hides it. Empty = every class shown.
    var hiddenClasses: Set<String> = []
    /// Patches with max probability below this aren't drawn.
    var minProbability: Float = 0.0

    /// Whether a predicted class is currently drawn on the canvas.
    func isClassVisible(_ label: String) -> Bool {
        !hiddenClasses.contains(label)
    }

    func setClassVisible(_ label: String, _ visible: Bool) {
        if visible {
            hiddenClasses.remove(label)
        } else {
            hiddenClasses.insert(label)
        }
    }

    /// The set of distinct predicted class labels, sorted.
    var predictedLabels: [String] {
        Set(predictions.map(\.predictedLabel)).sorted()
    }

    func showAllClasses() {
        hiddenClasses.removeAll()
    }

    func hideAllClasses() {
        hiddenClasses = Set(predictions.map(\.predictedLabel))
    }

    var isPredicting: Bool = false
    var progressCurrent: Int = 0
    var progressTotal: Int = 0

    var progressFraction: Double {
        progressTotal > 0 ? Double(progressCurrent) / Double(progressTotal) : 0
    }

    func reset() {
        predictions = []
        sourceAnnotationID = nil
        sourceAnnotationClass = nil
        perClassCount = [:]
        hiddenClasses = []
        passes = []
        isPredicting = false
        progressCurrent = 0
        progressTotal = 0
    }

    func recomputeStats() {
        perClassCount = Dictionary(grouping: predictions, by: \.predictedLabel)
            .mapValues { $0.count }
    }
}
