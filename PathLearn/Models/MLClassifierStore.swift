import Foundation
import Observation

/// Holds two classifier slots — one logistic-regression and one centroid-
/// derived — so the user can keep both around and flip which one is "active"
/// at predict time. The legacy `classifier` property is a computed view onto
/// whichever slot `activeKind` currently points at, so existing call sites
/// keep working unchanged.
@Observable
final class MLClassifierStore: @unchecked Sendable {
    var logisticClassifier: MLClassifier?
    var centroidClassifier: MLClassifier?
    var activeKind: MLClassifier.Kind = .logistic

    /// Live-training fields (still per-store rather than per-slot — there's
    /// only ever one logistic-regression run in flight).
    var isTraining: Bool = false
    var trainingProgress: Float = 0
    var trainingStatus: String = ""

    /// Source-of-truth URL the *active* classifier was last saved to / loaded
    /// from, used for "Save" vs "Save As…" flows. Reset whenever activeKind
    /// changes since the saved-file may not match the new slot.
    var lastFileURL: URL?

    /// Active classifier. Reading routes by `activeKind`. Writing auto-routes
    /// to the slot matching the new model's `classifierKind` and updates
    /// `activeKind` to match — "the model I just set is the one I want to use".
    var classifier: MLClassifier? {
        get {
            switch activeKind {
            case .logistic: return logisticClassifier
            case .centroid: return centroidClassifier
            }
        }
        set {
            guard let newValue else {
                // Clear the currently-active slot.
                switch activeKind {
                case .logistic: logisticClassifier = nil
                case .centroid: centroidClassifier = nil
                }
                lastFileURL = nil
                return
            }
            let kind = newValue.classifierKind
            switch kind {
            case .logistic: logisticClassifier = newValue
            case .centroid: centroidClassifier = newValue
            }
            if activeKind != kind {
                activeKind = kind
                // Different slot → previous lastFileURL no longer relevant.
                lastFileURL = nil
            }
        }
    }

    /// Convenience for views: how many of the two slots are populated.
    var slotsFilled: Int {
        (logisticClassifier == nil ? 0 : 1) + (centroidClassifier == nil ? 0 : 1)
    }

    /// Flip the active classifier kind. Resets `lastFileURL` since the new
    /// active model may not match what was last saved.
    func setActive(_ kind: MLClassifier.Kind) {
        guard kind != activeKind else { return }
        activeKind = kind
        lastFileURL = nil
    }
}
