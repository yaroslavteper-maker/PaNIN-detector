import Foundation

/// Snapshot of a single training run's quality numbers.
struct TrainingMetrics: Codable, Sendable {
    let trainAccuracy: Float
    let valAccuracy: Float
    /// Keys are class labels; values in [0, 1].
    let perClassPrecision: [String: Float]
    let perClassRecall: [String: Float]
    let perClassF1: [String: Float]
    /// Square K×K matrix; `confusionMatrix[trueIdx][predIdx] = count`.
    let confusionMatrix: [[Int]]
    /// Ordered class labels (indices match the confusion matrix).
    let classLabels: [String]
    let trainCount: Int
    let valCount: Int
    let finalLoss: Float
    let trainedAt: Date
    let featureExtractorRevision: Int
    let iterations: Int
}
