import Foundation

/// One classifier output for one patch sampled from a target annotation.
/// Coordinates are in level-0 slide-data-Y space — same convention as
/// `MLPatch`, so the heatmap drawing path mirrors back exactly like the
/// existing overlay code expects.
///
/// `passPatchSize` and `passStride` record which multi-pass configuration
/// produced this prediction, so the Model panel can show a per-pass breakdown.
struct PatchPrediction: Identifiable, Sendable {
    let id: UUID
    let dataX: Int64
    let dataY: Int64
    /// Patch side length in level-0 slide pixels (square).
    let sizeLevel0: Int
    let predictedLabel: String
    let probabilities: [String: Float]
    let maxProbability: Float
    let passPatchSize: Int
    let passStride: Int

    init(
        id: UUID = UUID(),
        dataX: Int64,
        dataY: Int64,
        sizeLevel0: Int,
        predictedLabel: String,
        probabilities: [String: Float],
        maxProbability: Float,
        passPatchSize: Int,
        passStride: Int
    ) {
        self.id = id
        self.dataX = dataX
        self.dataY = dataY
        self.sizeLevel0 = sizeLevel0
        self.predictedLabel = predictedLabel
        self.probabilities = probabilities
        self.maxProbability = maxProbability
        self.passPatchSize = passPatchSize
        self.passStride = passStride
    }
}
