import Foundation
import Vision
import CoreGraphics

/// Wraps Apple's built-in `VNGenerateImageFeaturePrintRequest`. No external
/// model file needed; runs on the Neural Engine when available.
/// Nonisolated so it can be called from a `Task.detached`.
nonisolated final class FeatureExtractor: AnyFeatureExtractor, @unchecked Sendable {
    let identity: ExtractorIdentity
    let inputSize: CGSize = CGSize(width: 299, height: 299)
    /// Settled to the real output size on first `extract`. Vision's typical
    /// output is 2048 floats on macOS 14+.
    private(set) var featureDim: Int = 2048

    enum ExtractError: Error, CustomStringConvertible {
        case noObservation
        var description: String {
            switch self {
            case .noObservation: return "Vision returned no feature print observation."
            }
        }
    }

    init(revision: Int? = nil) {
        let rev = revision ?? VNGenerateImageFeaturePrintRequest.currentRevision
        self.identity = .legacyVision(revision: rev)
    }

    func extract(_ image: CGImage) async throws -> FeatureExtractionResult {
        let request = VNGenerateImageFeaturePrintRequest()
        request.revision = identity.revision
        request.imageCropAndScaleOption = .centerCrop

        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try handler.perform([request])

        guard let obs = request.results?.first else {
            throw ExtractError.noObservation
        }
        self.featureDim = obs.elementCount
        return FeatureExtractionResult(
            data: obs.data,
            dim: obs.elementCount,
            elementType: Int(obs.elementType.rawValue),
            identity: identity
        )
    }
}
