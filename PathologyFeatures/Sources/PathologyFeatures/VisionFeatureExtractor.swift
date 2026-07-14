import Foundation
import Vision
import CoreGraphics

/// Wraps Apple's built-in `VNGenerateImageFeaturePrintRequest`.
/// Requires no external model file; runs on the Neural Engine when available.
/// Good baseline; outperformed on pathology-specific tasks by domain encoders
/// like UNI / Virchow / CTransPath.
public final class VisionFeatureExtractor: FeatureExtractor, @unchecked Sendable {
    public let identity: ExtractorIdentity
    public let inputSize: CGSize
    public private(set) var featureDim: Int

    public init(revision: Int? = nil) {
        let rev = revision ?? VNGenerateImageFeaturePrintRequest.currentRevision
        self.identity = ExtractorIdentity(kind: "vision", name: "vision", revision: rev)
        // Apple resizes internally; this is a hint only.
        self.inputSize = CGSize(width: 299, height: 299)
        // Determined on first `extract`; commonly 2048 on macOS 14+.
        self.featureDim = 2048
    }

    public func extract(_ image: CGImage) async throws -> FeatureResult {
        let request = VNGenerateImageFeaturePrintRequest()
        request.revision = identity.revision
        request.imageCropAndScaleOption = .centerCrop

        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try handler.perform([request])

        guard let obs = request.results?.first else {
            throw ExtractError.noObservation
        }
        let elementType = ElementType(rawValue: Int(obs.elementType.rawValue)) ?? .unknown
        // Update on first run in case Vision's revision changes the output size.
        self.featureDim = obs.elementCount
        return FeatureResult(
            data: obs.data,
            dim: obs.elementCount,
            elementType: elementType
        )
    }
}
